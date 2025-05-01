import os
import os
import torch
import requests
import argparse
from PIL import Image
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
import copy
import numpy as np
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def is_duplicate(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    if iou > 0.9 or (intersection_area / float(area_bbox1) > 0.9) or (intersection_area / float(area_bbox2) > 0.9):
        return 1
    return 0


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

class GDino:
    def __init__(self, args):

        ckpt_path = args.gdino_ckpt_path
        self.config_file = args.gdino_config_path

        self.model_checkpoint_path = ckpt_path
        self.box_threshold = 0.3
        self.text_threshold = None # we use token_spans instead
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # spatial info
        self.vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of', 'on the top of','on top of', 'right of', 'left of', 'below', 'above'] #locality words

        args = SLConfig.fromfile(self.config_file)
        args.device = 'cuda'
        self.model = build_model(args)

        checkpoint = torch.load(self.model_checkpoint_path)
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        for param in self.model.parameters():
            param.requires_grad = False
        
    @property
    def __name__(self):
        return 'GDino'

    def load_to_device(self, load_device):
        self.device = load_device
        self.model.to(load_device)
        self.model.eval()
        return self.model
    
    def get_grounding_output(self, image, caption, box_threshold, text_threshold=None, with_logits=True, token_spans=None):

        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = self.model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                self.model.tokenizer(caption),
                token_span=token_spans
            ).to(image.device) # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases, all_logits
    
    def make_prompt(self, nouns):
        token_spans = []
        pointer = 0
        for noun in nouns:
            n_split = noun.strip().split(" ")
            if len(n_split) == 1:
                length = len(n_split[0])
                token_spans.append([[pointer, pointer + length]])
                pointer += length + 3 # on the blank space after the noun
            else: # multiple words
                beg_len = len(n_split[0])
                total_length = len(noun)
                end_len = len(n_split[-1])
                token_spans.append([[pointer, pointer + beg_len], [pointer + total_length - end_len, pointer + total_length]])
                pointer += total_length + 3 # on the blank space after the noun
        text_prompt = ' . '.join(nouns) + "." # need to end with '.
        return text_prompt, token_spans

    def determine_position(self, locality, box1, box2, iou_threshold=0.1,distance_threshold=150):

        # Calculate centers of bounding boxes
        box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
        box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)

        # Calculate horizontal and vertical distances
        x_distance = box2_center[0] - box1_center[0]
        y_distance  = box2_center[1] - box1_center[1]

        # Calculate IoU
        x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
        y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
        intersection = x_overlap * y_overlap
        box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
        box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
        union = box1_area + box2_area - intersection
        iou = intersection / union

        # Determine position based on distances and IoU and give a soft score
        score=0
        if locality in ['next to', 'on side of', 'near']:
            if (abs(x_distance)< distance_threshold or abs(y_distance)< distance_threshold):
                score=1
            else:
                score=distance_threshold/max(abs(x_distance),abs(y_distance))
        elif locality in ['on the right of', 'right of']:
            if x_distance < 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score=1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
            else:
                score=0
        elif locality in ['on the left of', 'left of']:
            if x_distance > 0:
                if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                    score=1
                elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
            else:
                score=0
        elif locality in ['on the bottom of', 'below']:
            if y_distance < 0:
                if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                    score=1
                elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
        elif locality in ['on the top of', 'top of']:
            if y_distance > 0:
                if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                    score=1
                elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                    score=iou_threshold/iou
        else:
            score=0
        return score.item() if isinstance(score, torch.Tensor) else score

    def get_nms_boxes(self, boxes_filt, pred_phrases, all_logits, nouns, spatial_info):
        # do nms for each object
        all_logits = torch.cat(all_logits)
        obj1, obj2 = spatial_info['obj1'], spatial_info['obj2']

        obj1_boxes, obj2_boxes = [], []
        obj1_scores, obj2_scores = [], []
        for idx, p in enumerate(pred_phrases):
            if p == obj1:
                obj1_boxes.append(boxes_filt[idx])
                obj1_scores.append(all_logits[idx])
            elif p == obj2:
                obj2_boxes.append(boxes_filt[idx])
                obj2_scores.append(all_logits[idx])

        # select the most confident
        obj1_scores, obj2_scores = torch.stack(obj1_scores), torch.stack(obj2_scores)
        obj1_conf = torch.argmax(obj1_scores)
        obj2_conf = torch.argmax(obj2_scores)
        obj1_nms_boxes = obj1_boxes[obj1_conf][None]
        obj2_nms_boxes = obj2_boxes[obj2_conf][None]
   
        return obj1_nms_boxes, obj2_nms_boxes

    def get_spatial_score(self, boxes_filt, pred_phrases, all_logits, nouns, spatial_info):
        score = 0
        # determine if the two objects are in the image
        nouns = [spatial_info['obj1'], spatial_info['obj2']]
        for obj in nouns:
            if obj in pred_phrases:
                score += 0.2 # object appearance score
        # if all the objects are in the map, calculate the position score
        if score == 0.4:

            # convert boxes to xyxy
            boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
            boxes_filt[:, 2:] += boxes_filt[:, :2]

            # add nms for each object
            obj1_nms_boxes, obj2_nms_boxes = self.get_nms_boxes(boxes_filt, pred_phrases, all_logits, nouns, spatial_info)
            if obj1_nms_boxes.shape[0] == 1 and obj2_nms_boxes.shape[0] == 1: # only for nms
            # if True:
                obj1_nms_boxes, obj2_nms_boxes = obj1_nms_boxes[0], obj2_nms_boxes[0]
                box1, box2={},{}
                box1["x_min"] = obj1_nms_boxes[0]
                box1["y_min"] = obj1_nms_boxes[1]
                box1["x_max"] = obj1_nms_boxes[2]
                box1["y_max"] = obj1_nms_boxes[3]

                box2["x_min"] = obj2_nms_boxes[0]
                box2["y_min"] = obj2_nms_boxes[1]
                box2["x_max"] = obj2_nms_boxes[2]
                box2["y_max"] = obj2_nms_boxes[3]
                score += self.determine_position(spatial_info['locality'], box1, box2) / 2

        return score


    def get_numeracy_score(self, boxes_filt, pred_phrases, all_logits, nouns, numeracy_info):
        all_logits = torch.cat(all_logits)
        det_obj = defaultdict(list)

        score = 0.0
        # Normalize score
        weight = 1 / len(numeracy_info)

        for idx, num_item in enumerate(numeracy_info):
            expected_count, obj_name = num_item['num'], num_item['obj_name']
            # Get all detection boxes for current object
            detected_boxes = [boxes_filt[i] for i, phrase in enumerate(pred_phrases) if phrase == obj_name]
            
            if len(detected_boxes) == 0:
                continue

            detected_scores = [all_logits[i] for i, phrase in enumerate(pred_phrases) if phrase == obj_name]
            detected_boxes_index = nms(torch.stack(detected_boxes).to(self.device), torch.stack(detected_scores).to(self.device), 0.5)

            if detected_boxes_index.shape[0] == expected_count:
                score += 1 * weight
            else:
                score += 0.2 * weight

        return score

    def get_object_score(self, boxes_filt, pred_phrases, all_logits, nouns):
        weight = 1 / len(nouns)
        score = 0
        for noun in nouns:
            if noun in pred_phrases:
                score += weight
        return score
     
    def __call__(self, prompts, images, **kwargs):

        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        results = []
        for idx, image in enumerate(images):
            # if no nouns in the prompt, skip with reward is 1
            if len(kwargs['nouns'][idx]) == 0:
                results.append(1)
                continue

            image, _ = self.transform(image, None)
            image = image.to(device)

            text_prompt, token_spans = kwargs['det_prompt'][idx]['text_prompt'], kwargs['det_prompt'][idx]['token_spans']

            boxes_filt, pred_phrases, all_logits = self.get_grounding_output(
                image, text_prompt, self.box_threshold, self.text_threshold, token_spans=eval(f"{token_spans}")
            )

            # Three types of scores: 
            # 1. object
            # 2. position 
            # 3. numeracy
            if kwargs['task_type'][idx] == 'spatial':
                try:
                    score = self.get_spatial_score(boxes_filt, pred_phrases, all_logits, kwargs['nouns'][idx], kwargs['spatial_info'][idx])
                except:
                    print(f'FAILED!!!!!!!!spatial_info: {kwargs["spatial_info"][idx]}; pred_phrases: {pred_phrases}; nouns: {kwargs["nouns"][idx]}')
                    score = 0
            elif kwargs['task_type'][idx] == 'numeracy':
                score = self.get_numeracy_score(boxes_filt, pred_phrases, all_logits, kwargs['nouns'][idx], kwargs['numeracy_info'][idx])
            else:
                score = self.get_object_score(boxes_filt, pred_phrases, all_logits, kwargs['nouns'][idx])

            results.append(score)

        return results
