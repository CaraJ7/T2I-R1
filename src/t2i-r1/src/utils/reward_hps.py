import os
import os
import torch
import requests
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

class HPSv2:
    def __init__(self, args):
        self.ckpt_path = args.hps_ckpt_path

    @property
    def __name__(self):
        return 'HPSv2'
    
    def load_to_device(self, load_device):
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
                    'ViT-H-14',
                    pretrained='laion2b_s32b_b79k',
                    precision='amp',
                    device=load_device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_patch_dropout=False,
                    force_image_size=None,
                    pretrained_image=False,
                    image_mean=None,
                    image_std=None,
                    light_augmentation=True,
                    aug_cfg={},
                    output_dict=True,
                    with_score_predictor=False,
                    with_region_predictor=False
                )
        # workaround for the zero3
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(load_device)
        self.model.eval()
    
    def __call__(self, prompts, images, **kwargs):
        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        result = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):

            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])
        return result

class HPSv2Compare(HPSv2):
    def __init__(self):
        super().__init__()
    
    @property
    def __name__(self):
        return 'HPSv2Compare'

    def __call__(self, prompts, images, image_path, **kwargs): 
        
        image_before_list = [Image.open(i) for i in image_path]
        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        result = []
        for prompt, image, image_before in zip(prompts, images, image_before_list):
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
                image_before = self.preprocess_val(image_before).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()

                    outputs_before = self.model(image_before, text)
                    image_features_before, text_features_before = outputs_before["image_features"], outputs_before["text_features"]
                    logits_per_image_before = image_features_before @ text_features_before.T
                    hps_score_before = torch.diagonal(logits_per_image_before).cpu().numpy()
            result.append(hps_score[0] - hps_score_before[0])
        return result