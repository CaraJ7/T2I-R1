import os
import os
import torch
import requests
from PIL import Image
import numpy as np
import argparse
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, GitForCausalLM


class GIT:
    def __init__(self, args):

        ckpt_path = args.git_ckpt_path

        self.processor = AutoProcessor.from_pretrained(ckpt_path)
        config = AutoConfig.from_pretrained(ckpt_path)
        self.model = GitForCausalLM(config)
        # workaround for the zero3
        ckpt = torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)

        # get yes and no token ids
        self.yes_token_id = self.processor.tokenizer.encode('yes')[1] # [bos, yes, eos]
        self.no_token_id = self.processor.tokenizer.encode('no')[1] # [bos, no, eos]

    @property
    def __name__(self):
        return 'GIT'
    
    def load_to_device(self, load_device):

        self.model.to(load_device)

        # freeze all parameters
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        self.model.eval()

            
    def __call__(self, prompts, images, **kwargs):
        device = list(self.model.parameters())[0].device

        # single generation
        score = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            # we do not calculate the score for spatial and numeracy tasks
            if kwargs['task_type'][i] in ['spatial', 'numeracy']:
                score.append(0)
                continue

            # calculate attr nouns if exist, otherwise, calculate the nouns
            if kwargs['attr_nouns'][i] is not None:
                key = 'attr_nouns'
            else:
                key = 'nouns'
                if kwargs['nouns'][i] is None or len(kwargs['nouns'][i]) == 0:
                    score.append(1)
                    continue

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(device)
            temp_score = []
            for idx, attr_noun in enumerate(kwargs[key][i]): # all the attr nouns should be the same, so we take the first one
                vqa_prompts = f"{attr_noun}?"
                input_ids = self.processor(text=vqa_prompts, add_special_tokens=False).input_ids
                input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

                logits = self.model(pixel_values=pixel_values, input_ids=input_ids, return_dict=True).logits[:, -1]
                probs = torch.softmax(logits, dim=1)
                prob_yes = probs[:, self.yes_token_id]
                prob_no = probs[:, self.no_token_id]
                temp_score.append((prob_yes / (prob_yes + prob_no)).cpu().numpy())
            score.append(np.mean(temp_score).tolist())

        return score  # tensor
