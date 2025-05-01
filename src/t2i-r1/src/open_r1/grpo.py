# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import JanusT2IR1Trainer

# add image_generation_prompt in the GRPOConfig
@dataclass
class GRPOConfig(GRPOConfig):
    """
    Configuration class for the GRPO training script.
    """
    new_generations_image: int = field(default=1, metadata={"help": "The number of new generations of image to generate"})
    image_token_num_per_image: int = field(default=576, metadata={"help": "The number of image tokens to generate"})
    # image_gen_temperature: float = field(default=1.0, metadata={"help": "The temperature for image generation"}) # HACK, this is always 1.0
    cfg_weight: float = field(default=3.0, metadata={"help": "The cfg weight for image generation"})
    reasoning_prompt_path: Optional[str] = field(
        default='',
    )
    img_size: int = field(default=384, metadata={"help": "The size of the image to generate"})
    patch_size: int = field(default=16, metadata={"help": "The patch size of the image to generate"})
    max_textcot_length: int = field(default=None, metadata={"help": "The maximum length of the text cot"})
    hps_ckpt_path: str = field(default=None, metadata={"help": "The path to the hps checkpoint"})
    git_ckpt_path: str = field(default=None, metadata={"help": "The path to the git checkpoint"})
    gdino_ckpt_path: str = field(default=None, metadata={"help": "The path to the gdino checkpoint"})
    gdino_config_path: str = field(default=None, metadata={"help": "The path to the gdino config"})
    orm_ckpt_path: str = field(default=None, metadata={"help": "The path to the orm checkpoint"})
    
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'hps', 'git', 'gdino'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["hps", "git", "gdino", "orm"],
        metadata={"help": "List of reward functions. Possible values: 'hps', 'git', 'gdino', 'orm'"},
    )

def make_detection_prompt(nouns):
    if len(nouns) == 0:
        return '', []
    
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


reward_funcs_registry = {
    "hps": 'hps',
    'hps_compare': 'hps_compare',
    'git': 'git',
    'gdino': 'gdino',
    'orm': 'orm',
    'unify': 'unify',
}


def main(script_args, training_args, model_args):
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    if script_args.dataset_name.endswith('.csv'):
        suffix = 'csv'
    elif script_args.dataset_name.endswith('.json'):
        suffix = 'json'
    elif script_args.dataset_name.endswith('.parquet'):
        suffix = 'parquet'
    dataset = load_dataset(suffix, data_files=script_args.dataset_name)
    print('Dataset length: ', len(dataset['train']))

    # load cot prompt
    if training_args.reasoning_prompt_path:
        with open(training_args.reasoning_prompt_path, 'r') as f:
            cot_prompt = f.read()
            training_args.cot_prompt = cot_prompt
    
            
    # Format into conversation
    def make_conversation(example):
        # make detection prompt
        if 'nouns' in example and example['nouns'] is not None:
            det_text_prompt, det_token_spans = make_detection_prompt(example['nouns'])
        else:
            det_text_prompt = ''
            det_token_spans = []
        det_prompt_dict = {
            'text_prompt': det_text_prompt,
            'token_spans': det_token_spans,
        }
        # make vqa prompt
        if 'attr_nouns' in example and example['attr_nouns'] is not None:
            questions = [f"{attr_noun}?" for attr_noun in example['attr_nouns']]
            vqa_prompt = {'questions': questions}
        else:
            vqa_prompt = {'questions': []}  # Changed from None to empty list

        return {
            "prompt": [
                {"role": "User", "content": cot_prompt.format(example["prompt"])},
                {"role": "Assistant", "content": ""},
            ],
            'raw_prompt': example["prompt"],
            'det_prompt': det_prompt_dict,
            'task_type': example['task_type'],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "User",
                    "content": ref_prompt.format(ori_prompt=example['ori_prompt'], gen_prompt=example['gen_prompt']),
                    "images": [example['image_path']]
                },
                {"role": "Assistant", "content": ""},
            ],
            'raw_prompt': example['ori_prompt'],
            'image': example['image_path'],
        }



    if "image" in dataset[script_args.dataset_train_split].features or 'image_path' in dataset[script_args.dataset_train_split].features:
        print("***************has image in dataset***************")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("***************no image in dataset***************")
        dataset = dataset.map(
            make_conversation,
            num_proc=1,
            # remove_columns=['spatial_info', 'numeracy_info', 'attr_nouns', 'nouns']
        )
        # dataset = dataset.remove_columns("messages")

    
    trainer_cls = JanusT2IR1Trainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        script_args=script_args,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
