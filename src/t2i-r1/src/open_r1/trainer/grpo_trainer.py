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

'''
Two Forward Passes
'''

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image

import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from janus.models import MultiModalityCausalLM, VLChatProcessor
from utils.reward_hps import HPSv2
from utils.reward_git import GIT
from utils.reward_gdino import GDino
from utils.reward_orm import ORM
import shutil

import copy
import re


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class JanusT2IR1Trainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            # torch_dtype = model_init_kwargs.get("torch_dtype")
            # if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            #     pass  # torch_dtype is already a torch.dtype or "auto" or None
            # elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            #     torch_dtype = getattr(torch, torch_dtype)
            #     model_init_kwargs["torch_dtype"] = torch_dtype
            # else:
            #     raise ValueError(
            #         "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
            #         f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            #     )
            # # Disable caching if gradient checkpointing is enabled (not supported)
            # model_init_kwargs["use_cache"] = (
            #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            # )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        model.language_model.config._attn_implementation == "flash_attention_2"
        # freeze all vision encoders
        for name, param in model.named_parameters():
            if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"): # choose whatever you like here
                param.requires_grad = False
        # try gradient checkpointing
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable()
        # remove unnecessary parameters
        # del model.vision_model
        # del model.aligner
            
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled() and args.beta != 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        elif peft_config is None and args.beta != 0:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'hps' in reward_func:
                reward_funcs[i] = HPSv2(args)
            elif isinstance(reward_func, str) and 'git' in reward_func:
                reward_funcs[i] = GIT(args)
            elif isinstance(reward_func, str) and 'gdino' in reward_func:
                reward_funcs[i] = GDino(args)
            elif isinstance(reward_func, str) and 'orm' in reward_func:
                reward_funcs[i] = ORM(args)
            else:
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.new_generations_image = args.new_generations_image
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,  
        #     temperature=1, # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=pad_token_id,
        # )
        self.beta = args.beta


        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.beta != 0:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, HPSv2) or isinstance(reward_func, GDino) or isinstance(reward_func, GIT):
                reward_func.load_to_device(self.accelerator.device)
            elif isinstance(reward_func, ORM):
                reward_func.load_to_device(self.accelerator.device)
                reward_func.accelerator = self.accelerator
                if self.is_deepspeed_enabled:   
                    reward_func.model = prepare_deepspeed(reward_func.model, self.accelerator)
                else:
                    reward_func.model = self.accelerator.prepare_model(reward_func.model, evaluation_mode=True)
        
        # load cot prompt
        with open(args.reasoning_prompt_path, 'r') as f:
            self.cot_prompt = f.read()
        
        # record image start token id
        self.user_end_token_id = self.processing_class.tokenizer.encode('\n')[1]
        # image start token for generation
        self.image_start_token_id = self.processing_class.tokenizer.encode(self.processing_class.image_start_tag)[1]
        # 576
        self.image_token_num_per_image = args.image_token_num_per_image
        self.cfg_weight = args.cfg_weight
        self.image_gen_temperature = 1
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.max_textcot_length = args.max_textcot_length

        # image loss is moved to grpo_trainer_two_forward_imageloss
        # assert not self.image_loss


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_embeds, text_ids, img_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []

            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        # here, we only compute either text or image loss, so ids of other one could be omitted
        if img_ids is not None:
            # compute logits for image tokens
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            # (text input id, image start token, image input id)
            # text_ids: text input id + image start token
            # img_ids: img_id (image token)
            image_logits = model.gen_head(last_hidden_states[:, -(img_ids.size(1)+1):, :]) # image prediction
            
            img_input_ids = torch.cat([img_ids.new_zeros(img_ids.size(0), 1), img_ids], dim=1) # cat a random one here, since it is not used in the loss calculation
            per_token_logps_img = _get_per_token_logps_part(image_logits, img_input_ids) # only calculate image loss
            return torch.cat([
                per_token_logps_img.new_zeros(
                    (per_token_logps_img.size(0), input_embeds.size(1) - per_token_logps_img.size(1) - 1)
                ), # the return length should be the input length minus 1 (the last token does not need predict)
                per_token_logps_img
            ], 
            dim=1)
        else: # only calculate text ids
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            text_logits = model.language_model.lm_head(last_hidden_states) 
            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids) 
            return per_token_logps_text


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
    
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            self.processing_class.apply_sft_template_for_multi_turn_prompts(
            conversations=prompt,
            sft_format=self.processing_class.sft_format,
            system_prompt="You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.",
        ) for prompt in prompts]
        prompt_inputs= self.processing_class.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left", # left padding, same in the official repo
            add_special_tokens=True,
        ) # {'input_ids', 'attention_mask'}
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        torch.set_grad_enabled(False)
        # Generate completions for text cot
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.language_model.config.use_cache = False
            unwrapped_model.language_model.gradient_checkpointing_disable()

            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            input_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids)
            # TODO: if num_generations is too large, we need to split it into multiple batches
            if self.num_generations > 100:
                total_generations = []
                for i in range(prompt_ids.shape[0] // self.num_generations):
                    current_input_embeds = input_embeds[i*self.num_generations: (i+1)*self.num_generations]
                    current_attn_mask = prompt_mask[i*self.num_generations: (i+1)*self.num_generations]
                    prompt_completion_ids = unwrapped_model.language_model.generate(
                        inputs_embeds=current_input_embeds,
                        attention_mask=current_attn_mask,
                        pad_token_id=self.processing_class.tokenizer.eos_token_id,
                        bos_token_id=self.processing_class.tokenizer.bos_token_id,
                        eos_token_id=self.processing_class.tokenizer.eos_token_id,
                        max_new_tokens=self.max_completion_length,
                        do_sample=True,
                        use_cache=True,
                    )
                    total_generations.append(prompt_completion_ids)
                prompt_completion_ids = torch.cat(total_generations, dim=0)
            else: # if num_generations == 1, we directly generate all for the batch data
                prompt_completion_ids = unwrapped_model.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=prompt_mask,
                    pad_token_id=self.processing_class.tokenizer.eos_token_id,
                    bos_token_id=self.processing_class.tokenizer.bos_token_id,
                    eos_token_id=self.processing_class.tokenizer.eos_token_id,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    use_cache=True,
                )
            # for Janus, the prompt_completion_ids is only the answer, without the input

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_ids

            if self.max_textcot_length is not None:
                prompt_completion_ids = prompt_completion_ids[:, -self.max_textcot_length :]
                
            completion_ids = prompt_completion_ids

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Calculate semantic-cot loss
        loss_dict = {}
        model.module.language_model.gradient_checkpointing_enable()
        torch.set_grad_enabled(True)
        # Get Embedding and Mask
        prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_embeds = model.module.language_model.get_input_embeddings()(prompt_all_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # Get Per-token Logps and Ref Per-token Logps (With Completion Mask)
        per_token_logps = self._get_per_token_logps(
            model=model.module, 
            input_embeds=input_embeds,
            text_ids=prompt_all_ids, 
            img_ids=None, 
            attention_mask=attention_mask)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.language_model.gradient_checkpointing_enable()
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, 
                    input_embeds, 
                    prompt_all_ids, 
                    None,
                    attention_mask)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                # dummy ref_per_token_logps
                ref_per_token_logps = torch.zeros_like(per_token_logps)
        loss_dict['semantic-cot'] = {
            'per_token_logps': per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'completion_mask': completion_mask,
        }
        torch.set_grad_enabled(False)

        image_gen_prompt_list = []
        for i in range(completion_ids.shape[0]):
            answer = self.processing_class.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            raw_prompt = inputs[i // self.num_generations]['raw_prompt']
            image_gen_prompt = f"{raw_prompt}. {answer}" 

            conversation = [
                {
                    "role": "User",
                    "content": image_gen_prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processing_class.sft_format,
                system_prompt="",
            )
            # sft_format = sft_format + vl_chat_processor.image_start_tag
            image_gen_prompt_list.append(sft_format)
        
        # add image start token at the end
        prompt_inputs = self.processing_class.tokenizer(
            text=image_gen_prompt_list,
            return_tensors="pt",
            padding=True,
            padding_side="right", # Right is better 
            add_special_tokens=True,
        ) # {'input_ids', 'attention_mask'}

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        # # add image start token at the end
        prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), self.image_start_token_id)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

        # new_generations_image
        prompt_ids = prompt_ids.repeat_interleave(self.new_generations_image, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.new_generations_image, dim=0)

        # Generate the image tokens
        # torch.cuda.empty_cache()
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.language_model.config.use_cache = False
            unwrapped_model.language_model.gradient_checkpointing_disable()

            inputs_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids)
            pad_input_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), self.processing_class.pad_id))
            total_generated_tokens_img = []

            # Make cond and uncond inputs embeds and attention mask
            cond_inputs_embeds = inputs_embeds
            cond_attention_mask = attention_mask
            uncond_inputs_embeds = cond_inputs_embeds.clone()
            uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
            
            inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
            inputs_embeds_img[1::2] = uncond_inputs_embeds
            attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
            attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

            split_size = 32
            for jj in range(0, inputs_embeds_img.shape[0], split_size):
                print(f"Generating image {jj}")
                start = jj
                end = min(jj + split_size, inputs_embeds_img.shape[0])
                generated_tokens = torch.zeros(((end-start)//2, self.image_token_num_per_image), dtype=torch.int64).cuda()
                cur_inputs_embeds_img = inputs_embeds_img[start: end]
                cur_attention_mask_img = attention_mask_img[start: end]

                for k in range(self.image_token_num_per_image):
                    outputs = unwrapped_model.language_model.model(
                        inputs_embeds=cur_inputs_embeds_img, 
                        use_cache=True, 
                        past_key_values=outputs.past_key_values if k != 0 else None, 
                        attention_mask=cur_attention_mask_img
                    )
                    
                    hidden_states = outputs.last_hidden_state
                    logits = unwrapped_model.gen_head(hidden_states[:, -1, :])
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]
                    
                    logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)
                    probs = torch.softmax(logits / self.image_gen_temperature, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens[:, k] = next_token.squeeze(dim=-1)

                    next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                    img_embeds = unwrapped_model.prepare_gen_img_embeds(next_token)
                    cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                    cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)

                    del logits, probs, logit_cond, logit_uncond, hidden_states, next_token, img_embeds


                total_generated_tokens_img.append(generated_tokens)
        total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

        # Calculate token-cot loss
        model.module.language_model.gradient_checkpointing_enable()
        torch.set_grad_enabled(True)
        # Get the logp for all the generated tokens
        input_embeds = torch.cat(
            [
                model.module.language_model.get_input_embeddings()(prompt_ids),
                model.module.prepare_gen_img_embeds(
                    total_generated_tokens_img
                )
            ],
            dim=1
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(total_generated_tokens_img)
            ],
            dim=1
        )

        per_token_logps = self._get_per_token_logps(
            model=model.module, 
            input_embeds=input_embeds,
            text_ids=None, 
            img_ids=total_generated_tokens_img, 
            attention_mask=attention_mask
        )
        prompt_length = prompt_ids.size(1)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        completion_mask = torch.ones_like(total_generated_tokens_img)

        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.language_model.gradient_checkpointing_enable()
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, 
                    input_embeds=input_embeds,
                    text_ids=None, 
                    img_ids=total_generated_tokens_img, 
                    attention_mask=attention_mask
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                # dummy ref_per_token_logps
                ref_per_token_logps = torch.zeros_like(per_token_logps)
        loss_dict['token-cot'] = {
            'per_token_logps': per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'completion_mask': completion_mask,
        }
        torch.set_grad_enabled(False)

        # torch.cuda.empty_cache()
        total_generated_tokens_img = total_generated_tokens_img.detach()

        # Generate the image
        with unwrap_model_for_generation(model.module.gen_vision_model, self.accelerator) as unwrapped_model:
            dec = unwrapped_model.decode_code(total_generated_tokens_img.to(dtype=torch.int), shape=[total_generated_tokens_img.shape[0], 8, self.img_size//self.patch_size, self.img_size//self.patch_size])
            # convert tensor to numpy array and ensure correct format
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            # transpose if needed (B, C, H, W) -> (B, H, W, C)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((total_generated_tokens_img.shape[0], self.img_size, self.img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            images = [Image.fromarray(visual_img[idx]) for idx in range(visual_img.shape[0])]


        # Compute the rewards
        prompts = [input["raw_prompt"] for input in inputs for _ in range(self.num_generations) for __ in range(self.new_generations_image)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations * self.new_generations_image)
                output_reward_func = reward_func(prompts=prompts, images=images, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        torch.set_grad_enabled(True)

        # Calculate the loss together for semantic-cot and token-cot
        for key in loss_dict['semantic-cot']:
            loss_dict['semantic-cot'][key] = loss_dict['semantic-cot'][key].repeat_interleave(self.new_generations_image, dim=0)
        per_token_logps, ref_per_token_logps, completion_mask = [], [], []
        for key in ['semantic-cot', 'token-cot']:
            if loss_dict[key]['per_token_logps'] is None:
                loss_dict[key]['loss'] = None
                continue
            per_token_logps.append(loss_dict[key]['per_token_logps'])
            ref_per_token_logps.append(loss_dict[key]['ref_per_token_logps'])
            completion_mask.append(loss_dict[key]['completion_mask'])
        per_token_logps = torch.cat(per_token_logps, dim=1)
        ref_per_token_logps = torch.cat(ref_per_token_logps, dim=1)
        completion_mask = torch.cat(completion_mask, dim=1)
        # TODO: we cat each logp and completion_mask, and then compute the loss togeter
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        self._metrics[f"kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[f"loss"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(loss_dict['semantic-cot']['completion_mask'].sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
