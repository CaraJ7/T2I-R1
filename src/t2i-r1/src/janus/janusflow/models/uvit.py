# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# modified from: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/simple_diffusion.py
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple, Union

import numpy as np
import torchvision
import torchvision.utils
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class ImageHead(nn.Module):

    def __init__(self, decoder_cfg, gpt_cfg, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        cfg = (
            AttrDict(
                norm_type="layernorm",
                is_exp_norm=False,
                sequence_parallel=False,
                use_userbuffer=False,
                norm_eps=1e-5,
                norm_bias=True,
                gradient_accumulation_fusion=True,
                use_fp32_head_weight=False,
            )
            + gpt_cfg
        )
        group = PG.tensor_parallel_group()
        assert cfg.norm_type in [
            "layernorm",
            "rmsnorm",
        ], f"Norm type:{cfg.norm_type} not supported"
        if cfg.norm_type == "rmsnorm":
            self.norm = DropoutAddRMSNorm(
                cfg.n_embed,
                prenorm=False,
                eps=cfg.norm_eps,
                is_exp_norm=cfg.is_exp_norm,
                sequence_parallel=cfg.sequence_parallel,
            )
        else:
            self.norm = DropoutAddLayerNorm(
                cfg.n_embed,
                prenorm=False,
                eps=cfg.norm_eps,
                is_exp_norm=cfg.is_exp_norm,
                sequence_parallel=cfg.sequence_parallel,
                bias=cfg.norm_bias,
            )

        multiple_of = 256
        if decoder_cfg.in_channels % multiple_of != 0:
            warnings.warn(
                f"建议把 vocab_size 设置为 {multiple_of} 的倍数, 否则会影响矩阵乘法的性能"
            )

        dtype = default_dtype = torch.get_default_dtype()
        if cfg.use_fp32_head_weight:
            dtype = torch.float32
            print(
                "使用 fp32 head weight!!!! 与原来的 bf16 head weight 不兼容\n",
                end="",
                flush=True,
            )
        torch.set_default_dtype(dtype)
        self.head = ColumnParallelLinear(
            cfg.n_embed,
            decoder_cfg.in_channels,
            bias=True,
            group=group,
            sequence_parallel=cfg.sequence_parallel,
            use_userbuffer=cfg.use_userbuffer,
            gradient_accumulation_fusion=cfg.gradient_accumulation_fusion,
            use_fp32_output=False,
        )
        torch.set_default_dtype(default_dtype)

        self.use_fp32_head_weight = cfg.use_fp32_head_weight

    def forward(
        self, input_args, images_split_mask: Optional[torch.BoolTensor] = None, **kwargs
    ):
        residual = None
        if isinstance(input_args, tuple):
            x, residual = input_args
        else:
            x = input_args

        x = self.norm(x, residual)

        if self.use_fp32_head_weight:
            assert (
                self.head.weight.dtype == torch.float32
            ), f"head.weight is {self.head.weight.dtype}"
            x = x.float()

        if images_split_mask is None:
            logits = self.head(x)
        else:
            bs, n_images = images_split_mask.shape[:2]
            n_embed = x.shape[-1]

            images_embed = torch.masked_select(
                x.unsqueeze(1), images_split_mask.unsqueeze(-1)
            )
            images_embed = images_embed.view((bs * n_images, -1, n_embed))
            logits = self.head(images_embed)

        return logits


class GlobalResponseNorm(nn.Module):
    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)

        return torch.addcmul(self.bias, (self.weight * nx + 1), x, value=1)


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        stride=2,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        stride=2,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.stride = stride

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=self.stride, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        channels,
        norm_eps,
        elementwise_affine,
        use_bias,
        hidden_dropout,
        hidden_size,
        res_ffn_factor: int = 4,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=use_bias,
        )
        self.norm = RMSNorm(channels, norm_eps)
        self.channelwise_linear_1 = nn.Linear(
            channels, int(channels * res_ffn_factor), bias=use_bias
        )
        self.channelwise_act = nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        self.channelwise_linear_2 = nn.Linear(
            int(channels * res_ffn_factor), channels, bias=use_bias
        )
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        self.cond_embeds_mapper = nn.Linear(hidden_size, channels * 2, use_bias)

    def forward(self, x, cond_embeds):
        x_res = x

        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)
        x = x.permute(0, 3, 1, 2)

        x = x + x_res

        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        # x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = torch.addcmul(
            shift[:, :, None, None], x, (1 + scale)[:, :, None, None], value=1
        )

        return x


class Patchify(nn.Module):
    def __init__(
        self,
        in_channels,
        block_out_channels,
        patch_size,
        bias,
        elementwise_affine,
        eps,
        kernel_size=None,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.patch_conv = nn.Conv2d(
            in_channels,
            block_out_channels,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = RMSNorm(block_out_channels, eps)

    def forward(self, x):
        embeddings = self.patch_conv(x)
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = self.norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings


class Unpatchify(nn.Module):
    def __init__(
        self, in_channels, out_channels, patch_size, bias, elementwise_affine, eps
    ):
        super().__init__()
        self.norm = RMSNorm(in_channels, eps)
        self.unpatch_conv = nn.Conv2d(
            in_channels,
            out_channels * patch_size * patch_size,
            kernel_size=1,
            bias=bias,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        # [b, c, h, w]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.unpatch_conv(x)
        x = self.pixel_shuffle(x)
        return x


class UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        num_res_blocks,
        stride,
        hidden_size,
        hidden_dropout,
        elementwise_affine,
        norm_eps,
        use_bias,
        downsample: bool,
        upsample: bool,
        res_ffn_factor: int = 4,
        seq_len=None,
        concat_input=False,
        original_input_channels=None,
        use_zero=True,
        norm_type="RMS",
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            conv_block = ConvNextBlock(
                channels,
                norm_eps,
                elementwise_affine,
                use_bias,
                hidden_dropout,
                hidden_size,
                res_ffn_factor=res_ffn_factor,
            )

            self.res_blocks.append(conv_block)

        if downsample:
            self.downsample = Downsample2D(
                channels=channels,
                out_channels=out_channels,
                use_conv=True,
                name="Conv2d_0",
                kernel_size=3,
                padding=1,
                stride=stride,
                norm_type="rms_norm",
                eps=norm_eps,
                elementwise_affine=elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        if upsample:
            self.upsample = Upsample2D(
                channels=channels,
                out_channels=out_channels,
                use_conv_transpose=False,
                use_conv=True,
                kernel_size=3,
                padding=1,
                stride=stride,
                name="conv",
                norm_type="rms_norm",
                eps=norm_eps,
                elementwise_affine=elementwise_affine,
                bias=use_bias,
                interpolate=True,
            )
        else:
            self.upsample = None

    def forward(self, x, emb, recompute=False):
        for res_block in self.res_blocks:
            x = res_block(x, emb)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ShallowUViTEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        stride=4,
        kernel_size=7,
        padding=None,
        block_out_channels=(768,),
        layers_in_middle=2,
        hidden_size=2048,
        elementwise_affine=True,
        use_bias=True,
        norm_eps=1e-6,
        dropout=0.0,
        use_mid_block=True,
        **kwargs,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embed = TimestepEmbedding(
            block_out_channels[0], hidden_size, sample_proj_bias=use_bias
        )

        if padding is None:
            padding = math.ceil(kernel_size - stride)
        self.in_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=block_out_channels[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_mid_block:
            self.mid_block = UVitBlock(
                block_out_channels[-1],
                block_out_channels[-1],
                num_res_blocks=layers_in_middle,
                hidden_size=hidden_size,
                hidden_dropout=dropout,
                elementwise_affine=elementwise_affine,
                norm_eps=norm_eps,
                use_bias=use_bias,
                downsample=False,
                upsample=False,
                stride=1,
                res_ffn_factor=4,
            )
        else:
            self.mid_block = None

    def get_num_extra_tensors(self):
        return 2

    def forward(self, x, timesteps):

        bs = x.shape[0]
        dtype = x.dtype

        t_emb = self.time_proj(timesteps.flatten()).view(bs, -1).to(dtype)
        t_emb = self.time_embed(t_emb)
        x_emb = self.in_conv(x)

        if self.mid_block is not None:
            x_emb = self.mid_block(x_emb, t_emb)

        hs = [x_emb]
        return x_emb, t_emb, hs


class ShallowUViTDecoder(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=3,
        block_out_channels: Tuple[int] = (768,),
        upsamples=2,
        layers_in_middle=2,
        hidden_size=2048,
        elementwise_affine=True,
        norm_eps=1e-6,
        use_bias=True,
        dropout=0.0,
        use_mid_block=True,
        **kwargs,
    ):
        super().__init__()
        if use_mid_block:
            self.mid_block = UVitBlock(
                in_channels + block_out_channels[-1],
                block_out_channels[
                    -1
                ],  # In fact, the parameter is not used because it has no effect when both downsample and upsample are set to false.
                num_res_blocks=layers_in_middle,
                hidden_size=hidden_size,
                hidden_dropout=dropout,
                elementwise_affine=elementwise_affine,
                norm_eps=norm_eps,
                use_bias=use_bias,
                downsample=False,
                upsample=False,
                stride=1,
                res_ffn_factor=4,
            )
        else:
            self.mid_block = None
        self.out_convs = nn.ModuleList()
        for rank in range(upsamples):
            if rank == upsamples - 1:
                curr_out_channels = out_channels
            else:
                curr_out_channels = block_out_channels[-1]
            if rank == 0:
                curr_in_channels = block_out_channels[-1] + in_channels
            else:
                curr_in_channels = block_out_channels[-1]
            self.out_convs.append(
                Unpatchify(
                    curr_in_channels,
                    curr_out_channels,
                    patch_size=2,
                    bias=use_bias,
                    elementwise_affine=elementwise_affine,
                    eps=norm_eps,
                )
            )
        self.input_norm = RMSNorm(in_channels, norm_eps)

    def forward(self, x, hs, t_emb):

        x = x.permute(0, 2, 3, 1)
        x = self.input_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = torch.cat([x, hs.pop()], dim=1)
        if self.mid_block is not None:
            x = self.mid_block(x, t_emb)
        for out_conv in self.out_convs:
            x = out_conv(x)
        assert len(hs) == 0
        return x
