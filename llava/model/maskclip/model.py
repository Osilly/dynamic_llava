"""MaskCLIP model."""

import logging
from collections import OrderedDict
from typing import List, Tuple, Union, Tuple

import numpy as np
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    """Bottleneck.

    Parameters
    ----------
    inplanes : int
        Number of input channels.
    planes : int
        Number of hidden channels.
    stride : int
        Stride.
    """

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        """Initialize module."""
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input image batch.

        Returns
        -------
        Tensor
            Processed input batch.
        """
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(
        self,
        x: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input image batch.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            Processed input batch/query, key, value.
        """
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        q = self.c_proj(self.q_proj(x))
        k = self.c_proj(self.k_proj(x))
        v = self.c_proj(self.v_proj(x))
        # forward x
        x, weights = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True,
        )
        v = v + x / 2
        return x, q, k, v


class ModifiedResNet(nn.Module):
    """ModifiedResNet.

    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool

    Parameters
    ----------
    layers : List[int]
        Number of layers for model stages.
    output_dim : int
        Output dimension.
    heads : int
        Number of heads.
    input_resolution : int
        Input resolution of image.
    width : int
        Width of model.
    """

    def __init__(
        self,
        layers: List[int],
        output_dim: int,
        heads: int,
        input_resolution: int,
        width: int,
    ):
        """Initialize model."""
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.channels = [width] + list(layers)

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(
        self, planes: int, blocks: nn.Module, stride: int = 1
    ) -> nn.Sequential:
        """Make ResNet layer.

        Parameters
        ----------
        planes : int
            Number of input channels.
        blocks : nn.Module
            Block.
        stride : int = 1
            Stride.
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def resize_positional_embedding(
        self, input_resolution: int, mode: str = "bicubic"
    ) -> None:
        """Resize `positional_embedding`.

        Parameters
        ----------
        input_resolution : int
            Input resolution needed.
        mode : str = "bicubic"
            Interpolation mode.
        """
        if input_resolution != self.input_resolution:
            Nt = self.input_resolution // 32
            nNt = input_resolution // 32

            class_positional_embedding = self.attnpool.positional_embedding.data[[0], :]
            image_positional_embedding = self.attnpool.positional_embedding.data[1:, :]
            image_positional_embedding = image_positional_embedding.unsqueeze(
                0
            ).permute(0, 2, 1)
            B, D, L = image_positional_embedding.shape
            image_positional_embedding = image_positional_embedding.reshape(
                B, D, Nt, Nt
            )
            image_positional_embedding = F.interpolate(
                image_positional_embedding,
                size=(nNt, nNt),
                mode=mode,
                align_corners=False,
            )
            image_positional_embedding = (
                image_positional_embedding.squeeze(0).view(D, -1).permute(1, 0)
            )
            self.attnpool.positional_embedding = nn.Parameter(
                torch.cat(
                    [class_positional_embedding, image_positional_embedding], dim=0
                ),
                requires_grad=False,
            )
            self.input_resolution = input_resolution

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input image batch.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Classification/segmentation features.
        """
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, q, k, v = self.attnpool(x)
        x = x.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        class_features = x[:, 0, :]
        image_features = v[:, 1:, :]
        return class_features, image_features


class LayerNorm(nn.LayerNorm):
    """LayerNorm.

    Subclass torch's LayerNorm to handle fp16.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input batch.

        Returns
        -------
        Tensor
            Processed batch.
        """
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU."""

    def forward(self, x: Tensor) -> Tensor:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input batch.

        Returns
        -------
        Tensor
            Processed batch.
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """ResidualAttentionBlock.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_head : int
        Number of heads.
    attn_mask : Tensor = None
        Attention mask.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor) -> Tensor:
        """Apply attention.

        Parameters
        ----------
        x : Tensor
            Input batch.

        Returns
        -------
        Tensor
            Attention matrix.
        """
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(
        self,
        x: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input image batch.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            Processed input batch/query, key, value.
        """
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        # forward x
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        v = v + x
        return x, q, k, v


class Transformer(nn.Module):
    """Transformer.

    Parameters
    ----------
    width : int
        Model dimension.
    layers : int
        Number of Transformer layers.
    heads : int
        Number of heads.
    attn_mask : Tensor = None
        Attention mask.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        """Initialize model."""
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(
        self,
        x: Tensor,
        feature_layer: int = None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Process input batch.

        Parameters
        ----------
        x : Tensor
            Input image batch.
        feature_layer : int = None
            The layer from which to extract features.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            Processed input batch/query, key, value.
        """
        for i in range(self.layers):
            x, q, k, v = self.resblocks[i](x)
            if feature_layer and i == feature_layer:
                return x, q, k, v
        return x, q, k, v


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def resize_positional_embedding(self, input_resolution: int, mode: str = "bicubic"):
        if input_resolution != self.input_resolution:
            Nt = self.input_resolution // self.patch_size
            nNt = input_resolution // self.patch_size

            class_positional_embedding = self.positional_embedding.data[[0], :]
            image_positional_embedding = self.positional_embedding.data[1:, :]
            image_positional_embedding = image_positional_embedding.unsqueeze(
                0
            ).permute(0, 2, 1)
            B, D, L = image_positional_embedding.shape
            image_positional_embedding = image_positional_embedding.reshape(
                B, D, Nt, Nt
            )
            image_positional_embedding = F.interpolate(
                image_positional_embedding,
                size=(nNt, nNt),
                mode=mode,
                align_corners=False,
            )
            image_positional_embedding = (
                image_positional_embedding.squeeze(0).view(D, -1).permute(1, 0)
            )
            self.positional_embedding = nn.Parameter(
                torch.cat(
                    [class_positional_embedding, image_positional_embedding], dim=0
                ),
                requires_grad=False,
            )
            self.input_resolution = input_resolution

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        _, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_embedding = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_embedding, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, q, k, v = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        v = v.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        class_features = x[:, 0, :] @ self.proj
        image_features = v[:, 1:, :] @ self.proj
        return class_features, image_features


class MaskCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def initialize_text_embeddings(self, text_embeddings: torch.Tensor):
        self.register_buffer("text_embeddings", text_embeddings)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, normalize=True):
        class_features, image_features = self.visual(image.type(self.dtype))

        if normalize:
            class_features = class_features / class_features.norm(dim=1, keepdim=True)

        return class_features, image_features

    def encode_text(self, text, normalize=True, feature_layer=None):
        if feature_layer and feature_layer < 0:
            feature_layer = self.transformer.layers + feature_layer
            assert (
                feature_layer < self.transformer.layers
            ), f"Transformer has no layer: {feature_layer}"

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, feature_layer=feature_layer)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        text_features = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        )

        if normalize:
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def encode_features(
        self, image, text=None, normalize=True, text_feature_layer=None
    ):
        class_features, image_features = self.encode_image(image, normalize)

        if text is not None:
            text_features = self.encode_text(text, normalize, text_feature_layer)
        else:
            text_features = None

        return class_features, image_features, text_features

    def decode_features(self, class_features, image_features, text_features=None):
        if text_features is None:
            assert hasattr(self, "text_embeddings"), ValueError(
                "`text_embeddings` are not initialized"
            )
            text_features = self.text_embeddings

        logit_scale = self.logit_scale.exp()
        logits_per_class = logit_scale * class_features @ text_features.t()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_image = logits_per_image.permute(0, 2, 1)
        B, C, Nt = logits_per_image.shape
        logits_per_image = logits_per_image.reshape(
            B, C, int(math.sqrt(Nt)), int(math.sqrt(Nt))
        )

        return logits_per_class, logits_per_image

    def forward(self, image, text=None, normalize=True, text_feature_layer=None):
        class_features, image_features, text_features = self.encode_features(
            image=image,
            text=text,
            normalize=normalize,
            text_feature_layer=text_feature_layer,
        )
        logits_per_class, logits_per_image = self.decode_features(
            class_features=class_features,
            image_features=image_features,
            text_features=text_features,
        )
        return logits_per_image


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, **kwargs):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    model = MaskCLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
