from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


from transformers.models.clip.modeling_clip import (
    CLIPModel,
    CLIPConfig,
    CLIPVisionConfig,
    CLIPVisionEmbeddings,
    add_start_docstrings_to_model_forward,
    CLIP_VISION_INPUTS_DOCSTRING,
    replace_return_docstrings,
    CLIPMLP,
    dataclass,
    ModelOutput,
)

from transformers import CLIPTokenizer, CLIPProcessor

from .classes import (
    stuff_classes,
    all_pascal_context_classes,
    bg_classes,
)
import torch.nn.functional as F


@dataclass
class MaskCLIPWithBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    query_states: torch.FloatTensor = None
    key_states: torch.FloatTensor = None
    value_states: torch.FloatTensor = None


@dataclass
class MaskCLIPWithBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_features: torch.FloatTensor = None


class MaskCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        requires_qkv: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        # ----------------------------------------------------------#
        if requires_qkv:
            query_states = query_states.view(
                bsz, self.num_heads, tgt_len, self.head_dim
            )
            query_states = query_states.transpose(1, 2)
            query_states = query_states.reshape(bsz, tgt_len, embed_dim)
            query_states = self.out_proj(query_states)
            key_states = key_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
            key_states = key_states.transpose(1, 2)
            key_states = key_states.reshape(bsz, tgt_len, embed_dim)
            key_states = self.out_proj(key_states)
            value_states = value_states.view(
                bsz, self.num_heads, tgt_len, self.head_dim
            )
            value_states = value_states.transpose(1, 2)
            value_states = value_states.reshape(bsz, tgt_len, embed_dim)
            value_states = self.out_proj(value_states)
            return (
                attn_output,
                attn_weights_reshaped,
                (query_states, key_states, value_states),
            )
        else:
            return (
                attn_output,
                attn_weights_reshaped,
                (None, None, None),
            )
        # ----------------------------------------------------------#


class MaskCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MaskCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        requires_qkv: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # ----------------------------------------------------------#
        hidden_states, attn_weights, (query_states, key_states, value_states) = (
            self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                requires_qkv=requires_qkv,
            )
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # ----------------------------------------------------------#
        if requires_qkv:
            value_states = value_states + hidden_states
            # value_states = self.mlp(self.layer_norm2(value_states)) + value_states
        # ----------------------------------------------------------#

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs, (query_states, key_states, value_states)


class MaskCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MaskCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskCLIPWithBaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # ----------------------------------------------------------#
            if idx == len(self.layers) - 1:
                requires_qkv = True
            else:
                requires_qkv = False
            if self.gradient_checkpointing and self.training:
                layer_outputs, (query_states, key_states, value_states) = (
                    self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions,
                        requires_qkv,
                    )
                )
            else:
                layer_outputs, (query_states, key_states, value_states) = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                    requires_qkv=requires_qkv,
                )
            # ----------------------------------------------------------#

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    encoder_states,
                    all_attentions,
                    query_states,
                    key_states,
                    value_states,
                ]
                if v is not None
            )
        return MaskCLIPWithBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )


class MaskCLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = MaskCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MaskCLIPWithBaseModelOutputWithPooling,
        config_class=CLIPVisionConfig,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # ----------------------------------------------------------#
        value_states = encoder_outputs[-1]
        image_features = value_states[:, 1:, :]
        # ----------------------------------------------------------#

        if not return_dict:
            return (
                (last_hidden_state, pooled_output)
                + encoder_outputs[1:3]
                + image_features
            )

        return MaskCLIPWithBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            image_features=image_features,
        )


class MaskCLIPModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        vision_config = config.vision_config
        self.vision_model = MaskCLIPVisionTransformer(vision_config)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = vision_outputs[1]  # pooled_output
        # image_features = self.visual_projection(pooled_output)
        image_features = vision_outputs[-1]
        image_features = self.visual_projection(image_features)

        return image_features

    @torch.no_grad()
    def forward(
        self,
        text_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_embeds = self.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = image_embeds @ text_embeds.t() * logit_scale
        # logits_per_text = logits_per_image.t()
        logits_per_image = logits_per_image.permute(0, 2, 1)

        return logits_per_image


class MaskCLIP(nn.Module):
    def __init__(self, maskclip, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.is_initialized_negative_text_embeds = False

        self.maskclip_name = maskclip
        self.args = args

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.maskclip_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.maskclip_name
                )
            )
            return

        self.maskclip_processor = CLIPProcessor.from_pretrained(self.maskclip_name)
        self.maskclip_tokenizer = CLIPTokenizer.from_pretrained(self.maskclip_name)
        self.maskclip = MaskCLIPModel.from_pretrained(
            self.maskclip_name, device_map=device_map
        )
        self.maskclip.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def initialize_negative_text_embeds(self):
        # if self.is_initialized_negative_text_embeds:
        #     print(
        #         "negative_text_embeds is already loaded, `load_model` called again, skipping."
        #     )
        #     return

        negative_classes = list(
            set(stuff_classes() + all_pascal_context_classes() + bg_classes())
        )
        negative_classes_names = [
            "This is a photo of the " + class_name for class_name in negative_classes
        ]
        inputs = self.maskclip_tokenizer(
            negative_classes_names,
            padding=True,
            return_tensors="pt",
        ).to(self.maskclip.device)
        self.negative_text_embeds = self.maskclip.get_text_features(**inputs)

        self.is_initialized_negative_text_embeds = True

    @torch.no_grad()
    def get_full_text_embeds(self, text):
        inputs = self.maskclip_tokenizer(
            text,
            return_tensors="pt",
            max_length=77,
            truncation=True,
        ).to(self.maskclip.device)
        postive_text_embeds = self.maskclip.get_text_features(**inputs)
        self.negative_text_embeds = self.negative_text_embeds.to(self.maskclip.device)
        full_text_embeds = torch.cat(
            [postive_text_embeds, self.negative_text_embeds], dim=0
        )

        similarity = F.cosine_similarity(
            full_text_embeds[1:],
            full_text_embeds[:1].expand_as(full_text_embeds[1:]),
            dim=1,
        )
        _, closest_indices = torch.topk(
            similarity, self.args.similar_postive_num, largest=True
        )
        closest_indices += 1
        mask = torch.ones(full_text_embeds.size(0), dtype=torch.bool)
        mask[closest_indices] = False
        filtered_text_embeds = full_text_embeds[mask]
        return filtered_text_embeds

    def get_processed_image(self, image):
        return self.maskclip_processor(
            images=image,
            return_tensors="pt",
        )

    @torch.no_grad()
    def get_topk_mask(self, text_embeds, pixel_values, k):
        output = self.maskclip(text_embeds, pixel_values)
        B, C, Nt = output.shape
        softmax_output = F.softmax(output, dim=1)
        class_0_data = softmax_output[:, 0, :].squeeze()
        top_k_values, _ = torch.topk(class_0_data.view(-1), k)
        threshold = top_k_values.min()
        mask = (class_0_data >= threshold).int()
        return mask

    @torch.no_grad()
    def get_full_mask(self, text_embeds, pixel_values):
        pass

    def forward(self, text_embeds, pixel_values, k=None):
        if k is not None:
            return self.get_topk_mask(text_embeds, pixel_values, k)
        else:
            return self.get_full_mask(text_embeds, pixel_values)
