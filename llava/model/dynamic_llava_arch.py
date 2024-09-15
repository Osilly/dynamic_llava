#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape

from .maskclip_hf.builder import build_maskclip


special_text = {"ASSISTANT:": [319, 1799, 9047, 13566, 29901], "USER:": [11889, 29901]}


class DynamicLlavaMetaModel:

    def __init__(self, config):
        super(DynamicLlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

    def get_maskclip(self):
        maskclip = getattr(self, "maskclip", None)
        llm_tokenizer = getattr(self, "llm_tokenizer", None)
        return (
            maskclip,
            llm_tokenizer,
        )

    def initialize_maskclip(self, sparse_args, tokenizer, fsdp=None):
        # self.maskclip_model, self.maskclip_preprocess = maskclip.load(
        #     sparse_args.maskclip,
        #     templates=None,
        #     device=device,
        # )
        # full_negative_classes = list(
        #     set(stuff_classes() + all_pascal_context_classes() + bg_classes())
        # )
        # self.full_negative_classes_names = [
        #     "This is a photo of the " + class_name
        #     for class_name in full_negative_classes
        # ]
        self.maskclip = build_maskclip(sparse_args.maskclip, sparse_args)
        # self.maskclip.initialize_negative_text_embeds()
        self.llm_tokenizer = tokenizer


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class DynamicLlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_maskclip(self):
        return self.get_model().get_maskclip()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    # ----------------------------------------------------------#
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            ), (None, None)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        input_embeds_indices = []
        cur_image_idx = 0

        # new_system_embeds = []
        # new_system_labels = []
        # new_image_embeds = []
        # new_image_labels = []
        # new_instruct_embeds = []
        # new_instruct_labels = []
        # new_answer_embeds = []
        # new_answer_labels = []

        # ----------------------------------------------------------#
        maskclip_mask = []
        (
            maskclip,
            llm_tokenizer,
        ) = self.get_maskclip()
        # ----------------------------------------------------------#

        for batch_idx in range(len(input_ids)):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx]

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            # # for parallel training
            # if num_images == 0:
            #     image_features[cur_image_idx] = torch.zeros_like(
            #         image_features[cur_image_idx]
            #     )
            #     cur_input_ids = torch.cat(
            #         [
            #             cur_input_ids[:34],
            #             torch.tensor([29871, IMAGE_TOKEN_INDEX, 29871]).to(
            #                 device=cur_input_ids.device, dtype=cur_input_ids.dtype
            #             ),
            #             cur_input_ids[34:],
            #         ],
            #         dim=0,
            #     )
            #     cur_labels = torch.cat(
            #         [
            #             cur_labels[:34],
            #             torch.tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]).to(
            #                 device=cur_labels.device, dtype=cur_labels.dtype
            #             ),
            #             cur_labels[34:],
            #         ],
            #         dim=0,
            #     )
            #     num_images = 1

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # if num_images == 0:
            #     cur_image_idx += 1
            #     continue

            image_start_position = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[
                0
            ].item()
            instruct_start_position = image_start_position + 1
            # answer_start_position = (
            #     instruct_start_position
            #     + (cur_labels == IGNORE_INDEX).sum().item()
            #     - image_start_position
            #     - 1
            # )
            answer_start_position = torch.where(cur_labels == -100)[0][-1].item() + 1
            # cur_image_len = cur_image_features.shape[0]

            input_ids_indices = [
                (0, image_start_position),
                (instruct_start_position, answer_start_position),
                (answer_start_position, cur_input_ids.shape[0]),
            ]

            cur_system_ids = cur_input_ids[
                input_ids_indices[0][0] : input_ids_indices[0][1]
            ]
            cur_system_labels = cur_labels[
                input_ids_indices[0][0] : input_ids_indices[0][1]
            ]

            cur_instruct_ids = cur_input_ids[
                input_ids_indices[1][0] : input_ids_indices[1][1]
            ]
            cur_instruct_labels = cur_labels[
                input_ids_indices[1][0] : input_ids_indices[1][1]
            ]

            cur_answer_ids = cur_input_ids[
                input_ids_indices[2][0] : input_ids_indices[2][1]
            ]
            cur_answer_labels = cur_labels[
                input_ids_indices[2][0] : input_ids_indices[2][1]
            ]

            cur_text_embeds = self.get_model().embed_tokens(
                torch.cat([cur_system_ids, cur_instruct_ids, cur_answer_ids])
            )

            cur_system_embeds = cur_text_embeds[
                input_ids_indices[0][0] : input_ids_indices[0][1]
            ]

            # considering multi-images
            cur_image_embeds = []
            cur_image_labels = []
            for i in range(num_images):
                cur_image_features = image_features[cur_image_idx].to(self.device)
                cur_image_embeds.append(cur_image_features)
                cur_image_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                cur_image_idx += 1
            cur_image_embeds = torch.cat(cur_image_embeds)
            cur_image_labels = torch.cat(cur_image_labels)

            cur_instruct_embeds = cur_text_embeds[
                input_ids_indices[1][0] - 1 : input_ids_indices[1][1] - 1
            ]

            cur_answer_embeds = cur_text_embeds[
                input_ids_indices[2][0] - 1 : input_ids_indices[2][1] - 1
            ]

            cur_new_input_embeds = torch.cat(
                [
                    cur_system_embeds,
                    cur_image_embeds,
                    cur_instruct_embeds,
                    cur_answer_embeds,
                ]
            )
            cur_new_labels = torch.cat(
                [
                    cur_system_labels,
                    cur_image_labels,
                    cur_instruct_labels,
                    cur_answer_labels,
                ]
            )
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            # ----------------------------------------------------------#
            special_user = torch.tensor(
                special_text["USER:"], device=cur_instruct_ids.device
            )
            special_user_seq_len = len(special_user)
            special_user_matches = [
                i
                for i in range(len(cur_instruct_ids) - special_user_seq_len + 1)
                if torch.equal(
                    cur_instruct_ids[i : i + special_user_seq_len], special_user
                )
            ]
            special_user_starts = special_user_matches
            # special_user_ends = [
            #     m + special_user_seq_len - 1 for m in special_user_matches
            # ]

            # special_assistant = torch.tensor(
            #     special_text["ASSISTANT:"], device=cur_instruct_ids.device
            # )
            # special_assistant_seq_len = len(special_assistant)
            # special_assistant_matches = [
            #     i
            #     for i in range(len(cur_instruct_ids) - special_assistant_seq_len + 1)
            #     if torch.equal(
            #         cur_instruct_ids[i : i + special_assistant_seq_len],
            #         special_assistant,
            #     )
            # ]
            # special_assistant_starts = special_assistant_matches
            # special_assistant_ends = [
            #     m + special_assistant_seq_len - 1 for m in special_assistant_matches
            # ]

            if len(special_user_starts):
                cur_last_instruct_start_position = special_user_starts[-1]
            else:
                cur_last_instruct_start_position = 0
            # cur_last_instruct_end_position = special_assistant_starts[-1]

            # if maskclip is not None:
            #     maskclip.initialize_negative_text_embeds()

            #     cur_last_instruct_ids = cur_instruct_ids[
            #         cur_last_instruct_start_position:cur_last_instruct_end_position
            #     ]

            #     cur_postive_classes_names = [
            #         llm_tokenizer.decode(
            #             torch.cat([cur_last_instruct_ids, cur_answer_ids[:-1]])
            #         )
            #     ]

            #     cur_maskclip_text_embeds = maskclip.get_full_text_embeds(
            #         cur_postive_classes_names
            #     )
            #     cur_image = images[cur_image_idx - 1 : cur_image_idx]
            #     if self.config.sparse_config["maskclip_distill_token_rate"] is not None:
            #         k = int(
            #             cur_image_embeds.shape[0]
            #             * self.config.sparse_config["maskclip_distill_token_rate"]
            #         )
            #     else:
            #         k = None

            #     cur_maskclip_mask = maskclip(cur_maskclip_text_embeds, cur_image, k=k)

            #     maskclip_mask.append(cur_maskclip_mask)

            # ----------------------------------------------------------#

            # embedding indices
            image_embeds_start_position = cur_system_embeds.shape[0]
            instruct_embeds_start_position = (
                image_embeds_start_position + cur_image_embeds.shape[0]
            )
            answer_embeds_start_position = (
                instruct_embeds_start_position + cur_instruct_embeds.shape[0]
            )
            last_instruct_embeds_start_position = (
                instruct_embeds_start_position + cur_last_instruct_start_position
            )
            # last_instruct_embeds_end_position = (
            #     instruct_embeds_start_position + cur_last_instruct_end_position
            # )
            cur_input_embeds_indices = {
                "system": [0, image_embeds_start_position],
                "image": [image_embeds_start_position, instruct_embeds_start_position],
                "instruct": [
                    instruct_embeds_start_position,
                    answer_embeds_start_position,
                ],
                "answer": [
                    answer_embeds_start_position,
                    cur_new_input_embeds.shape[0],
                ],
                "last_instruct": [
                    last_instruct_embeds_start_position,
                    answer_embeds_start_position,
                    # last_instruct_embeds_end_position,
                ],
            }
            input_embeds_indices.append(cur_input_embeds_indices)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # prepare for indices
            for x in input_embeds_indices:
                for key, value in x.items():
                    value[0] = min(value[0], tokenizer_model_max_length)
                    value[1] = min(value[1], tokenizer_model_max_length)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )

                # prepare for indices
                for key, value in input_embeds_indices[i].items():
                    value[0] += max_len - cur_len
                    value[1] += max_len - cur_len

                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if len(maskclip_mask):
            maskclip_mask = torch.stack(maskclip_mask, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        ), (input_embeds_indices, maskclip_mask)

    # ----------------------------------------------------------#

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
