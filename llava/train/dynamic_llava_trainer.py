import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

import math
from typing import Dict, List, Optional

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_torch_tpu_available,
    LengthGroupedSampler,
)

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments

from functools import partial


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


@torch.no_grad()
def forward_hook(forward_data, module, input, output):
    # print(f"Inside {module.__class__.__name__} forward")
    forward_data["inputs"] = input
    forward_data["outputs"] = output


class CustomCallback(TrainerCallback):

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training.
        """
        self.forward_data = {"inputs": None, "outputs": None}
        forward_hook_function = partial(forward_hook, self.forward_data)
        self.inserted_forward_hook = kwargs["model"].model.register_forward_hook(
            forward_hook_function
        )

        # self.forward_data_1 = {"inputs": None, "outputs": None}
        # forward_hook_function_1 = partial(forward_hook, self.forward_data_1)
        # self.inserted_forward_hook_1 = kwargs["model"].register_forward_hook(
        #     forward_hook_function_1
        # )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.
        """
        self.inserted_forward_hook.remove()
        # self.inserted_forward_hook_1.remove()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        kwargs["model"].model.gumbel_tau = args.gumbel_start_tau * math.pow(
            args.gumbel_end_tau / args.gumbel_start_tau,
            state.global_step / state.max_steps,
        )

        # self.forward_data = {"inputs": [], "outputs": []}

        # self.forward_data = {"inputs": [], "outputs": []}
        # forward_hook_function = partial(forward_hook, self.forward_data)
        # self.inserted_forward_hook = kwargs["model"].model.register_forward_hook(
        #     forward_hook_function
        # )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        # args.forward_data = self.forward_data
        # self.inserted_forward_hook.remove()
        pass


class DynamicLLaVATrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_callback(CustomCallback)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # if self.args.mm_projector_lr is not None:
            #     projector_parameters = [
            #         name
            #         for name, _ in opt_model.named_parameters()
            #         if "mm_projector" in name
            #     ]
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (
            #                     n in decay_parameters
            #                     and n not in projector_parameters
            #                     and p.requires_grad
            #                 )
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (
            #                     n not in decay_parameters
            #                     and n not in projector_parameters
            #                     and p.requires_grad
            #                 )
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (
            #                     n in decay_parameters
            #                     and n in projector_parameters
            #                     and p.requires_grad
            #                 )
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #             "lr": self.args.mm_projector_lr,
            #         },
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (
            #                     n not in decay_parameters
            #                     and n in projector_parameters
            #                     and p.requires_grad
            #                 )
            #             ],
            #             "weight_decay": 0.0,
            #             "lr": self.args.mm_projector_lr,
            #         },
            #     ]
            # else:
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (n in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p
            #                 for n, p in opt_model.named_parameters()
            #                 if (n not in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #     ]
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            predictor_parameters = [
                name for name, _ in opt_model.named_parameters() if "predictor" in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in predictor_parameters
                            and n in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "lr": self.args.predictor_lr,
                    "weight_decay": self.args.predictor_weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in predictor_parameters
                            and n not in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "lr": self.args.predictor_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in predictor_parameters
                            and n in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in predictor_parameters
                            and n not in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            # ----------------------------------------------------------#
            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )

            with torch.no_grad():
                # mask loss
                image_masks = (
                    self.callback_handler.__dict__["callbacks"][-1]
                    .forward_data["outputs"]
                    .image_masks
                )
                if image_masks is not None and len(image_masks):
                    image_masks_loss = 0.0
                    for mask in image_masks:
                        batch_ratio = mask.mean(dim=1)
                        image_masks_loss = (
                            image_masks_loss
                            + (
                                (
                                    self.model.config.sparse_config["vision_keep_rate"]
                                    - batch_ratio
                                )
                                ** 2
                            ).mean()
                        ).item()
                    image_masks_loss = (
                        self.model.config.sparse_config["mask_loss_weight"]
                        * image_masks_loss
                    )
                    logs["image_masks_loss"] = round(image_masks_loss, 4)

                output_text_masks_batch_list = (
                    self.callback_handler.__dict__["callbacks"][-1]
                    .forward_data["outputs"]
                    .output_text_masks_batch_list
                )
                if output_text_masks_batch_list is not None and len(
                    output_text_masks_batch_list
                ):
                    output_text_mask_loss = 0.0
                    for mask_batch_list in output_text_masks_batch_list:
                        batch_ratio = torch.stack(
                            [mask.mean() for mask in mask_batch_list]
                        )
                        output_text_mask_loss = (
                            output_text_mask_loss
                            + (
                                (
                                    self.model.config.sparse_config[
                                        "output_text_keep_rate"
                                    ]
                                    - batch_ratio
                                )
                                ** 2
                            ).mean()
                        )
                    output_text_mask_loss = (
                        self.model.config.sparse_config["mask_loss_weight"]
                        * output_text_mask_loss
                    )
                    logs["output_text_mask_loss"] = round(output_text_mask_loss, 4)

            logs["llm_learning_rate"] = self.lr_scheduler.get_last_lr()[2]
            logs["predictor_learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            logs["gumbel_tau"] = round(self.model.model.gumbel_tau, 4)
            # ----------------------------------------------------------#

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        else:
            super(DynamicLLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(DynamicLLaVATrainer, self)._save(output_dir, state_dict)
