from typing import Any, Dict, List, Optional, Tuple

import torch


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError(
            "Make sure to implement `get_max_length` in a subclass."
        )

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length


class DynamicCachePlus(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

        # ----------------------------------------------------------#
        self.true_cache_length: List[torch.Tensor] = []  # L * [B]
        # ----------------------------------------------------------#

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        cache_decision: Optional[torch.Tensor] = None,  # [B * N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        B, _, N, _ = key_states.shape

        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

            # ----------------------------------------------------------#
            # for prefill stage
            if cache_decision is not None:
                self.true_cache_length.append(cache_decision.sum(dim=-1))
            else:
                self.true_cache_length.append(torch.tensor([N]).repeat(B))
            # ----------------------------------------------------------#
        else:
            # ----------------------------------------------------------#
            if cache_decision is not None:
                if B == 1 and N == 1:
                    if cache_decision[0, 0]:
                        self.key_cache[layer_idx] = torch.cat(
                            [self.key_cache[layer_idx], key_states], dim=-2
                        )
                        self.value_cache[layer_idx] = torch.cat(
                            [self.value_cache[layer_idx], value_states], dim=-2
                        )

                        self.true_cache_length[layer_idx] += N
                    else:
                        pass
                else:  # TODO, efficiency needs to be optimized
                    cur_layer_key_cache_batch_list = []
                    cur_layer_value_cache_batch_list = []
                    for b in range(B):
                        cur_keep_indice = cache_decision[b]
                        keep_key_states = key_states[
                            b, :, cur_keep_indice, :
                        ]  # H * N * C
                        keep_value_states = value_states[
                            b, :, cur_keep_indice, :
                        ]  # H * N * C
                        cur_layer_key_cache = torch.cat(
                            [
                                self.key_cache[layer_idx][
                                    b, :, : self.true_cache_length[layer_idx][b], :
                                ],
                                keep_key_states,
                            ],
                            dim=-2,
                        )
                        cur_layer_value_cache = torch.cat(
                            [
                                self.value_cache[layer_idx][
                                    b, :, : self.true_cache_length[layer_idx][b], :
                                ],
                                keep_value_states,
                            ],
                            dim=-2,
                        )
                        cur_layer_key_cache_batch_list.append(cur_layer_key_cache)
                        cur_layer_value_cache_batch_list.append(cur_layer_value_cache)

                        self.true_cache_length[layer_idx][b] += (
                            cache_decision[b].sum().item()
                        )

                    max_cur_layer_kv_cache_length = max(
                        cur_layer_key_cache_batch_list[b].shape[-2] for b in range(B)
                    )
                    for b in range(B):
                        cur_len = cur_layer_key_cache_batch_list[b].shape[-2]
                        cur_layer_key_cache_batch_list[b] = torch.cat(
                            [
                                cur_layer_key_cache_batch_list[b],
                                torch.zeros(
                                    (
                                        cur_layer_key_cache_batch_list[b].shape[0],
                                        max_cur_layer_kv_cache_length - cur_len,
                                        cur_layer_key_cache_batch_list[b].shape[-1],
                                    ),
                                    dtype=cur_layer_key_cache_batch_list[b].dtype,
                                    device=cur_layer_key_cache_batch_list[b].device,
                                ),
                            ],
                            dim=-2,
                        )
                        cur_layer_value_cache_batch_list[b] = torch.cat(
                            [
                                cur_layer_value_cache_batch_list[b],
                                torch.zeros(
                                    (
                                        cur_layer_value_cache_batch_list[b].shape[0],
                                        max_cur_layer_kv_cache_length - cur_len,
                                        cur_layer_value_cache_batch_list[b].shape[-1],
                                    ),
                                    dtype=cur_layer_value_cache_batch_list[b].dtype,
                                    device=cur_layer_value_cache_batch_list[b].device,
                                ),
                            ],
                            dim=-2,
                        )
                    self.key_cache[layer_idx] = torch.stack(
                        cur_layer_key_cache_batch_list
                    )
                    self.value_cache[layer_idx] = torch.stack(
                        cur_layer_value_cache_batch_list
                    )
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

                self.true_cache_length[layer_idx] += N
            # ----------------------------------------------------------#

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # ----------------------------------------------------------#
    def get_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            return key_states, value_states
        else:
            return torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            ), torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

    # ----------------------------------------------------------#

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )

    # ----------------------------------------------------------#
    def to_legacy_cache(
        self,
    ) -> Tuple[Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]], torch.Tensor]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return (legacy_cache, self.true_cache_length)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[
            Tuple[Tuple[Tuple[torch.FloatTensor]], torch.Tensor]
        ] = None,
    ) -> "DynamicCachePlus":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values[0])):
                key_states, value_states = past_key_values[0][layer_idx]
                cache.update(key_states, value_states, layer_idx)
            cache.true_cache_length = past_key_values[1]
        return cache

    # ----------------------------------------------------------#
