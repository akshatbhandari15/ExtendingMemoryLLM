"""
Importance-Aware Memory Dropping Strategies for MemoryLLM v1.

Extends the base MemoryLLM class with four dropping strategies:
  1. random     - baseline (original MemoryLLM behavior)
  2. attention  - drop tokens with lowest accumulated attention scores (EMA)
  3. age        - protect recent tokens, preferentially drop older ones
  4. surprise   - drop tokens most similar (redundant) to incoming knowledge
  5. fisher     - drop tokens whose removal least affects output distribution

Usage:
    from modeling_memoryllm_strategies import MemoryLLMWithStrategies
    model = MemoryLLMWithStrategies.from_pretrained("YuWangX/memoryllm-8b")
    model.set_drop_strategy("attention")  # or "age", "surprise", "fisher", "random"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from modeling_memoryllm import MemoryLLM


class MemoryLLMWithStrategies(MemoryLLM):
    """MemoryLLM with pluggable importance-aware dropping strategies."""

    STRATEGIES = ("random", "attention", "age", "surprise", "fisher")

    def __init__(self, config):
        super().__init__(config)

        N = self.num_blocks * self.num_tokens  # total memory pool size per layer

        # --- Metadata for importance-aware dropping ---
        # Attention EMA scores: [L, N] - accumulated attention received per token
        self._attention_ema = [torch.zeros(N) for _ in range(self.L)]
        self._attention_alpha = 0.9  # EMA decay factor

        # Age tracking: [L, N] - number of update steps since each token was injected
        self._token_ages = [np.zeros(N, dtype=np.int64) for _ in range(self.L)]

        # Current strategy
        self._drop_strategy = "random"

        # Age-stratified config
        self._age_protection_window = self.num_tokens  # protect last P injected tokens

        # Fisher config
        self._fisher_scores = [torch.zeros(N) for _ in range(self.L)]
        self._fisher_update_interval = 10  # recompute every N updates
        self._fisher_update_counter = 0
        self._fisher_num_probes = 5  # number of probe inputs for Fisher estimation

        # Surprise: computed on-the-fly in update_memory_with_delta_memory, no persistent state needed

        # Track total update steps
        self._update_count = 0

    def set_drop_strategy(self, strategy: str, **kwargs):
        """Set the dropping strategy. Options: random, attention, age, surprise, fisher."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")
        self._drop_strategy = strategy

        # Allow overriding strategy-specific params
        if "alpha" in kwargs:
            self._attention_alpha = kwargs["alpha"]
        if "protection_window" in kwargs:
            self._age_protection_window = kwargs["protection_window"]
        if "fisher_interval" in kwargs:
            self._fisher_update_interval = kwargs["fisher_interval"]
        if "fisher_num_probes" in kwargs:
            self._fisher_num_probes = kwargs["fisher_num_probes"]

        print(f"Drop strategy set to: {strategy}")

    def get_strategy_info(self):
        """Return current strategy and metadata summary."""
        info = {"strategy": self._drop_strategy, "update_count": self._update_count}
        if self._drop_strategy == "attention":
            scores = self._attention_ema[0]
            info["attention_score_stats"] = {
                "mean": scores.mean().item(),
                "std": scores.std().item(),
                "min": scores.min().item(),
                "max": scores.max().item(),
            }
        elif self._drop_strategy == "age":
            ages = self._token_ages[0]
            info["age_stats"] = {
                "mean": ages.mean(),
                "max": ages.max(),
                "protected": int((ages <= self._age_protection_window).sum()),
            }
        return info

    # -------------------------------------------------------------------------
    # Core: importance-aware drop_memory
    # -------------------------------------------------------------------------
    def drop_memory(self, current_memory, drop_length=None, unsequeezed=True,
                    layer_idx=None, delta_memory=None):
        """
        Drop tokens from memory using the current strategy.

        Args:
            current_memory: tensor of shape [1, N, d] (unsequeezed) or [N, d]
            drop_length: number of tokens to drop (default: N / num_blocks)
            unsequeezed: whether current_memory has a batch dimension
            layer_idx: which layer (needed for per-layer metadata)
            delta_memory: incoming new tokens (needed for surprise strategy)

        Returns:
            Tuple of (remaining_memory, remaining_indices, dropped_indices)
        """
        if unsequeezed:
            N = current_memory.shape[1]
        else:
            N = current_memory.shape[0]

        if drop_length is None:
            drop_length = int(N * (1 / self.num_blocks))

        keep_length = N - drop_length

        # Compute importance scores based on strategy
        importance = self._compute_importance(layer_idx, N, current_memory, delta_memory)

        # Select indices to keep (highest importance) and drop (lowest importance)
        sorted_indices = importance.argsort(descending=True)
        remaining_indices = sorted_indices[:keep_length].sort()[0]
        dropped_indices = sorted_indices[keep_length:].sort()[0]

        if unsequeezed:
            current_memory = current_memory[:, remaining_indices, :]
        else:
            current_memory = current_memory[remaining_indices, :]

        return current_memory, remaining_indices, dropped_indices

    def _compute_importance(self, layer_idx, N, current_memory, delta_memory):
        """Compute per-token importance scores for the given strategy."""

        if self._drop_strategy == "random":
            return torch.rand(N)

        elif self._drop_strategy == "attention":
            if layer_idx is not None and layer_idx < len(self._attention_ema):
                scores = self._attention_ema[layer_idx][:N].clone()
                # Add small noise to break ties
                scores += torch.rand(N) * 1e-6
                return scores
            return torch.rand(N)

        elif self._drop_strategy == "age":
            if layer_idx is not None and layer_idx < len(self._token_ages):
                ages = self._token_ages[layer_idx][:N]
                importance = torch.zeros(N)
                for i in range(N):
                    if ages[i] <= self._age_protection_window:
                        # Recently injected: high importance (protected)
                        importance[i] = float('inf')
                    else:
                        # Older tokens: importance decreases with age
                        # Inverse age = younger tokens more important
                        importance[i] = 1.0 / (ages[i] + 1)
                # Add small noise to break ties among protected tokens
                importance += torch.rand(N) * 1e-8
                return importance
            return torch.rand(N)

        elif self._drop_strategy == "surprise":
            if delta_memory is not None:
                # Compute cosine similarity between each existing token and incoming tokens
                # current_memory: [N, d] or [1, N, d]
                if current_memory.dim() == 3:
                    mem = current_memory[0]  # [N, d]
                else:
                    mem = current_memory  # [N, d]

                # delta_memory could be [L, K, d] or [1, L, K, d] - extract this layer
                if delta_memory.dim() == 4:
                    new_tokens = delta_memory[0, layer_idx] if layer_idx is not None else delta_memory[0, 0]
                elif delta_memory.dim() == 3:
                    new_tokens = delta_memory[layer_idx] if layer_idx is not None else delta_memory[0]
                else:
                    new_tokens = delta_memory

                # Normalize
                mem_norm = F.normalize(mem.float(), dim=-1)
                new_norm = F.normalize(new_tokens.float(), dim=-1)

                # Max cosine similarity to any incoming token
                # High similarity = redundant with new knowledge = drop it
                sim = torch.mm(mem_norm, new_norm.t())  # [N, K]
                max_sim = sim.max(dim=1)[0]  # [N]

                # Importance = 1 - max_similarity (most redundant = least important)
                importance = 1.0 - max_sim
                importance += torch.rand(N, device=importance.device) * 1e-6
                return importance.cpu()
            return torch.rand(N)

        elif self._drop_strategy == "fisher":
            if layer_idx is not None and layer_idx < len(self._fisher_scores):
                scores = self._fisher_scores[layer_idx][:N].clone()
                scores += torch.rand(N) * 1e-6
                return scores
            return torch.rand(N)

        return torch.rand(N)

    # -------------------------------------------------------------------------
    # Override update_memory_with_delta_memory to track metadata
    # -------------------------------------------------------------------------
    def update_memory_with_delta_memory(self, delta_memory):
        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        if self.initialized == 0:
            # First injection: fill memory pool
            if delta_memory.shape[1] < (self.num_tokens * self.num_blocks):
                if ((self.num_tokens * self.num_blocks) % delta_memory.shape[1]) == 0:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_tokens * self.num_blocks) // delta_memory.shape[1]), dim=1
                    )
                else:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_tokens * self.num_blocks) // delta_memory.shape[1]) +
                        [delta_memory[:, -((self.num_tokens * self.num_blocks) % delta_memory.shape[1]):]], dim=1
                    )
            else:
                delta_memory = delta_memory[:, -self.num_tokens * self.num_blocks:]

            self.memory.data = delta_memory

            # Initialize metadata
            N = self.num_tokens * self.num_blocks
            for idx in range(self.L):
                self._attention_ema[idx] = torch.zeros(N)
                self._token_ages[idx] = np.zeros(N, dtype=np.int64)
                self._fisher_scores[idx] = torch.zeros(N)

        else:
            K = delta_memory.shape[1]

            if self.drop_memory_per_layer:
                for idx in range(len(self.memory)):
                    current_memory = self.memory.data[idx].detach()

                    current_memory, remaining_indices, dropped_indices = self.drop_memory(
                        current_memory,
                        drop_length=K,
                        unsequeezed=False,
                        layer_idx=idx,
                        delta_memory=delta_memory
                    )

                    self.memory.data[idx] = torch.cat([current_memory, delta_memory[idx]], dim=0)

                    # Update metadata for this layer
                    self._update_metadata_after_drop(idx, remaining_indices, dropped_indices, K)

            else:
                current_memory = self.memory.data.detach()

                current_memory, remaining_indices, dropped_indices = self.drop_memory(
                    current_memory,
                    drop_length=K,
                    unsequeezed=True,
                    layer_idx=0,  # use layer 0 scores for shared dropping
                    delta_memory=delta_memory
                )

                if current_memory.device != delta_memory.device:
                    self.memory.data = torch.cat([current_memory, delta_memory.to(current_memory.device)], dim=1)
                else:
                    self.memory.data = torch.cat([current_memory, delta_memory], dim=1)

                # Update metadata for all layers (shared indices)
                for idx in range(self.L):
                    self._update_metadata_after_drop(idx, remaining_indices, dropped_indices, K)

        if not self.initialized:
            self.initialized += 1

        self._update_count += 1

    def _update_metadata_after_drop(self, layer_idx, remaining_indices, dropped_indices, num_new_tokens):
        """Update ages, attention scores, and fisher scores after a drop operation."""
        remaining_indices_np = remaining_indices.cpu().numpy()

        # Ages: keep remaining, add age 0 for new tokens, increment all
        old_ages = self._token_ages[layer_idx][remaining_indices_np]
        new_ages = np.zeros(num_new_tokens, dtype=np.int64)
        self._token_ages[layer_idx] = np.concatenate([old_ages + 1, new_ages])

        # Attention EMA: keep remaining scores, initialize new tokens to mean
        old_scores = self._attention_ema[layer_idx][remaining_indices]
        mean_score = old_scores.mean() if len(old_scores) > 0 else 0.0
        new_scores = torch.full((num_new_tokens,), mean_score)
        self._attention_ema[layer_idx] = torch.cat([old_scores, new_scores])

        # Fisher scores: keep remaining, initialize new to mean
        old_fisher = self._fisher_scores[layer_idx][remaining_indices]
        mean_fisher = old_fisher.mean() if len(old_fisher) > 0 else 0.0
        new_fisher = torch.full((num_new_tokens,), mean_fisher)
        self._fisher_scores[layer_idx] = torch.cat([old_fisher, new_fisher])

    # -------------------------------------------------------------------------
    # Attention tracking: called after forward passes to accumulate EMA
    # -------------------------------------------------------------------------
    def update_attention_scores(self, attentions):
        """
        Update attention EMA from a forward pass that returned attention weights.

        Args:
            attentions: tuple of [batch, heads, seq_len, seq_len] per layer.
                        The first N columns correspond to memory tokens.
        """
        if attentions is None:
            return

        N = self.num_blocks * self.num_tokens
        has_bos = 1 if self.add_bos_embedding else 0
        memory_end = has_bos + N  # memory tokens occupy positions [has_bos : has_bos + N]

        alpha = self._attention_alpha
        for idx, layer_attn in enumerate(attentions):
            if idx >= self.L:
                break
            # layer_attn: [batch, heads, total_seq, total_seq]
            # Attention from input tokens to memory tokens
            # Input tokens start at position memory_end
            if layer_attn.shape[-1] > memory_end:
                # Mean attention over heads and input query positions to each memory token
                attn_to_memory = layer_attn[:, :, memory_end:, has_bos:memory_end]  # [B, H, input_len, N]
                avg_attn = attn_to_memory.mean(dim=(0, 1, 2))  # [N]
                avg_attn = avg_attn.detach().cpu().float()

                if len(avg_attn) == len(self._attention_ema[idx]):
                    self._attention_ema[idx] = alpha * self._attention_ema[idx] + (1 - alpha) * avg_attn

    def update_fisher_scores(self, input_ids, attention_mask=None):
        """
        Estimate Fisher-inspired importance by measuring KL divergence when
        masking blocks of memory tokens.

        This is expensive — call periodically, not every update.
        """
        N = self.num_blocks * self.num_tokens
        block_size = self.num_tokens  # mask one block at a time

        with torch.no_grad():
            # Get baseline output distribution
            baseline_out = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            baseline_logits = baseline_out.logits[:, -1, :].float()  # last token logits
            baseline_probs = F.softmax(baseline_logits, dim=-1)

            # For each block, mask it and measure distribution shift
            num_full_blocks = N // block_size
            for block_idx in range(num_full_blocks):
                start = block_idx * block_size
                end = start + block_size

                # Save and zero out the block
                for layer_idx in range(self.L):
                    saved = self.memory.data[layer_idx, start:end, :].clone()
                    self.memory.data[layer_idx, start:end, :] = 0

                # Forward pass with masked memory
                masked_out = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                masked_logits = masked_out.logits[:, -1, :].float()
                masked_probs = F.softmax(masked_logits, dim=-1)

                # KL divergence: higher = more important
                kl = F.kl_div(
                    masked_probs.log(), baseline_probs,
                    reduction='batchmean'
                ).item()

                # Assign importance score to all tokens in this block
                for layer_idx in range(self.L):
                    self._fisher_scores[layer_idx][start:end] = kl
                    # Restore memory
                    self.memory.data[layer_idx, start:end, :] = saved

    # -------------------------------------------------------------------------
    # Override inject_memory to track attention scores
    # -------------------------------------------------------------------------
    def inject_memory(self, context_ids,
                      context_attention_mask=None,
                      delta_memory=None,
                      update_memory=False):
        """Inject memory and optionally track attention for importance scoring."""

        # For attention strategy, we need attention weights during injection
        need_attention = (self._drop_strategy == "attention" and self.initialized)

        output = self(input_ids=context_ids,
                      attention_mask=context_attention_mask,
                      delta_memory=delta_memory,
                      is_injection=True,
                      output_delta_memory=True,
                      output_attentions=need_attention,
                      return_dict=True)

        # Update attention EMA if available
        if need_attention and output.attentions is not None:
            self.update_attention_scores(output.attentions)

        if update_memory:
            self.update_memory_with_delta_memory(output.delta_memory)
            return output.delta_memory
        else:
            return output.delta_memory

    # -------------------------------------------------------------------------
    # Utility: reset metadata (useful for evaluation)
    # -------------------------------------------------------------------------
    def reset_metadata(self):
        """Reset all tracking metadata. Call when starting a new evaluation run."""
        N = self.num_blocks * self.num_tokens
        for idx in range(self.L):
            self._attention_ema[idx] = torch.zeros(N)
            self._token_ages[idx] = np.zeros(N, dtype=np.int64)
            self._fisher_scores[idx] = torch.zeros(N)
        self._update_count = 0
