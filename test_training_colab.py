"""
Test script for MemoryLLM dropping strategies on Google Colab.

Profiles GPU memory usage and runs knowledge retention evaluation
using the actual SQuAD and NaturalQA datasets with all four strategies.

Supports two modes:
  - Pretrained MemoryLLM checkpoint (e.g. YuWangX/memoryllm-8b)
  - Base LLM + fresh memory pool (e.g. openlm-research/open_llama_3b_v2)

Usage on Colab:
    # Quick smoke test with 3B model (no datasets needed)
    python test_training_colab.py --base-model openlm-research/open_llama_3b_v2 --smoke-test

    # Full comparison with 3B model on SQuAD
    python test_training_colab.py --base-model openlm-research/open_llama_3b_v2 \
        --strategies random attention age surprise \
        --datasets squad --nuc 5 --num-samples 50

    # Use pretrained 8B checkpoint instead
    python test_training_colab.py --pretrained YuWangX/memoryllm-8b --smoke-test
"""

import argparse
import gc
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm


def print_gpu_stats(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{label}] Allocated: {allocated:.2f}GB | "
              f"Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB")


# =========================================================================
# Model loading
# =========================================================================

def load_model_from_base(base_model_path, num_blocks=4, num_tokens=256,
                         dtype=torch.bfloat16):
    """
    Initialize MemoryLLM with a fresh memory pool on top of a base LLM.
    This is what you'd do before training from scratch.

    Args:
        base_model_path: HuggingFace path like "openlm-research/open_llama_3b_v2"
                         or "meta-llama/Llama-3.2-3B"
        num_blocks: number of memory blocks (memory size = num_blocks * num_tokens)
        num_tokens: tokens per block (compression ratio)
        dtype: model dtype
    """
    from transformers import AutoTokenizer, AutoConfig
    from modeling_memoryllm_strategies import MemoryLLMWithStrategies
    from configuration_memoryllm import MemoryLLMConfig

    print(f"Loading base model: {base_model_path}")
    print(f"Memory config: {num_blocks} blocks x {num_tokens} tokens")

    # Load the base config and add MemoryLLM fields
    base_config = AutoConfig.from_pretrained(base_model_path)

    # Build a MemoryLLMConfig with the base model's architecture
    config = MemoryLLMConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads',
                                    base_config.num_attention_heads),
        hidden_act=getattr(base_config, 'hidden_act', 'silu'),
        max_position_embeddings=base_config.max_position_embeddings,
        rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        tie_word_embeddings=getattr(base_config, 'tie_word_embeddings',
                                    False),
        attention_bias=getattr(base_config, 'attention_bias', False),
        attention_dropout=getattr(base_config, 'attention_dropout', 0.0),
        mlp_bias=getattr(base_config, 'mlp_bias', False),
        # MemoryLLM-specific fields
        num_blocks=num_blocks,
        num_tokens=num_tokens,
        num_memory_tokens=num_blocks * num_tokens,
        add_bos_embedding=True,
        drop_memory_per_layer=True,
        add_decoder_lora=False,
        lora_config=None,
    )

    # Handle RoPE scaling if memory + input exceeds max_position_embeddings
    total_positions = (num_blocks * num_tokens) + 512 + 1  # memory + input + bos
    if total_positions > config.max_position_embeddings:
        factor = total_positions / config.max_position_embeddings
        config.rope_scaling = {"type": "linear", "factor": float(factor)}
        print(f"  RoPE scaling enabled: factor={factor:.2f}")

    # Load weights from base model into MemoryLLMWithStrategies
    model = MemoryLLMWithStrategies.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_model_pretrained(pretrained_path, dtype=torch.bfloat16):
    """Load a pretrained MemoryLLM checkpoint (e.g. memoryllm-8b)."""
    from transformers import AutoTokenizer
    from modeling_memoryllm_strategies import MemoryLLMWithStrategies

    print(f"Loading pretrained MemoryLLM: {pretrained_path}")

    model = MemoryLLMWithStrategies.from_pretrained(
        pretrained_path, torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model.to(dtype)

    return model, tokenizer


# =========================================================================
# Evaluation helpers
# =========================================================================

def calculate_exact_hit_accuracy(predictions, targets):
    hit = sum(1 for p, t in zip(predictions, targets)
              if t.replace("</s>", "").strip() in p)
    return hit / len(predictions) if predictions else 0.0


def collate_fn_qa(data, tokenizer, max_length, num_tokens,
                  add_special_tokens=False, end_special_token=None,
                  padding='longest'):
    eval_max_length = max_length
    contexts, questions, answers, unrelated_contexts = zip(*data)

    if end_special_token is not None:
        answers = [x + end_special_token for x in list(answers)]

    contexts_tokenized = tokenizer(
        list(contexts), max_length=max_length, padding=padding,
        truncation=True, return_tensors='pt',
        add_special_tokens=add_special_tokens)
    questions_tokenized = tokenizer(
        list(questions), max_length=eval_max_length, truncation=True,
        padding='longest', return_tensors='pt',
        add_special_tokens=add_special_tokens)
    answers_tokenized = tokenizer(
        list(answers), max_length=eval_max_length, truncation=True,
        padding='longest', return_tensors='pt',
        add_special_tokens=add_special_tokens)

    unrelated_contexts = np.array(unrelated_contexts).transpose().tolist()

    all_unrelated = {}
    all_unrelated_mask = {}
    for i in range(len(unrelated_contexts)):
        all_unrelated[i] = tokenizer(
            unrelated_contexts[i], max_length=max_length,
            truncation=True, padding=padding, return_tensors='pt',
            add_special_tokens=add_special_tokens)
        all_unrelated_mask[i] = torch.cat([
            torch.ones(1, num_tokens).repeat(
                all_unrelated[i].input_ids.shape[0], 1),
            all_unrelated[i].attention_mask
        ], dim=-1)

    ctx_mask = torch.cat([
        torch.ones(1, num_tokens).repeat(
            contexts_tokenized.input_ids.shape[0], 1),
        contexts_tokenized.attention_mask
    ], dim=-1)
    q_mask = torch.cat([
        torch.ones(1, num_tokens).repeat(
            contexts_tokenized.input_ids.shape[0], 1),
        questions_tokenized.attention_mask
    ], dim=-1)
    a_mask = torch.cat([
        torch.ones(1, num_tokens).repeat(
            contexts_tokenized.input_ids.shape[0], 1),
        answers_tokenized.attention_mask
    ], dim=-1)

    outputs = (contexts_tokenized.input_ids, ctx_mask,
               questions_tokenized.input_ids, q_mask,
               answers_tokenized.input_ids, a_mask)

    for i in range(len(all_unrelated)):
        outputs += (all_unrelated[i].input_ids,)
        outputs += (all_unrelated_mask[i],)

    return outputs


def ensure_index_files():
    squad_idx = "./data/squad/indices_squad_3.npy"
    nq_idx = "./data/nq/indices_nq_4.npy"
    if not os.path.exists(squad_idx):
        os.makedirs("./data/squad", exist_ok=True)
        np.save(squad_idx, np.arange(100))
        print(f"Created default {squad_idx}")
    if not os.path.exists(nq_idx):
        os.makedirs("./data/nq", exist_ok=True)
        np.save(nq_idx, np.arange(100))
        print(f"Created default {nq_idx}")


# =========================================================================
# Evaluation: Knowledge Retention Curves
# =========================================================================

def run_retention_eval(model, tokenizer, dataset_name,
                       num_unrelated_contexts, num_samples=50,
                       strategy="random", tokenizer_path=None):
    from dataset.nq import NQDataset
    from dataset.squad import SQuADDataset

    tok_path = tokenizer_path or "openlm-research/open_llama_3b_v2"

    if dataset_name == "squad":
        if not os.path.exists("./data/squad/dev-v2.0.json"):
            print(f"  Skipping {dataset_name}: data not found")
            return None
        dataset = SQuADDataset(
            filename="./data/squad/dev-v2.0.json",
            num=num_samples,
            num_unrelated_contexts=num_unrelated_contexts,
            tokenizer="llama", tokenizer_path=tok_path)
    elif dataset_name == "naturalqa":
        if not os.path.exists(
                "./data/nq/v1.0-simplified_nq-dev-all.jsonl"):
            print(f"  Skipping {dataset_name}: data not found")
            return None
        dataset = NQDataset(
            filename="./data/nq/v1.0-simplified_nq-dev-all.jsonl",
            num=num_samples,
            num_unrelated_contexts=num_unrelated_contexts,
            tokenizer="llama", tokenizer_path=tok_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_tokens = model.num_tokens
    num_blocks = model.num_blocks

    collate = partial(
        collate_fn_qa, tokenizer=tokenizer, max_length=512,
        num_tokens=num_tokens, add_special_tokens=False,
        end_special_token="</s>")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate)

    print(f"  Loaded {len(dataset)} samples for {dataset_name}")
    model.eval()

    step_preds = {
        f"step_{i}": [] for i in range(num_unrelated_contexts + 1)}
    all_targets = []

    backup_memory = model.memory.data.detach().cpu().clone()

    with torch.no_grad():
        for batch in tqdm(dataloader,
                          desc=f"{strategy}/{dataset_name}"):
            batch = [x.cuda() for x in batch]

            ctx_ids, ctx_mask = batch[0], batch[1]
            q_ids, q_mask = batch[2], batch[3]
            a_ids = batch[4]

            unrel = batch[6:]
            unrel_ids = [unrel[i * 2] for i in range(len(unrel) // 2)]
            unrel_masks = [unrel[i*2+1] for i in range(len(unrel) // 2)]

            # Restore memory for each sample
            model.memory.data = backup_memory.clone().to(
                model.memory.device)
            if hasattr(model, 'reset_metadata'):
                model.reset_metadata()

            all_ctx_ids = [ctx_ids] + unrel_ids
            all_ctx_masks = [ctx_mask] + unrel_masks

            query_mask = torch.cat([
                torch.ones(q_mask.shape[0],
                           num_tokens * (num_blocks - 1)).cuda(),
                q_mask
            ], dim=1)

            for idx, (ids, mask) in enumerate(
                    zip(all_ctx_ids, all_ctx_masks)):
                model.inject_memory(ids, mask, update_memory=True)

                output = model.generate(
                    inputs=q_ids, attention_mask=query_mask,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, len(q_ids[0]):][0].detach().cpu()

                step_preds[f"step_{idx}"].append(
                    tokenizer.decode(output, skip_special_tokens=False))

            all_targets.append(
                tokenizer.decode(a_ids[0], skip_special_tokens=False))

    accuracies = {}
    for step_name, preds in step_preds.items():
        if preds:
            accuracies[step_name] = calculate_exact_hit_accuracy(
                preds, all_targets)
    return accuracies


# =========================================================================
# Smoke test
# =========================================================================

def run_smoke_test(model, tokenizer, strategies):
    print("\n" + "=" * 60)
    print("SMOKE TEST: Verify strategies with synthetic data")
    print("=" * 60)

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        torch.cuda.reset_peak_memory_stats()

        # Reset memory
        model.memory.data = torch.randn_like(model.memory.data)
        model.initialized.fill_(1)
        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        contexts = [
            "The capital of France is Paris. The Eiffel Tower is "
            "located in Paris.",
            "Albert Einstein developed the theory of relativity.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Python was created by Guido van Rossum in 1991.",
            "The human body has 206 bones in the adult skeleton.",
        ]

        for ctx in contexts:
            ctx_input = tokenizer(
                ctx, return_tensors="pt",
                add_special_tokens=False).to("cuda")
            attn_mask = torch.cat([
                torch.ones(1, model.num_tokens).cuda(),
                ctx_input.attention_mask
            ], dim=1)
            model.inject_memory(
                ctx_input.input_ids, attn_mask, update_memory=True)

        query = "What is the capital of France?"
        q_input = tokenizer(query, return_tensors="pt").to("cuda")
        q_mask = torch.cat([
            torch.ones(
                1, model.num_tokens * (model.num_blocks - 1)).cuda(),
            q_input.attention_mask
        ], dim=1)

        output = model.generate(
            inputs=q_input.input_ids, attention_mask=q_mask,
            max_new_tokens=15, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(
            output[0][len(q_input.input_ids[0]):],
            skip_special_tokens=True)
        print(f"  Query: {query}")
        print(f"  Response: {response}")
        print_gpu_stats(f"After {strategy}")

        if hasattr(model, 'get_strategy_info'):
            print(f"  Info: {model.get_strategy_info()}")

    print("\n  Smoke test PASSED for all strategies")


# =========================================================================
# Memory profiling
# =========================================================================

def profile_memory(model, tokenizer, strategies):
    print("\n" + "=" * 60)
    print("MEMORY PROFILING")
    print("=" * 60)

    print_gpu_stats("Model loaded")

    results = {}
    for strategy in strategies:
        print(f"\n--- {strategy} ---")

        model.memory.data = torch.randn_like(model.memory.data)
        model.initialized.fill_(1)
        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        torch.cuda.reset_peak_memory_stats()

        ctx_text = ("This is a test context with factual information "
                    "about various topics. ") * 10
        ctx_input = tokenizer(
            ctx_text, return_tensors="pt", max_length=512,
            truncation=True, add_special_tokens=False).to("cuda")
        attn_mask = torch.cat([
            torch.ones(1, model.num_tokens).cuda(),
            ctx_input.attention_mask
        ], dim=1)

        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                model.inject_memory(
                    ctx_input.input_ids, attn_mask, update_memory=True)
                torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        results[strategy] = {
            "avg_injection_time_ms": avg_time * 1000,
            "peak_gpu_memory_gb": peak_mem,
        }
        print(f"  Avg injection time: {avg_time*1000:.1f} ms")
        print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    print(f"\n{'Strategy':<15} {'Injection (ms)':<18} "
          f"{'Peak GPU (GB)':<15}")
    print("-" * 48)
    for strat, stats in results.items():
        print(f"{strat:<15} "
              f"{stats['avg_injection_time_ms']:<18.1f} "
              f"{stats['peak_gpu_memory_gb']:<15.2f}")

    return results


# =========================================================================
# Full comparison
# =========================================================================

def run_full_comparison(model, tokenizer, strategies, datasets,
                        num_nuc, num_samples, tokenizer_path):
    print("\n" + "=" * 60)
    print("FULL COMPARISON: Knowledge Retention Across Strategies")
    print("=" * 60)

    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*40}")
        print(f"Strategy: {strategy}")
        print(f"{'='*40}")

        torch.cuda.reset_peak_memory_stats()

        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)

        strategy_results = {}
        for ds_name in datasets:
            print(f"\n  Dataset: {ds_name}, NUC: {num_nuc}")
            accs = run_retention_eval(
                model, tokenizer, ds_name,
                num_unrelated_contexts=num_nuc,
                num_samples=num_samples,
                strategy=strategy,
                tokenizer_path=tokenizer_path)
            if accs is not None:
                strategy_results[ds_name] = accs
                for step, acc in accs.items():
                    print(f"    {step}: {acc:.4f}")

        all_results[strategy] = strategy_results
        print_gpu_stats(f"Peak for {strategy}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\n  {ds_name}:")
        header = f"  {'Step':<10}"
        for s in strategies:
            header += f" {s:<12}"
        print(header)
        print("  " + "-" * (10 + 12 * len(strategies)))

        max_steps = 0
        for s in strategies:
            if ds_name in all_results.get(s, {}):
                max_steps = max(max_steps,
                                len(all_results[s][ds_name]))

        for step_idx in range(max_steps):
            step_name = f"step_{step_idx}"
            row = f"  {step_name:<10}"
            for s in strategies:
                acc = (all_results.get(s, {})
                       .get(ds_name, {}).get(step_name))
                if acc is not None:
                    row += f" {acc:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
            print(row)

    results_dir = "results/strategy_comparison"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir,
                                "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    return all_results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test MemoryLLM dropping strategies")

    # Model source: pick one
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--base-model", type=str, default=None,
        help="Base LLM to initialize MemoryLLM from scratch "
             "(e.g. openlm-research/open_llama_3b_v2)")
    group.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained MemoryLLM checkpoint "
             "(e.g. YuWangX/memoryllm-8b)")

    # Memory config (only used with --base-model)
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Number of memory blocks")
    parser.add_argument("--num-tokens", type=int, default=256,
                        help="Tokens per memory block")

    # Experiment config
    parser.add_argument(
        "--strategies", nargs="+",
        default=["random", "attention", "age", "surprise"],
        choices=["random", "attention", "age",
                 "surprise", "fisher"])
    parser.add_argument(
        "--datasets", nargs="+", default=["squad"],
        choices=["squad", "naturalqa"])
    parser.add_argument("--nuc", type=int, default=5,
                        help="Number of unrelated contexts")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of QA samples")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test with synthetic data only")
    parser.add_argument("--profile-only", action="store_true",
                        help="Only run memory profiling")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No GPU. Use a Colab GPU runtime.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load model
    if args.base_model:
        model, tokenizer = load_model_from_base(
            args.base_model,
            num_blocks=args.num_blocks,
            num_tokens=args.num_tokens,
            dtype=dtype)
        tok_path = args.base_model
    else:
        model, tokenizer = load_model_pretrained(
            args.pretrained, dtype=dtype)
        tok_path = args.pretrained

    model = model.cuda()

    print(f"\nModel: {model.L} layers, "
          f"{model.num_blocks} blocks x {model.num_tokens} tokens, "
          f"hidden_dim={model.d}")
    print(f"Memory pool: "
          f"{model.memory.numel() / 1e6:.1f}M parameters "
          f"({model.memory.numel() * 2 / 1e9:.3f} GB in fp16)")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total model: {total / 1e9:.2f}B parameters")
    print_gpu_stats("Model loaded")

    # Run tests
    if args.smoke_test:
        run_smoke_test(model, tokenizer, args.strategies)
        profile_memory(model, tokenizer, args.strategies)
    elif args.profile_only:
        profile_memory(model, tokenizer, args.strategies)
    else:
        ensure_index_files()
        profile_memory(model, tokenizer, args.strategies)
        run_full_comparison(
            model, tokenizer, args.strategies, args.datasets,
            args.nuc, args.num_samples, tok_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
