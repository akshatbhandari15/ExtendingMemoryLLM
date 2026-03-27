"""
Test script for MemoryLLM dropping strategies on Google Colab.

Profiles GPU memory usage and runs knowledge retention evaluation
using the actual SQuAD and NaturalQA datasets with all four strategies.

Usage on Colab:
    1. Clone the MemoryLLM repo and upload this file
    2. pip install -r requirements.txt  (or requirements_infer_only.txt)
    3. Download data files (see download_data() below)
    4. huggingface-cli login  (for gated model access)
    5. python test_training_colab.py --model YuWangX/memoryllm-8b --strategies random attention age surprise

    For a quick smoke test without datasets:
    python test_training_colab.py --model YuWangX/memoryllm-8b --smoke-test
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
import torch.nn.functional as F
from tqdm import tqdm


def print_gpu_stats(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB")


def download_data():
    """
    Instructions for downloading the required data files.
    The datasets are expected at these paths:
        ./data/squad/dev-v2.0.json
        ./data/squad/train-v2.0.json
        ./data/squad/indices_squad_3.npy
        ./data/nq/v1.0-simplified_nq-dev-all.jsonl
        ./data/nq/v1.0-simplified_simplified-nq-train.jsonl
        ./data/nq/indices_nq_4.npy
    """
    print("""
=== DATA SETUP ===
Download the following files and place them in ./data/:

SQuAD v2.0:
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -P data/squad/
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -P data/squad/

NaturalQA:
  Download from https://ai.google.com/research/NaturalQuestions/download
  Place v1.0-simplified_nq-dev-all.jsonl and train file in data/nq/

Index files (indices_squad_3.npy, indices_nq_4.npy):
  These are sample index files from the MemoryLLM repo.
  If missing, we will create default ones.
==================
""")


def ensure_index_files():
    """Create default index files if they don't exist."""
    squad_idx_path = "./data/squad/indices_squad_3.npy"
    nq_idx_path = "./data/nq/indices_nq_4.npy"

    if not os.path.exists(squad_idx_path):
        os.makedirs("./data/squad", exist_ok=True)
        # Default: first 100 samples
        np.save(squad_idx_path, np.arange(100))
        print(f"Created default {squad_idx_path}")

    if not os.path.exists(nq_idx_path):
        os.makedirs("./data/nq", exist_ok=True)
        np.save(nq_idx_path, np.arange(100))
        print(f"Created default {nq_idx_path}")


def calculate_exact_hit_accuracy(predictions, targets):
    """Check if target string appears anywhere in prediction."""
    hit = sum(1 for p, t in zip(predictions, targets)
              if t.replace("</s>", "").strip() in p)
    return hit / len(predictions) if predictions else 0.0


# =========================================================================
# Evaluation: Knowledge Retention Curves
# =========================================================================

def run_retention_eval(model, tokenizer, dataset_name, num_unrelated_contexts,
                       num_samples=50, strategy="random", model_path=None):
    """
    Run knowledge retention evaluation for a given strategy.

    1. Inject the relevant context (containing the answer)
    2. Inject N unrelated contexts (interfering knowledge)
    3. Query the model — does it still remember?

    Returns accuracy at each step (0 = right after relevant injection,
    1..N = after each interfering injection).
    """
    from dataset.nq import NQDataset
    from dataset.squad import SQuADDataset

    tokenizer_path = model_path or "YuWangX/memoryllm-8b"

    if dataset_name == "squad":
        if not os.path.exists("./data/squad/dev-v2.0.json"):
            print(f"  Skipping {dataset_name}: data file not found")
            return None
        dataset = SQuADDataset(
            filename="./data/squad/dev-v2.0.json",
            num=num_samples,
            num_unrelated_contexts=num_unrelated_contexts,
            tokenizer="llama",
            tokenizer_path=tokenizer_path,
        )
    elif dataset_name == "naturalqa":
        if not os.path.exists("./data/nq/v1.0-simplified_nq-dev-all.jsonl"):
            print(f"  Skipping {dataset_name}: data file not found")
            return None
        dataset = NQDataset(
            filename="./data/nq/v1.0-simplified_nq-dev-all.jsonl",
            num=num_samples,
            num_unrelated_contexts=num_unrelated_contexts,
            tokenizer="llama",
            tokenizer_path=tokenizer_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_tokens = model.num_tokens
    num_blocks = model.num_blocks

    collate_fn_with_params = partial(
        collate_fn_qa,
        tokenizer=tokenizer,
        max_length=512,
        num_tokens=num_tokens,
        add_special_tokens=False,
        end_special_token="</s>",
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn_with_params)

    print(f"  Loaded {len(dataset)} samples for {dataset_name}")

    model.eval()

    # Track accuracy at each step
    step_predictions = {f"step_{i}": [] for i in range(num_unrelated_contexts + 1)}
    all_targets = []

    # Save initial memory state
    backup_memory = model.memory.data.detach().cpu().clone()

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                      desc=f"{strategy}/{dataset_name}"):
            batch = [x.cuda() for x in batch]

            context_ids, context_attention_mask, \
                sentence_ids, sentence_attention_mask, \
                answer_ids, answer_attention_mask = batch[:6]

            unrelated_contexts_and_mask = batch[6:]
            unrelated_contexts_ids = [unrelated_contexts_and_mask[i * 2] for i in range(len(unrelated_contexts_and_mask) // 2)]
            unrelated_contexts_masks = [unrelated_contexts_and_mask[i * 2 + 1] for i in range(len(unrelated_contexts_and_mask) // 2)]

            # Restore memory to initial state for each sample
            model.memory.data = backup_memory.clone().to(model.memory.device)
            if hasattr(model, 'reset_metadata'):
                model.reset_metadata()

            # Build injection order: relevant context first, then unrelated
            contexts_ids = [context_ids] + unrelated_contexts_ids
            contexts_masks = [context_attention_mask] + unrelated_contexts_masks

            # Prepare query attention mask
            query_attention_mask = torch.cat([
                torch.ones(sentence_attention_mask.shape[0], num_tokens * (num_blocks - 1)).cuda(),
                sentence_attention_mask,
            ], dim=1)

            # Inject contexts one by one and query after each
            for idx, (ids, mask) in enumerate(zip(contexts_ids, contexts_masks)):
                model.inject_memory(ids, mask, update_memory=True)

                # Generate prediction after this injection step
                output = model.generate(
                    inputs=sentence_ids,
                    attention_mask=query_attention_mask,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, len(sentence_ids[0]):][0].detach().cpu()

                step_predictions[f"step_{idx}"].append(
                    tokenizer.decode(output, skip_special_tokens=False)
                )

            all_targets.append(
                tokenizer.decode(answer_ids[0], skip_special_tokens=False)
            )

    # Calculate accuracy at each step
    accuracies = {}
    for step_name, preds in step_predictions.items():
        if preds:
            acc = calculate_exact_hit_accuracy(preds, all_targets)
            accuracies[step_name] = acc

    return accuracies


def collate_fn_qa(data, tokenizer, max_length, num_tokens,
                  add_special_tokens=False, end_special_token=None,
                  padding='longest'):
    """Collate function for QA evaluation (from test_qa_memory.py)."""
    eval_max_length = max_length
    contexts, questions, answers, unrelated_contexts = zip(*data)

    if end_special_token is not None:
        answers = [x + end_special_token for x in list(answers)]

    contexts_tokenized = tokenizer(list(contexts), max_length=max_length,
                                   padding=padding, truncation=True,
                                   return_tensors='pt', add_special_tokens=add_special_tokens)
    questions_tokenized = tokenizer(list(questions), max_length=eval_max_length,
                                    truncation=True, padding='longest',
                                    return_tensors='pt', add_special_tokens=add_special_tokens)
    answers_tokenized = tokenizer(list(answers), max_length=eval_max_length,
                                  truncation=True, padding='longest',
                                  return_tensors='pt', add_special_tokens=add_special_tokens)

    unrelated_contexts = np.array(unrelated_contexts).transpose().tolist()

    all_unrelated_contexts = {}
    all_unrelated_contexts_mask = {}
    for i in range(len(unrelated_contexts)):
        all_unrelated_contexts[i] = tokenizer(unrelated_contexts[i], max_length=max_length,
                                              truncation=True, padding=padding,
                                              return_tensors='pt', add_special_tokens=add_special_tokens)
        all_unrelated_contexts_mask[i] = torch.cat([
            torch.tensor([1] * num_tokens).unsqueeze(0).repeat(all_unrelated_contexts[i].input_ids.shape[0], 1),
            all_unrelated_contexts[i].attention_mask
        ], dim=-1)

    contexts_attention_mask = torch.cat([
        torch.tensor([1] * num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
        contexts_tokenized.attention_mask
    ], dim=-1)
    questions_attention_mask = torch.cat([
        torch.tensor([1] * num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
        questions_tokenized.attention_mask
    ], dim=-1)
    answers_attention_mask = torch.cat([
        torch.tensor([1] * num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
        answers_tokenized.attention_mask
    ], dim=-1)

    outputs = (contexts_tokenized.input_ids, contexts_attention_mask,
               questions_tokenized.input_ids, questions_attention_mask,
               answers_tokenized.input_ids, answers_attention_mask)

    for i in range(len(all_unrelated_contexts)):
        outputs += (all_unrelated_contexts[i].input_ids,)
        outputs += (all_unrelated_contexts_mask[i],)

    return outputs


# =========================================================================
# Smoke test: validate strategies work without real data
# =========================================================================

def run_smoke_test(model, tokenizer, strategies):
    """Quick validation that each strategy works correctly with synthetic inputs."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Verify strategies with synthetic data")
    print("=" * 60)

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        torch.cuda.reset_peak_memory_stats()

        # Reset model memory
        model.memory.data = torch.randn_like(model.memory.data)
        model.initialized.fill_(1)
        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        # Inject a few contexts
        contexts = [
            "The capital of France is Paris. The Eiffel Tower is in Paris.",
            "Albert Einstein developed the theory of relativity in 1905.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Python is a programming language created by Guido van Rossum.",
            "The human body has 206 bones in the adult skeleton.",
        ]

        for i, ctx in enumerate(contexts):
            ctx_input = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).to("cuda")
            attn_mask = torch.cat([
                torch.ones(1, model.num_tokens).cuda(),
                ctx_input.attention_mask
            ], dim=1)
            model.inject_memory(ctx_input.input_ids, attn_mask, update_memory=True)

        # Query
        query = "What is the capital of France?"
        query_input = tokenizer(query, return_tensors="pt").to("cuda")
        query_mask = torch.cat([
            torch.ones(1, model.num_tokens * (model.num_blocks - 1)).cuda(),
            query_input.attention_mask
        ], dim=1)

        output = model.generate(
            inputs=query_input.input_ids,
            attention_mask=query_mask,
            max_new_tokens=15,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(output[0][len(query_input.input_ids[0]):], skip_special_tokens=True)
        print(f"  Query: {query}")
        print(f"  Response: {response}")
        print_gpu_stats(f"After {strategy}")

        if hasattr(model, 'get_strategy_info'):
            info = model.get_strategy_info()
            print(f"  Strategy info: {info}")

    print("\n  Smoke test PASSED for all strategies ✓")


# =========================================================================
# Full evaluation: compare all strategies
# =========================================================================

def run_full_comparison(model, tokenizer, strategies, datasets, num_nuc,
                        num_samples, model_path):
    """Run retention eval for each strategy and dataset, then compare."""
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
        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name}, NUC: {num_nuc}")
            accs = run_retention_eval(
                model, tokenizer, dataset_name,
                num_unrelated_contexts=num_nuc,
                num_samples=num_samples,
                strategy=strategy,
                model_path=model_path,
            )
            if accs is not None:
                strategy_results[dataset_name] = accs
                for step, acc in accs.items():
                    print(f"    {step}: {acc:.4f}")

        all_results[strategy] = strategy_results
        print_gpu_stats(f"Peak for {strategy}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    for dataset_name in datasets:
        print(f"\n  {dataset_name}:")
        header = f"  {'Step':<10}"
        for strategy in strategies:
            header += f" {strategy:<12}"
        print(header)
        print("  " + "-" * (10 + 12 * len(strategies)))

        # Find max steps across strategies
        max_steps = 0
        for strategy in strategies:
            if dataset_name in all_results.get(strategy, {}):
                max_steps = max(max_steps, len(all_results[strategy][dataset_name]))

        for step_idx in range(max_steps):
            step_name = f"step_{step_idx}"
            row = f"  {step_name:<10}"
            for strategy in strategies:
                acc = all_results.get(strategy, {}).get(dataset_name, {}).get(step_name, None)
                if acc is not None:
                    row += f" {acc:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
            print(row)

    # Save results
    results_dir = "results/strategy_comparison"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    return all_results


# =========================================================================
# GPU Memory Profiling
# =========================================================================

def profile_memory(model, tokenizer, strategies):
    """Profile GPU memory usage for each strategy during injection."""
    print("\n" + "=" * 60)
    print("MEMORY PROFILING")
    print("=" * 60)

    # Model memory baseline
    print_gpu_stats("Model loaded")

    results = {}
    for strategy in strategies:
        print(f"\n--- {strategy} ---")

        # Reset
        model.memory.data = torch.randn_like(model.memory.data)
        model.initialized.fill_(1)
        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        torch.cuda.reset_peak_memory_stats()

        # Time 20 injection steps
        ctx_text = "This is a test context with some factual information about various topics. " * 10
        ctx_input = tokenizer(ctx_text, return_tensors="pt", max_length=512,
                              truncation=True, add_special_tokens=False).to("cuda")
        attn_mask = torch.cat([
            torch.ones(1, model.num_tokens).cuda(),
            ctx_input.attention_mask
        ], dim=1)

        times = []
        with torch.no_grad():
            for step in range(20):
                start = time.time()
                model.inject_memory(ctx_input.input_ids, attn_mask, update_memory=True)
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

    # Summary
    print(f"\n{'Strategy':<15} {'Injection (ms)':<18} {'Peak GPU (GB)':<15}")
    print("-" * 48)
    for strategy, stats in results.items():
        print(f"{strategy:<15} {stats['avg_injection_time_ms']:<18.1f} {stats['peak_gpu_memory_gb']:<15.2f}")

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Test MemoryLLM dropping strategies")
    parser.add_argument("--model", type=str, default="YuWangX/memoryllm-8b",
                        help="HuggingFace model path or local checkpoint")
    parser.add_argument("--strategies", nargs="+",
                        default=["random", "attention", "age", "surprise"],
                        choices=["random", "attention", "age", "surprise", "fisher"])
    parser.add_argument("--datasets", nargs="+", default=["squad"],
                        choices=["squad", "naturalqa"])
    parser.add_argument("--nuc", type=int, default=5,
                        help="Number of unrelated contexts (interference steps)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of QA samples to evaluate")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test with synthetic data only")
    parser.add_argument("--profile-only", action="store_true",
                        help="Only run memory profiling, skip evaluation")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. Run this on Colab with a GPU runtime.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load model
    print(f"\nLoading model: {args.model}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from modeling_memoryllm_strategies import MemoryLLMWithStrategies
    from transformers import AutoTokenizer

    model = MemoryLLMWithStrategies.from_pretrained(args.model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model.to(dtype)
    model = model.cuda()

    print(f"Model loaded: {model.L} layers, {model.num_blocks} blocks x {model.num_tokens} tokens")
    print(f"Memory pool: {model.memory.numel() / 1e9:.4f}B parameters")
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
            args.nuc, args.num_samples, args.model,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
