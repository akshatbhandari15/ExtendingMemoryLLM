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

    # Initialize memory from real text (not random noise)
    warmup_texts = [
        "The Earth orbits the Sun at an average distance of 93 million miles.",
        "Water freezes at 32 degrees Fahrenheit or 0 degrees Celsius.",
        "The human heart beats approximately 100,000 times per day.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]
    init_memory_from_contexts(model, tokenizer, warmup_texts)
    print(f"  Memory initialized from {len(warmup_texts)} warmup contexts")

    step_preds = {
        f"step_{i}": [] for i in range(num_unrelated_contexts + 1)}
    all_targets = []

    backup_memory = model.memory.data.detach().cpu().clone()

    with torch.no_grad():
        for batch in tqdm(dataloader,
                          desc=f"{strategy}/{dataset_name}"):
            batch = [x.cuda() for x in batch]

            ctx_ids, ctx_mask = batch[0], batch[1]
            q_ids, _ = batch[2], batch[3]
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

            for idx, (ids, mask) in enumerate(
                    zip(all_ctx_ids, all_ctx_masks)):
                model.inject_memory(ids, mask, update_memory=True)

                output = manual_generate(
                    model, tokenizer, q_ids, max_new_tokens=10
                )
                pred_tokens = output[0, q_ids.shape[1]:].detach().cpu()
                step_preds[f"step_{idx}"].append(
                    tokenizer.decode(pred_tokens, skip_special_tokens=False))

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

def manual_generate(model, tokenizer, input_ids, max_new_tokens=30,
                    debug=False):
    """Generate tokens one at a time without relying on model.generate().
    This bypasses potential issues with GenerationMixin + MemoryLLM interaction."""
    generated = input_ids.clone()
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=generated, use_cache=False)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            last_logits = logits[:, -1, :]

            if debug and step == 0:
                # Print logit diagnostics
                print(f"    [debug] logits shape: {logits.shape}")
                print(f"    [debug] last_logits stats: "
                      f"min={last_logits.min().item():.3f} "
                      f"max={last_logits.max().item():.3f} "
                      f"mean={last_logits.mean().item():.3f} "
                      f"std={last_logits.std().item():.3f}")
                top5 = torch.topk(last_logits[0], 5)
                for i in range(5):
                    tid = top5.indices[i].item()
                    val = top5.values[i].item()
                    tok = tokenizer.decode([tid])
                    print(f"    [debug] top-{i+1}: id={tid} val={val:.3f} -> '{tok}'")

            next_token = last_logits.argmax(dim=-1, keepdim=True)
            tok_id = next_token.item()
            # Stop on EOS or padding
            if tok_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                if step == 0 and debug:
                    print(f"    WARNING: model produced EOS/PAD on first token!")
                break
            # Stop on <unk> (token 0) — model is broken
            if tok_id == 0:
                if step == 0 and debug:
                    print(f"    WARNING: model produced <unk> (id=0) on first token!")
                break
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def init_memory_from_contexts(model, tokenizer, warmup_texts):
    """Initialize memory pool from real text instead of random noise.
    This fills the pool with actual hidden states so the model can function."""
    model.initialized.fill_(0)  # Reset to uninitialized
    # First injection with initialized=0 fills the entire memory pool
    first = warmup_texts[0]
    ctx_input = tokenizer(first, return_tensors="pt",
                          add_special_tokens=False).to("cuda")
    # When uninitialized, inject_memory doesn't prepend memory tokens
    # so attention_mask matches just the input
    model.inject_memory(ctx_input.input_ids,
                        ctx_input.attention_mask,
                        update_memory=True)
    # Now initialized=1, inject remaining warmup contexts normally
    for text in warmup_texts[1:]:
        ctx_input = tokenizer(text, return_tensors="pt",
                              add_special_tokens=False).to("cuda")
        inj_mask = torch.cat([
            torch.ones(1, model.num_tokens, device="cuda"),
            ctx_input.attention_mask
        ], dim=1)
        model.inject_memory(ctx_input.input_ids, inj_mask,
                            update_memory=True)


def run_smoke_test(model, tokenizer, strategies):
    print("\n" + "=" * 60)
    print("SMOKE TEST: Verify strategies with real QA examples")
    print("=" * 60)

    # Warmup contexts to fill memory with real hidden states
    warmup_texts = [
        "The Earth orbits the Sun at an average distance of 93 million miles.",
        "Water freezes at 32 degrees Fahrenheit or 0 degrees Celsius.",
        "The human heart beats approximately 100,000 times per day.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]

    # Real SQuAD-style QA examples
    qa_examples = [
        {
            "context": (
                "Nikola Tesla was a Serbian-American inventor and electrical "
                "engineer. He is best known for his contributions to the design "
                "of the modern alternating current (AC) electricity supply system. "
                "Tesla was born on 10 July 1856 in Smiljan, Austrian Empire "
                "(modern-day Croatia)."
            ),
            "distractors": [
                "The Amazon rainforest produces about 20 percent of the world's oxygen.",
                "The Great Wall of China stretches over 13,000 miles across northern China.",
            ],
            "query": "Where was Nikola Tesla born?",
            "expected": "Smiljan",
        },
        {
            "context": (
                "Marie Curie was a Polish and naturalized-French physicist and "
                "chemist. She was the first woman to win a Nobel Prize, the first "
                "person to win the Nobel Prize twice, and the only person to win "
                "the Nobel Prize in two scientific fields: physics in 1903 and "
                "chemistry in 1911."
            ),
            "distractors": [
                "Mount Everest is 8,849 meters tall and located in the Himalayas.",
                "The speed of light in vacuum is approximately 299,792 km per second.",
            ],
            "query": "In what year did Marie Curie win the Nobel Prize in chemistry?",
            "expected": "1911",
        },
    ]

    # -- Diagnose the generation problem --
    print("\n[DIAG] ============================================")
    print("[DIAG] Diagnosing generation issue")
    print("[DIAG] ============================================")

    model.initialized.fill_(0)  # Disable memory
    test_prompt = "The capital of France is"
    test_input = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    print(f"[DIAG] Input ids: {test_input.input_ids}")
    print(f"[DIAG] Decoded input: '{tokenizer.decode(test_input.input_ids[0])}'")
    print(f"[DIAG] Model initialized: {model.initialized.item()}")
    print(f"[DIAG] Vocab size: {model.config.vocab_size}")
    print(f"[DIAG] EOS={tokenizer.eos_token_id} PAD={tokenizer.pad_token_id} UNK={tokenizer.unk_token_id}")

    with torch.no_grad():
        # Check embeddings
        embeds = model.model.embed_tokens(test_input.input_ids)
        print(f"[DIAG] Embed shape: {embeds.shape}")
        print(f"[DIAG] Embed stats: min={embeds.min().item():.4f} "
              f"max={embeds.max().item():.4f} mean={embeds.mean().item():.6f}")

        # Check lm_head
        print(f"[DIAG] lm_head weight shape: {model.lm_head.weight.shape}")
        print(f"[DIAG] lm_head weight stats: min={model.lm_head.weight.min().item():.4f} "
              f"max={model.lm_head.weight.max().item():.4f}")

        # Full forward pass
        print("[DIAG] Running full forward pass...")
        outputs = model(input_ids=test_input.input_ids, use_cache=False)
        print(f"[DIAG] Output type: {type(outputs)}")

        if isinstance(outputs, tuple):
            print(f"[DIAG] Output is tuple, len={len(outputs)}")
            logits = outputs[0]
        else:
            print(f"[DIAG] Output has logits attr: {hasattr(outputs, 'logits')}")
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        print(f"[DIAG] Logits shape: {logits.shape}")
        print(f"[DIAG] Logits dtype: {logits.dtype}")
        print(f"[DIAG] Logits[-1] stats: min={logits[0,-1].min().item():.4f} "
              f"max={logits[0,-1].max().item():.4f} "
              f"mean={logits[0,-1].mean().item():.6f} "
              f"std={logits[0,-1].std().item():.4f}")
        print(f"[DIAG] Any NaN in logits: {torch.isnan(logits).any().item()}")
        print(f"[DIAG] Any Inf in logits: {torch.isinf(logits).any().item()}")

        # Top-5 tokens
        top5 = torch.topk(logits[0, -1], 5)
        print("[DIAG] Top-5 predicted tokens:")
        for i in range(5):
            tid = top5.indices[i].item()
            val = top5.values[i].item()
            tok = tokenizer.decode([tid])
            print(f"[DIAG]   #{i+1}: id={tid} logit={val:.4f} -> '{tok}'")

        # Check if logits are all the same (degenerate)
        unique_vals = torch.unique(logits[0, -1]).shape[0]
        print(f"[DIAG] Unique logit values in last position: {unique_vals}")

    # Trace through decoder layers MANUALLY (bypass MemoryLLM forward)
    print("\n[DIAG] --- Layer-by-layer trace (raw decoder, no MemoryLLM) ---")
    with torch.no_grad():
        embeds = model.model.embed_tokens(test_input.input_ids)
        print(f"[DIAG] Embeds: nan={torch.isnan(embeds).any().item()} "
              f"range=[{embeds.min().item():.4f}, {embeds.max().item():.4f}]")

        hidden = embeds
        pos_ids = torch.arange(0, hidden.shape[1], device="cuda").unsqueeze(0)

        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden, position_ids=pos_ids)[0]
            has_nan = torch.isnan(hidden).any().item()
            if has_nan or i < 3 or i >= len(model.model.layers) - 2:
                hmin = hidden.min().item() if not has_nan else float('nan')
                hmax = hidden.max().item() if not has_nan else float('nan')
                print(f"[DIAG] Layer {i:2d}: nan={has_nan} range=[{hmin:.4f}, {hmax:.4f}]")
            if has_nan:
                print(f"[DIAG] *** NaN FIRST APPEARS at layer {i} ***")
                break

        if not torch.isnan(hidden).any():
            hidden = model.model.norm(hidden)
            logits_raw = model.lm_head(hidden).float()
            print(f"[DIAG] Final logits: nan={torch.isnan(logits_raw).any().item()}")
            top5 = torch.topk(logits_raw[0, -1], 5)
            for j in range(5):
                tid = top5.indices[j].item()
                val = top5.values[j].item()
                print(f"[DIAG]   #{j+1}: id={tid} logit={val:.4f} -> '{tokenizer.decode([tid])}'")

    print("[DIAG] ============================================\n")

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        torch.cuda.reset_peak_memory_stats()

        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        # Initialize memory from real text (not random noise)
        init_memory_from_contexts(model, tokenizer, warmup_texts)
        print(f"  Memory initialized from {len(warmup_texts)} warmup contexts")

        for ex in qa_examples:
            # Inject the relevant context
            ctx_input = tokenizer(
                ex["context"], return_tensors="pt",
                add_special_tokens=False).to("cuda")
            inj_mask = torch.cat([
                torch.ones(1, model.num_tokens, device="cuda"),
                ctx_input.attention_mask
            ], dim=1)
            model.inject_memory(
                ctx_input.input_ids, inj_mask, update_memory=True)

            # Inject distractors
            for dist in ex["distractors"]:
                d_input = tokenizer(
                    dist, return_tensors="pt",
                    add_special_tokens=False).to("cuda")
                d_mask = torch.cat([
                    torch.ones(1, model.num_tokens, device="cuda"),
                    d_input.attention_mask
                ], dim=1)
                model.inject_memory(
                    d_input.input_ids, d_mask, update_memory=True)

            # Query
            prompt = f"Q: {ex['query']}\nA:"
            q_input = tokenizer(prompt, return_tensors="pt").to("cuda")

            output = manual_generate(model, tokenizer, q_input.input_ids,
                                     max_new_tokens=30)
            response = tokenizer.decode(
                output[0][q_input.input_ids.shape[1]:],
                skip_special_tokens=True).strip()

            contains_answer = ex["expected"].lower() in response.lower()
            print(f"  Q: {ex['query']}")
            print(f"  A: {response[:150]}")
            print(f"  Expected: {ex['expected']} | Found: {contains_answer}")

        print_gpu_stats(f"After {strategy}")
        if hasattr(model, 'get_strategy_info'):
            print(f"  Info: {model.get_strategy_info()}")

    print("\n  Smoke test PASSED for all strategies")


# =========================================================================
# Mini retention curve (toy, 5-step, built-in)
# =========================================================================

def run_mini_retention(model, tokenizer, strategies):
    """Run a tiny retention experiment inline — no external dataset needed.
    Inject a target fact, then inject N distractors, query after each step."""
    print("\n" + "=" * 60)
    print("MINI RETENTION CURVE (5 distractor steps)")
    print("=" * 60)

    target_context = (
        "Professor James Henderson discovered a new species of deep-sea "
        "jellyfish called Aurelia profunda in the Mariana Trench in 2019."
    )
    query_prompt = "Q: What species did Professor James Henderson discover?\nA:"
    expected = "aurelia profunda"

    distractors = [
        "The Nile River is approximately 6,650 kilometers long and flows through northeastern Africa.",
        "Ludwig van Beethoven composed his Ninth Symphony in 1824 while almost completely deaf.",
        "The mitochondria is often called the powerhouse of the cell because it generates most of the cell's ATP.",
        "Jupiter is the largest planet in our solar system with a mass of 1.898 times 10 to the 27 kilograms.",
        "The Treaty of Westphalia in 1648 ended the Thirty Years War in the Holy Roman Empire.",
    ]

    warmup_texts = [
        "The Earth orbits the Sun at an average distance of 93 million miles.",
        "Water freezes at 32 degrees Fahrenheit or 0 degrees Celsius.",
        "The human heart beats approximately 100,000 times per day.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]

    results = {}

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        if hasattr(model, 'set_drop_strategy'):
            model.set_drop_strategy(strategy)
            model.reset_metadata()

        # Fresh memory from warmup
        init_memory_from_contexts(model, tokenizer, warmup_texts)

        # Inject target fact
        ctx_input = tokenizer(target_context, return_tensors="pt",
                              add_special_tokens=False).to("cuda")
        inj_mask = torch.cat([
            torch.ones(1, model.num_tokens, device="cuda"),
            ctx_input.attention_mask
        ], dim=1)
        model.inject_memory(ctx_input.input_ids, inj_mask, update_memory=True)

        step_responses = []

        # Query immediately (step 0 = right after target injection)
        q_input = tokenizer(query_prompt, return_tensors="pt").to("cuda")
        out = manual_generate(model, tokenizer, q_input.input_ids, max_new_tokens=20)
        resp = tokenizer.decode(out[0][q_input.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
        hit = expected.lower() in resp.lower()
        step_responses.append((0, resp, hit))
        print(f"  Step 0 (just injected): '{resp[:80]}' | hit={hit}")

        # Inject distractors one by one, query after each
        for i, dist in enumerate(distractors):
            d_input = tokenizer(dist, return_tensors="pt",
                                add_special_tokens=False).to("cuda")
            d_mask = torch.cat([
                torch.ones(1, model.num_tokens, device="cuda"),
                d_input.attention_mask
            ], dim=1)
            model.inject_memory(d_input.input_ids, d_mask, update_memory=True)

            out = manual_generate(model, tokenizer, q_input.input_ids,
                                  max_new_tokens=20)
            resp = tokenizer.decode(out[0][q_input.input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
            hit = expected.lower() in resp.lower()
            step_responses.append((i + 1, resp, hit))
            print(f"  Step {i+1} (after {i+1} distractors): '{resp[:80]}' | hit={hit}")

        results[strategy] = step_responses

    # Summary table
    print("\n" + "-" * 60)
    print("RETENTION SUMMARY")
    print("-" * 60)
    header = f"{'Strategy':<15}" + "".join(f"{'Step '+str(i):<10}" for i in range(len(distractors) + 1))
    print(header)
    for strategy, steps in results.items():
        row = f"{strategy:<15}" + "".join(
            f"{'HIT' if s[2] else 'MISS':<10}" for s in steps)
        print(row)

    return results


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
        run_mini_retention(model, tokenizer, args.strategies)
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
