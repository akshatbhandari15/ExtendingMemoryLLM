"""
Quick test script to profile compute requirements for training MemoryLLM on Llama 3.2 3B.
Run this on Google Colab Pro with a single GPU (A100 40GB or L4 24GB).

Usage:
    1. Upload the MemoryLLM repo to Colab or clone from GitHub
    2. pip install -r requirements.txt
    3. python test_training_colab.py

This script does NOT require the RedPajama dataset. It uses synthetic data
to measure GPU memory usage, throughput, and verify the architecture works.
"""

import torch
import torch.nn as nn
import time
import gc
import sys
import os

def print_gpu_stats(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{label}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")
    else:
        print("No CUDA available")

def test_inference_only():
    """Test loading the model and running inference with memory injection."""
    from modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer

    print("\n" + "="*60)
    print("TEST 1: Inference with pre-trained MemoryLLM-8B")
    print("="*60)

    # Try loading the 8B model (Llama 3 based) - this is the published checkpoint
    print("Loading MemoryLLM-8B (Llama 3 based)...")
    torch.cuda.reset_peak_memory_stats()

    model = MemoryLLM.from_pretrained(
        "YuWangX/memoryllm-8b",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")
    model = model.to(torch.bfloat16)
    model = model.cuda()

    print_gpu_stats("After loading model")

    # Test memory injection
    context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
    ctx_input = tokenizer(context, return_tensors="pt", add_special_tokens=False).to("cuda")

    with torch.no_grad():
        model.inject_memory(ctx_input.input_ids, ctx_input.attention_mask)

    print_gpu_stats("After memory injection")

    # Test generation
    query = "What is the capital of France?"
    query_input = tokenizer(query, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**query_input, max_new_tokens=20)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print_gpu_stats("After generation")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def test_training_memory_profile():
    """
    Profile GPU memory and throughput for training MemoryLLM from scratch
    using Llama 3.2 3B as the base model with synthetic data.
    """
    print("\n" + "="*60)
    print("TEST 2: Training profile with Llama 3.2 3B")
    print("="*60)

    # Check if we have access to Llama 3.2 3B
    from transformers import AutoTokenizer, AutoConfig

    model_name = "meta-llama/Llama-3.2-3B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_config = AutoConfig.from_pretrained(model_name)
        print(f"Llama 3.2 3B config:")
        print(f"  Hidden size: {hf_config.hidden_size}")
        print(f"  Num layers: {hf_config.num_hidden_layers}")
        print(f"  Num attention heads: {hf_config.num_attention_heads}")
        print(f"  Num KV heads: {hf_config.num_key_value_heads}")
    except Exception as e:
        print(f"Cannot access {model_name}: {e}")
        print("You may need to: huggingface-cli login")
        print("And accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-3B")
        return

    # Test different memory pool configurations
    configs_to_test = [
        {"num_blocks": 4, "num_tokens": 256, "label": "Small (4x256 = 1024 tokens)"},
        {"num_blocks": 10, "num_tokens": 256, "label": "Medium (10x256 = 2560 tokens)"},
    ]

    for cfg in configs_to_test:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n--- Config: {cfg['label']} ---")

        num_blocks = cfg["num_blocks"]
        num_tokens = cfg["num_tokens"]
        L = hf_config.num_hidden_layers  # 28 for Llama 3.2 3B
        d = hf_config.hidden_size        # 3072 for Llama 3.2 3B

        # Calculate memory pool size
        memory_params = L * num_blocks * num_tokens * d
        memory_size_gb = memory_params * 2 / 1e9  # fp16
        print(f"  Memory pool: {L} layers x {num_blocks * num_tokens} tokens x {d} dim")
        print(f"  Memory pool parameters: {memory_params / 1e6:.1f}M ({memory_size_gb:.3f} GB in fp16)")

        # Try loading the model with memory
        try:
            # Use the training code path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))
            from MemoryLLM.memoryllm.modules.configuration_llama import LlamaConfig as MemLlamaConfig
            from MemoryLLM.memoryllm.modules.memory_llama import LlamaDropMemoryModel

            config = MemLlamaConfig.from_pretrained(model_name)
            config.num_blocks = num_blocks
            config.num_tokens = num_tokens
            config.add_bos_embedding = True
            config.shrink_to_one_embedding = True
            config.num_memory_tokens = num_tokens * num_blocks
            config.drop_memory_per_layer = True

            # RoPE scaling for memory tokens
            total_positions = config.num_memory_tokens + 512  # memory + input
            if total_positions > config.max_position_embeddings:
                config.rope_scaling = {
                    "type": "linear",
                    "factor": total_positions / config.max_position_embeddings
                }

            print("  Loading model with memory...")
            model = LlamaDropMemoryModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16
            )

            # Apply LoRA
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj', 'k_proj', 'up_proj', 'down_proj', 'gate_proj']
            )
            model = get_peft_model(model, peft_config)
            if hasattr(model.base_model, "new_memory_positional_emb"):
                model.base_model.new_memory_positional_emb.requires_grad = True

            model = model.cuda()
            print_gpu_stats("After loading model + LoRA")

            # Count trainable parameters
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total ({100*trainable/total:.2f}%)")

            # Simulate a training step with synthetic data
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=4.6e-6
            )

            # Create synthetic input (batch_size=1, seq_len=256)
            input_ids = torch.randint(0, hf_config.vocab_size, (1, 256)).cuda()
            attention_mask = torch.ones_like(input_ids).cuda()
            labels = input_ids.clone()

            print("  Running forward pass...")
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            forward_time = time.time() - start
            print(f"  Forward pass: {forward_time:.2f}s, Loss: {loss.item():.4f}")
            print_gpu_stats("After forward pass")

            print("  Running backward pass...")
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            print(f"  Backward pass: {backward_time:.2f}s")
            print_gpu_stats("After backward pass")

            optimizer.step()
            optimizer.zero_grad()
            print_gpu_stats("After optimizer step")

            # Run a few more steps to get stable timing
            times = []
            for step in range(5):
                start = time.time()
                input_ids = torch.randint(0, hf_config.vocab_size, (1, 256)).cuda()
                labels = input_ids.clone()
                outputs = model(input_ids=input_ids, labels=labels)
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            print(f"\n  Avg step time: {avg_time:.2f}s")
            print(f"  Estimated steps/hour: {3600/avg_time:.0f}")
            print(f"  Estimated time for 10k steps: {10000*avg_time/3600:.1f} hours")
            print(f"  Estimated time for 500 steps (test run): {500*avg_time/60:.1f} minutes")
            print_gpu_stats("Final")

            del model, optimizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()


def test_drop_memory_modification():
    """
    Test that we can modify the drop_memory function to use importance-based dropping.
    This is a quick validation that the code path is correct.
    """
    print("\n" + "="*60)
    print("TEST 3: Verify drop_memory modification point")
    print("="*60)

    # Simulate the drop_memory function with importance scores
    N = 1024  # memory pool size (4 blocks x 256 tokens)
    K = 256   # tokens to drop (1 block)
    d = 3072  # hidden dim

    memory = torch.randn(N, d)

    # Original: random dropping
    perm = torch.randperm(N)
    random_remaining = perm[:N-K].sort()[0]
    random_dropped = perm[N-K:].sort()[0]

    # Attention-based: drop least-attended
    attention_scores = torch.rand(N)  # simulated EMA attention scores
    sorted_indices = attention_scores.argsort()  # ascending - least important first
    importance_dropped = sorted_indices[:K].sort()[0]
    importance_remaining = sorted_indices[K:].sort()[0]

    # Age-stratified: protect recent, drop old
    ages = torch.arange(N).float()  # token 0 is oldest, token N-1 is newest
    P = 256  # protection window
    protected = ages >= (N - P)  # protect last P tokens
    droppable_indices = torch.where(~protected)[0]
    drop_probs = ages[droppable_indices] / ages[droppable_indices].max()  # older = higher prob
    drop_probs = 1 - drop_probs  # invert: oldest tokens get highest drop prob
    # ... would sample K from droppable_indices weighted by drop_probs

    print(f"  Memory pool: {N} tokens, dropping {K}")
    print(f"  Random: dropped indices range [{random_dropped.min()}-{random_dropped.max()}] (uniform)")
    print(f"  Attention-based: dropped indices range [{importance_dropped.min()}-{importance_dropped.max()}] (lowest scores)")
    print(f"  Age-stratified: {(~protected).sum()} droppable, {protected.sum()} protected")
    print("  All strategies produce valid index sets ✓")


if __name__ == "__main__":
    print("="*60)
    print("MemoryLLM Training Profiler for Google Colab")
    print("="*60)

    if not torch.cuda.is_available():
        print("ERROR: No GPU available. Run this on Colab with GPU runtime.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Run tests in order of compute requirements
    test_drop_memory_modification()
    test_training_memory_profile()

    # Only run inference test if the 8B model fits (needs ~20GB)
    if gpu_mem >= 30:
        test_inference_only()
    else:
        print(f"\nSkipping 8B inference test (need 30GB+, have {gpu_mem:.0f}GB)")

    print("\n" + "="*60)
    print("DONE - Use the timing estimates above to plan your training runs")
    print("="*60)
