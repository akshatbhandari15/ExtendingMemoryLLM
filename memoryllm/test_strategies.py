# save as: test_strategies.py
import sys, os, torch, json, copy
sys.path.insert(0, ".")

from tqdm import tqdm
from modeling_memoryllm_strategies import MemoryLLMWithStrategies
from transformers import AutoTokenizer

# Load model once
model = MemoryLLMWithStrategies.from_pretrained(
    "YuWangX/memoryllm-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")

# Save clean memory state
backup_memory = model.memory.data.clone()
backup_initialized = model.initialized.clone()

# Import dataset
from dataset.squad import SQuADDataset
NUC = 5
dataset = SQuADDataset(filename="./data/squad/dev-v2.0.json", num=20, num_unrelated_contexts=NUC)
print(f"Loaded {len(dataset)} samples")


results = {}

for strategy in ["random", "age", "surprise"]:  # add "attention" later
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    model.set_drop_strategy(strategy)
    
    step_hits = {f"step_{i}": 0 for i in range(NUC)}
    step_total = {f"step_{i}": 0 for i in range(NUC)}
    
    for sample_idx in tqdm(range(len(dataset))):
        context, question, answer, unrelated_contexts = dataset[sample_idx]
        
        # Reset memory to clean state
        model.memory.data = backup_memory.clone().to(model.memory.device)
        model.initialized.data = backup_initialized.clone()
        model.reset_metadata()
        
        # Tokenize
        ctx_tok = tokenizer(context, max_length=256, truncation=True,
                           return_tensors="pt", add_special_tokens=False)
        q_tok = tokenizer(question, return_tensors="pt", add_special_tokens=False)
        ans_text = answer.strip().lower()
        
        # Build unrelated context list
        unrel_list = unrelated_contexts if isinstance(unrelated_contexts, list) else [unrelated_contexts]
        
        # Inject target context first
        model.inject_memory(
            ctx_tok.input_ids.cuda(),
            ctx_tok.attention_mask.cuda(),
            update_memory=True
        )
        
        # Inject unrelated contexts, query after each
        for step_idx in range(min(NUC, len(unrel_list))):
            unrel_tok = tokenizer(unrel_list[step_idx], max_length=256, truncation=True,
                                 return_tensors="pt", add_special_tokens=False)
            model.inject_memory(
                unrel_tok.input_ids.cuda(),
                unrel_tok.attention_mask.cuda(),
                update_memory=True
            )
            
            # Build attention mask with memory prefix
            q_ids = q_tok.input_ids.cuda()
            q_mask = torch.cat([
                torch.ones(1, model.num_tokens * (model.num_blocks - 1)).cuda(),
                q_tok.attention_mask.cuda()
            ], dim=1)
            
            output = model.generate(
                inputs=q_ids,
                attention_mask=q_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id
            )
            
            gen_text = tokenizer.decode(output[0][len(q_ids[0]):], skip_special_tokens=True).strip().lower()
            hit = ans_text in gen_text
            
            step_hits[f"step_{step_idx}"] += int(hit)
            step_total[f"step_{step_idx}"] += 1
    
    # Print results
    print(f"\nResults for {strategy}:")
    for i in range(NUC):
        key = f"step_{i}"
        acc = step_hits[key] / max(step_total[key], 1)
        print(f"  Step {i}: {acc:.4f} ({step_hits[key]}/{step_total[key]})")
    
    results[strategy] = {k: step_hits[k] / max(step_total[k], 1) for k in step_hits}

# Summary
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Step':<8}", end="")
for s in results:
    print(f"{s:<12}", end="")
print()
for i in range(NUC):
    print(f"{i:<8}", end="")
    for s in results:
        print(f"{results[s][f'step_{i}']:<12.3f}", end="")
    print()