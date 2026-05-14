"""
Self-distillation data generation for FastMTP training.
Uses SGLang Engine for high-throughput batch inference (matches paper Section 2.4).

Pipeline:
  1. Prepare: download datasets, extract prompts, sample by domain ratio
  2. Generate: launch SGLang Engine, batch-generate responses via API
  3. Clean:  MinHash dedup + heuristics (separate script: clean_data.py)

Paper generation parameters:
  temperature=0.6, top_k=20, top_p=0.95, max_new_tokens=4096

Usage:
  python scripts/generate_distilled_data.py --step prepare
  python scripts/generate_distilled_data.py --step generate --gpus 4,5,6,7
  python scripts/generate_distilled_data.py --step generate --gpus 4,5,6,7 --max-samples 100
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Model for self-distillation (local path)
BASE_MODEL = "/ssd/yangxw/FastMTP/model/FastMTP"

# Generation parameters (paper Section 2.4)
GEN_TEMPERATURE = 0.6
GEN_TOP_K = 20
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 4096

# SGLang settings
SGLANG_PORT = 30000
SGLANG_MEM_FRACTION = 0.85

# Note: this script's generate step must run in the 'sglang-gen' conda env
# (Python 3.12 + SGLang 0.5.11), NOT the FastMTP training env.
# The prepare step runs in the FastMTP env.
# SGLang loads the model via auto_map (trust_remote_code=True).

# Output
OUTPUT_DIR = Path("/ssd/yangxw/FastMTP/data")

# Domain ratios (docs/training_data_plan.md)
DOMAIN_RATIOS = {
    "code":    0.25,
    "general": 0.25,
    "rag":     0.20,
    "chinese": 0.20,
    "math":    0.10,
}
TOTAL_TARGET = 389400

# ---- Dataset sources → domain mapping ----
TULU_SOURCE_DOMAIN = {
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated":       "code",
    "ai2-adapt-dev/personahub_code_v2_34999":                   "code",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated":         "math",
    "ai2-adapt-dev/personahub_math_v5_regen_149960":            "math",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k":           "math",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "math",
    "allenai/tulu-3-sft-personas-math-grade":                   "math",
    "ai2-adapt-dev/flan_v2_converted":                          "general",
    "ai2-adapt-dev/no_robots_converted":                        "general",
    "ai2-adapt-dev/oasst1_converted":                           "general",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980":     "general",
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k":                    "general",
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k":                      "general",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k":                     "general",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10":                "general",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "general",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k": "general",
    "ai2-adapt-dev/coconot_converted":                          "rag",
    "ai2-adapt-dev/tulu_v3.9_aya_100k":                         "chinese",
}

EXTRA_DATASETS = [
    ("BelleGroup/train_3.5M_CN", None, "belle_conversations", "chinese"),
]

LONGBENCH_V2_KWARGS = {"path": "THUDM/LongBench-v2", "split": "train"}

RAG_FILLER_DATASETS = [
    ("cnn_dailymail", "3.0.0", "train", "article"),
]


# ============================================================================
# Prompt extraction helpers
# ============================================================================

def extract_prompt_from_messages(messages):
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    elif isinstance(messages, str):
        return messages
    return ""


def extract_prompt(obj, key):
    if key == "messages":
        return extract_prompt_from_messages(obj.get("messages", []))
    elif key == "belle_conversations":
        convs = obj.get("conversations", [])
        for conv in convs:
            if isinstance(conv, dict) and conv.get("from") == "human":
                return conv.get("value", "")
        return ""
    elif key == "longbench_v2":
        ctx = obj.get("context", "")
        q = obj.get("question", "")
        choices = []
        for letter in ["A", "B", "C", "D"]:
            c = obj.get(f"choice_{letter}", "")
            if c:
                choices.append(f"  {letter}. {c}")
        parts = [ctx] if ctx else []
        if q:
            parts.append(q)
        if choices:
            parts.append("\n".join(choices))
        return "\n\n".join(parts)
    elif key == "instruction":
        inp = obj.get("instruction", "")
        if obj.get("input"):
            inp += "\n" + obj.get("input", "")
        return inp
    elif key == "input":
        return obj.get("input", "")
    else:
        return obj.get(key, "")


# ============================================================================
# Step 1: Prepare Prompts (unchanged, no GPU needed)
# ============================================================================

def prepare_prompts(output_dir, seed=42):
    print("=" * 60)
    print("Step 1: Prepare Prompts")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_prompts = {d: [] for d in DOMAIN_RATIOS}
    domain_targets = {d: int(TOTAL_TARGET * r) for d, r in DOMAIN_RATIOS.items()}

    # ---- Tulu 3 ----
    print("\nLoading Tulu 3 (streaming, filtering by source)...")
    try:
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        source_counts = defaultdict(int)
        total_scanned, total_kept = 0, 0
        unknown_sources = set()

        pbar = tqdm(desc="  Tulu-3", unit=" samples")
        for sample in ds:
            total_scanned += 1
            source = sample.get("source", "")
            domain = TULU_SOURCE_DOMAIN.get(source)
            if domain is None:
                unknown_sources.add(source)
                pbar.update(1)
                if len(unknown_sources) <= 30 and len(unknown_sources) % 5 == 0:
                    pbar.write(f"  unknown source: '{source}'")
                continue
            if len(domain_prompts[domain]) >= domain_targets[domain]:
                pbar.update(1)
                if all(len(domain_prompts[d]) >= domain_targets[d] for d in DOMAIN_RATIOS):
                    break
                continue

            prompt = extract_prompt(sample, "messages")
            if prompt and len(prompt.strip()) > 10:
                domain_prompts[domain].append(prompt.strip())
                source_counts[source] += 1
                total_kept += 1
            pbar.update(1)
            if total_scanned % 1000 == 0:
                status = "|".join(f"{d[:4]}:{len(domain_prompts[d])}/{domain_targets[d]}"
                                  for d in ["code","general","rag","math","chinese"])
                pbar.set_postfix_str(status)
        pbar.close()
        print(f"  Scanned: {total_scanned}, kept: {total_kept}")
        if unknown_sources:
            print(f"  Unknown sources ({len(unknown_sources)}): {sorted(unknown_sources)[:20]}")
    except Exception as e:
        print(f"  WARNING Tulu-3: {e}")
        traceback.print_exc()

    # ---- BELLE ----
    print("\nLoading extra datasets...")
    for ds_id, _, prompt_key, domain in EXTRA_DATASETS:
        if len(domain_prompts[domain]) >= domain_targets[domain]:
            continue
        try:
            ds = load_dataset(ds_id, split="train", streaming=True)
            kept = 0
            pbar = tqdm(desc=f"  {ds_id.split('/')[-1]:30s}", unit=" samples")
            for sample in ds:
                pbar.update(1)
                prompt = extract_prompt(sample, prompt_key)
                if prompt and len(prompt.strip()) > 10:
                    domain_prompts[domain].append(prompt.strip())
                    kept += 1
                if len(domain_prompts[domain]) >= domain_targets[domain]:
                    break
                if kept % 5000 == 0:
                    pbar.set_postfix_str(f"kept:{kept}")
            pbar.close()
            print(f"  {ds_id}: kept {kept} ({domain})")
        except Exception as e:
            print(f"  WARNING {ds_id}: {e}")

    # ---- LongBench v2 ----
    print("\nLoading LongBench v2...")
    if len(domain_prompts["rag"]) < domain_targets["rag"]:
        try:
            ds = load_dataset(**LONGBENCH_V2_KWARGS, streaming=True)
            kept = 0
            pbar = tqdm(desc="  longbench-v2", unit=" samples")
            for sample in ds:
                pbar.update(1)
                prompt = extract_prompt(sample, "longbench_v2")
                if prompt and len(prompt.strip()) > 10:
                    domain_prompts["rag"].append(prompt.strip())
                    kept += 1
                if len(domain_prompts["rag"]) >= domain_targets["rag"]:
                    break
            pbar.close()
            print(f"  LongBench v2: kept {kept} (rag total: {len(domain_prompts['rag'])}/{domain_targets['rag']})")
        except Exception as e:
            print(f"  WARNING LongBench v2: {e}")

    # ---- RAG filler ----
    if len(domain_prompts["rag"]) < domain_targets["rag"]:
        print("\nLoading RAG filler datasets...")
        for ds_name, ds_version, ds_split, prompt_field in RAG_FILLER_DATASETS:
            if len(domain_prompts["rag"]) >= domain_targets["rag"]:
                break
            try:
                ds = load_dataset(ds_name, ds_version, split=ds_split, streaming=True)
                kept = 0
                pbar = tqdm(desc=f"  {ds_name:30s}", unit=" samples")
                for sample in ds:
                    pbar.update(1)
                    text = sample.get(prompt_field, "")
                    if text and len(text.strip()) > 100:
                        prompt = "Summarize the following text in detail:\n\n" + text.strip()
                        domain_prompts["rag"].append(prompt)
                        kept += 1
                    if len(domain_prompts["rag"]) >= domain_targets["rag"]:
                        pbar.set_postfix_str(f"rag: {len(domain_prompts['rag'])}/{domain_targets['rag']}")
                        break
                pbar.close()
                print(f"  {ds_name}: kept {kept} (rag: {len(domain_prompts['rag'])}/{domain_targets['rag']})")
            except Exception as e:
                print(f"  WARNING {ds_name}: {e}")

    # ---- Report & save ----
    print("\n--- Domain Prompt Counts ---")
    for domain in DOMAIN_RATIOS:
        actual = len(domain_prompts[domain])
        target = domain_targets[domain]
        pct = actual / target * 100 if target > 0 else 0
        print(f"  {domain:10s}: {actual:>6}/{target:>6} ({pct:.1f}%)")

    for domain in DOMAIN_RATIOS:
        random.shuffle(domain_prompts[domain])
        domain_prompts[domain] = domain_prompts[domain][:domain_targets[domain]]

    prompts_file = output_dir / "prompts_by_domain.json"
    with open(prompts_file, "w") as f:
        json.dump(domain_prompts, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {prompts_file}")

    flat_file = output_dir / "prompts_flat.jsonl"
    total = 0
    with open(flat_file, "w") as f:
        for domain, prompts in domain_prompts.items():
            for prompt in prompts:
                f.write(json.dumps({"domain": domain, "prompt": prompt}, ensure_ascii=False) + "\n")
                total += 1
    print(f"Saved: {flat_file} ({total} total)")
    return total


# ============================================================================
# Step 2: SGLang-based Self-Distillation
# ============================================================================

def generate_responses_sglang(input_dir, output_dir, gpu_list, resume=True,
                               max_samples=None):
    """
    Uses SGLang Engine for high-throughput batch inference.
    Each GPU runs a separate SGLang server instance (data parallelism).
    """
    print("=" * 60)
    print("Step 2: Self-Distillation (SGLang Engine)")
    print(f"GPUs: {gpu_list}")
    print(f"Params: temp={GEN_TEMPERATURE}, top_k={GEN_TOP_K}, "
          f"top_p={GEN_TOP_P}, max_new_tokens={GEN_MAX_NEW_TOKENS}")
    print("=" * 60)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = input_dir / "prompts_flat.jsonl"
    if not prompts_file.exists():
        print(f"ERROR: {prompts_file} not found. Run --step prepare first.")
        sys.exit(1)

    with open(prompts_file) as f:
        all_prompts = [json.loads(line) for line in f]

    if max_samples and max_samples < len(all_prompts):
        random.seed(42)
        all_prompts = random.sample(all_prompts, max_samples)
        print(f"Sampled {max_samples} prompts for testing")
    else:
        print(f"Total prompts: {len(all_prompts)}")

    # Resume
    done_hashes = set()
    merged_file = output_dir / "distilled_data.jsonl"
    if resume:
        for shard in sorted(output_dir.glob("distilled_shard_*.jsonl")):
            with open(shard) as f:
                for line in f:
                    obj = json.loads(line)
                    done_hashes.add(hashlib.md5(obj["messages"][0]["content"].encode()).hexdigest())
        if merged_file.exists():
            with open(merged_file) as f:
                for line in f:
                    obj = json.loads(line)
                    done_hashes.add(hashlib.md5(obj["messages"][0]["content"].encode()).hexdigest())
        print(f"Resume: {len(done_hashes)} already completed")

    pending = [p for p in all_prompts
               if hashlib.md5(p["prompt"].encode()).hexdigest() not in done_hashes]

    if not pending:
        print("All prompts already generated!")
        return

    # Split across GPUs
    num_gpus = len(gpu_list)
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(pending):
        chunks[i % num_gpus].append(item)
    print(f"Pending: {len(pending)}, per GPU: ~{len(pending)//num_gpus}")

    # Launch SGLang workers via multiprocessing
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    processes = []
    for idx, gpu_id in enumerate(gpu_list):
        shard_path = output_dir / f"distilled_shard_{gpu_id}.jsonl"
        p = mp.Process(
            target=_sglang_worker,
            args=(gpu_id, chunks[idx], str(shard_path), idx, SGLANG_PORT + idx),
        )
        p.start()
        processes.append(p)
        time.sleep(3)  # stagger launches to avoid port conflicts

    for p in processes:
        p.join()

    # Merge
    _merge_shards(output_dir)


def _sglang_worker(gpu_id, prompt_items, output_path, worker_id, port):
    """Single-GPU SGLang worker: launches Engine, generates, writes output."""
    import torch
    import sglang as sgl

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting SGLang Engine on port {port}...")

    engine = sgl.Engine(
        model_path=BASE_MODEL,
        tp_size=1,
        mem_fraction_static=SGLANG_MEM_FRACTION,
        trust_remote_code=True,   # model needs local modeling_mimo.py via auto_map
        port=port,
        log_level="error",
    )

    sampling_params = {
        "temperature": GEN_TEMPERATURE,
        "top_k": GEN_TOP_K,
        "top_p": GEN_TOP_P,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
    }

    print(f"[GPU {gpu_id}] Engine ready. Generating {len(prompt_items)} prompts...")

    completed = 0
    errors = 0
    batch_size = 32

    output_path = Path(output_path)

    with open(output_path, "w", buffering=1) as f:
        for batch_start in range(0, len(prompt_items), batch_size):
            batch = prompt_items[batch_start:batch_start + batch_size]
            batch_prompts = [item["prompt"] for item in batch]

            try:
                outputs = engine.generate(batch_prompts, sampling_params)
                for item, output in zip(batch, outputs):
                    sample = {
                        "messages": [
                            {"role": "user", "content": item["prompt"]},
                            {"role": "assistant", "content": output["text"]}
                        ]
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    completed += 1
            except Exception:
                for item in batch:
                    try:
                        outputs = engine.generate([item["prompt"]], sampling_params)
                        sample = {
                            "messages": [
                                {"role": "user", "content": item["prompt"]},
                                {"role": "assistant", "content": outputs[0]["text"]}
                            ]
                        }
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        completed += 1
                    except Exception:
                        errors += 1
                        if errors <= 5:
                            traceback.print_exc()

            if completed % 200 == 0:
                print(f"[GPU {gpu_id}] {completed}/{len(prompt_items)} completed")

    engine.shutdown()
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] Done: {completed} generated, {errors} errors")


def _merge_shards(output_dir):
    """Merge all shard files into a single output."""
    output_dir = Path(output_dir)
    merged_file = output_dir / "distilled_data.jsonl"

    all_lines = []
    if merged_file.exists():
        with open(merged_file) as f:
            all_lines = f.readlines()

    seen = set()
    for line in all_lines:
        try:
            obj = json.loads(line)
            seen.add(hashlib.md5(obj["messages"][0]["content"].encode()).hexdigest())
        except Exception:
            pass

    added = 0
    for shard in sorted(output_dir.glob("distilled_shard_*.jsonl")):
        with open(shard) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    h = hashlib.md5(obj["messages"][0]["content"].encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        all_lines.append(line)
                        added += 1
                except Exception:
                    pass
        shard.unlink()

    with open(merged_file, "w") as f:
        f.writelines(all_lines)

    print(f"Merged: {len(all_lines)} total samples ({added} new)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FastMTP Self-Distillation Data Generator")
    parser.add_argument("--step", choices=["prepare", "generate", "all"], default="prepare")
    parser.add_argument("--gpus", type=str, default="2",
                        help="GPU indices e.g. '2' or '4,5,6,7'")
    parser.add_argument("--input-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit prompts for testing")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
    steps = ["prepare", "generate"] if args.step == "all" else [args.step]

    for step in steps:
        if step == "prepare":
            prepare_prompts(args.output_dir, seed=args.seed)
        elif step == "generate":
            generate_responses_sglang(
                args.input_dir, args.output_dir,
                gpu_list=gpu_list, resume=not args.no_resume,
                max_samples=args.max_samples,
            )


if __name__ == "__main__":
    main()
