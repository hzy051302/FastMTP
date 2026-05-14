"""
Predictor training label generation: simulating speculative decoding verification chain.

This script demonstrates the label generation process without training,
so we can inspect the labels and understand the chain simulation.
"""
import os, json, torch
import numpy as np
from collections import defaultdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def demo_label_generation():
    """Walk through label generation step by step."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        '/ssd/yangxw/FastMTP/model/FastMTP',
        torch_dtype=torch.bfloat16, device_map='cuda:0',
        trust_remote_code=False, low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('/ssd/yangxw/FastMTP/model/FastMTP', trust_remote_code=False)
    K = 3

    # Load diverse samples
    samples = []
    for fname, key in [
        ('spec_bench/spec_bench_qa.jsonl', 'turns'),
        ('spec_bench/spec_bench_rag.jsonl', 'turns'),
        ('livecodebench_v6/livecodebench_sampled_80_reformat_2.jsonl', 'question_content'),
        ('mt_bench/question.jsonl', 'turns'),
    ]:
        fp = Path('/ssd/yangxw/FastMTP/evaluation') / fname
        if fp.exists():
            for line in open(fp):
                obj = json.loads(line)
                prompt = obj.get(key, '')
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ''
                if prompt:
                    samples.append({
                        'task': fname.split('/')[0],
                        'prompt': prompt,
                    })

    print(f'Loaded {len(samples)} samples\n')
    np.random.seed(42)
    demo_samples = np.random.choice(samples, 3, replace=False)

    for si, sample in enumerate(demo_samples):
        print(f'{"="*70}')
        print(f'Sample {si}: [{sample["task"]}]')
        print(f'Prompt: {sample["prompt"][:80]}...')
        print(f'{"="*70}')

        enc = tokenizer(sample['prompt'], return_tensors='pt', truncation=True, max_length=512)
        pid = enc['input_ids'][0].to(model.device)
        L = len(pid)

        # ── Step 1: Generate ground truth targets (no grad) ──
        with torch.no_grad():
            gen = model.generate(pid.unsqueeze(0), max_new_tokens=K+1,
                                do_sample=False, num_beams=1,
                                pad_token_id=0, eos_token_id=-1)
            full_ids = gen[0]
            targets = full_ids[L:].tolist()  # [t_0, t_1, ..., t_K]

        print(f'\nTarget tokens (greedy decoded):')
        for i, tid in enumerate(targets[:K+1]):
            print(f'  t_{i}: {tid:>6} -> "{tokenizer.decode(tid)}"')

        # ── Step 2: MTP forward to get drafts at last prompt position ──
        with torch.no_grad():
            outputs = model.model(input_ids=full_ids.unsqueeze(0), use_cache=False)

            # Main model draft (t_0)
            main_logit = model.lm_head(outputs.last_hidden_state[:, L-1:L, :])
            main_draft = main_logit.argmax(dim=-1).item()
            main_conf = torch.softmax(main_logit.float(), dim=-1).max().item()

            # MTP drafts (t_1..t_K)
            mtp_drafts = []
            mtp_confs = []
            mtp_hidden = []
            for k in range(K):
                h = outputs.hidden_states_mtp[k][:, L-1, :]  # hidden at last prompt pos
                logit = model.lm_head(outputs.hidden_states_mtp[k][:, L-1:L, :])
                probs = torch.softmax(logit.float(), dim=-1)
                mtp_drafts.append(logit.argmax(dim=-1).item())
                mtp_confs.append(probs.max().item())
                mtp_hidden.append(h)

        print(f'\nDraft tokens from MTP:')
        print(f'  Main (t_0): {main_draft:>6} -> "{tokenizer.decode(main_draft)}" (conf={main_conf:.4f})')
        for k in range(K):
            print(f'  MTP step {k+1} (t_{k+1}): {mtp_drafts[k]:>6} -> '
                  f'"{tokenizer.decode(mtp_drafts[k])}" (conf={mtp_confs[k]:.4f})')

        # ── Step 3: Simulate speculative decoding verification chain ──
        print(f'\n{"─"*50}')
        print(f'Speculative Decoding Chain Simulation:')
        print(f'{"─"*50}')

        # Step 0: main model always "accepted" (it IS the target model)
        accepted_chain = [True]
        labels = {}

        print(f'  Step 0 (main): draft={main_draft}({tokenizer.decode(main_draft)!r}), '
              f'target={targets[0]}({tokenizer.decode(targets[0])!r}) → ALWAYS ACCEPTED')

        for k in range(1, K+1):
            draft = mtp_drafts[k-1]
            target = targets[k] if k < len(targets) else -1
            match = (draft == target)
            # CONDITIONAL: only "verified" if chain is still alive
            will_be_verified = accepted_chain[-1]
            effective_label = match if will_be_verified else 'NOT_VERIFIED'

            print(f'  Step {k} (MTP): draft={draft}({tokenizer.decode(draft)!r}), '
                  f'target={target}({tokenizer.decode(target)!r})')
            print(f'    match={match}, chain_alive={will_be_verified}, '
                  f'effective_label={effective_label}')

            if will_be_verified:
                labels[k] = 1 if match else 0
                accepted_chain.append(match)
            else:
                labels[k] = -100  # IGNORE (never verified)
                accepted_chain.append(False)

        # ── Step 4: Summary of training labels ──
        print(f'\nTraining labels for predictor:')
        print(f'  (IGNORE=-100 means step was never reached in chain)')
        for k in range(1, K+1):
            label = labels.get(k, -100)
            label_str = f'{label}' if label != -100 else 'IGNORE'
            print(f'  Step {k}: label={label_str}, '
                  f'conf={mtp_confs[k-1]:.4f}, '
                  f'draft="{tokenizer.decode(mtp_drafts[k-1])}"')

        if K+1 < len(targets):
            # K=5 scenario: show what labels would look like beyond trained K
            print(f'\nNote: model trained up to K=3. Beyond that:')
            for k in range(K+1, min(len(targets), 6)):
                print(f'  Step {k}: unknown (model not trained for this step)')

        print()

    # ── Aggregate statistics ──
    print(f'\n{"="*70}')
    print('Aggregate analysis across all samples...')
    print(f'{"="*70}')

    all_labels = defaultdict(list)
    for sample in samples[:50]:
        enc = tokenizer(sample['prompt'], return_tensors='pt', truncation=True, max_length=512)
        pid = enc['input_ids'][0].to(model.device)
        L = len(pid)

        with torch.no_grad():
            gen = model.generate(pid.unsqueeze(0), max_new_tokens=K+1, do_sample=False,
                                num_beams=1, pad_token_id=0, eos_token_id=-1)
            targets = gen[0, L:].tolist()
            full_ids = gen[0]
            outputs = model.model(input_ids=full_ids.unsqueeze(0), use_cache=False)

        chain_ok = True
        for k in range(1, K+1):
            logit = model.lm_head(outputs.hidden_states_mtp[k-1][:, L-1:L, :])
            draft = logit.argmax(dim=-1).item()
            target = targets[k] if k < len(targets) else -1

            if chain_ok:
                label = 1 if draft == target else 0
                chain_ok = (label == 1)
            else:
                label = -100

            all_labels[k].append(label)

    print(f'\nLabel distribution (50 samples):')
    print(f'{"Step":>6} {"pos":>6} {"neg":>6} {"ignored":>9} {"pos_rate":>10}')
    for k in range(1, K+1):
        labels_k = all_labels[k]
        pos = sum(1 for l in labels_k if l == 1)
        neg = sum(1 for l in labels_k if l == 0)
        ign = sum(1 for l in labels_k if l == -100)
        total_valid = pos + neg
        rate = pos / total_valid if total_valid > 0 else 0
        print(f'{k:>6} {pos:>6} {neg:>6} {ign:>9} {rate:>10.4f}')

    print(f'\nKey observation: later steps have fewer valid labels (more IGNORE)')
    print(f'because the chain is broken earlier. This is the conditional nature.')
    print(f'\nLabel generation approach used: SINGLE forward pass with teacher-forcing.')


if __name__ == '__main__':
    demo_label_generation()
