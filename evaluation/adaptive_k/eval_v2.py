"""
Adaptive K evaluation v2 — uses model.forward() with labels for correct alignment.

Alignment (derived from modeling_mimo.py L239-253):
  At last prompt position pos=L-1:
    main_logits[L-1]  vs labels[L-1] = token_L   (t_0)
    mtp_step1[L-1]    vs labels[L]   = token_L+1 (t_1) — labels rolled 1x
    mtp_step2[L-1]    vs labels[L+1] = token_L+2 (t_2) — labels rolled 2x
    mtp_step3[L-1]    vs labels[L+2] = token_L+3 (t_3) — labels rolled 3x

Approach: run forward(full_seq, labels=full_seq), extract logits at prompt
boundary, compare with corresponding continuation tokens.
"""
import json, os, argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def roll_tensor(tensor, shifts=-1, dims=-1, fill_num=0.0):
    rolled = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled.select(dims, shifts).fill_(fill_num)
    return rolled


def compute_metrics(logits_last_pos):
    """From logits at a single position (vocab_size,), compute confidence/entropy/draft."""
    probs = torch.softmax(logits_last_pos.float(), dim=-1)
    confidence = probs.max(dim=-1).values.item()
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
    draft_token = logits_last_pos.argmax(dim=-1).item()
    return draft_token, confidence, entropy


def evaluate_sample(model, tokenizer, sample, max_k=3, max_cont=128):
    """
    Evaluate one sample.

    Returns per-step acceptance, confidence, entropy at the prompt boundary.
    """
    prompt = sample['turns'][0]

    # Tokenize prompt
    prompt_enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    prompt_ids = prompt_enc['input_ids'][0]
    L = len(prompt_ids)

    # Generate greedy continuation as ground truth
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=prompt_ids.unsqueeze(0).to(model.device),
            max_new_tokens=max(4, max_k + 1),
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=-1,
        )
    full_ids = gen_out[0].cpu()
    cont_ids = full_ids[L:]  # continuation tokens t_0, t_1, ...

    # Build full sequence: prompt + continuation
    full_seq = full_ids.to(model.device)

    # Run forward with labels to get correctly aligned predictions
    with torch.no_grad():
        outputs = model.model(input_ids=full_seq.unsqueeze(0), use_cache=False)
        hidden_main = outputs.last_hidden_state  # (1, total_len, hidden)
        hidden_mtp = outputs.hidden_states_mtp   # tuple of K tensors

        # Logits at all positions
        main_logits = model.lm_head(hidden_main)  # (1, total_len, vocab)
        mtp_logits_list = [model.lm_head(h[:, -1:, :]).squeeze(0).squeeze(0) for h in hidden_mtp]
        # mtp_logits_list[k] = logits at ALL positions for MTP step k+1

    # Extract at last prompt position L-1
    main_at_Lm1 = main_logits[0, L-1, :]  # (vocab,)
    draft_t0, conf_t0, ent_t0 = compute_metrics(main_at_Lm1)

    # MTP at L-1
    mtp_drafts = []
    mtp_confs = []
    mtp_ents = []

    # Alignment: MTP step k at pos L-1 predicts token at position L+k
    # (labels are rolled k+1 times in the loss)
    # In training: labels rolled k+1 times, so mtp_step_k at pos i → labels[i+k+1]
    # At pos L-1: mtp_step_k → labels[L-1+k+1] = labels[L+k] = token_L+k+1
    # So MTP step 1 → token_L+1 = t_1, MTP step 2 → token_L+2 = t_2, ...

    for k in range(max_k):
        mtp_logit_at_Lm1 = model.lm_head(hidden_mtp[k][:, L-1:L, :]).squeeze(0).squeeze(0)
        d, c, e = compute_metrics(mtp_logit_at_Lm1)
        mtp_drafts.append(d)
        mtp_confs.append(c)
        mtp_ents.append(e)

    # Acceptance check
    drafts = [draft_t0] + mtp_drafts  # t_0, t_1, t_2, t_3
    confs = [conf_t0] + mtp_confs
    ents = [ent_t0] + mtp_ents

    results = {
        'prompt_len': L,
        'cont_len': len(cont_ids),
    }
    for k in range(max_k + 1):
        if k < len(cont_ids) and k < len(drafts):
            accepted = 1 if drafts[k] == cont_ids[k].item() else 0
        else:
            accepted = 0
        results[f'accept_{k}'] = accepted
        results[f'conf_{k}'] = confs[k] if k < len(confs) else 0
        results[f'ent_{k}'] = ents[k] if k < len(ents) else 0

    return results


def run_evaluation(model, tokenizer, datasets, args):
    """Run evaluation on all datasets."""
    all_results = {}

    for cat, samples in datasets.items():
        eval_samples = samples[:args.max_samples]
        print(f'\n{"="*60}')
        print(f'Evaluating: {cat} ({len(eval_samples)} samples)')
        print(f'{"="*60}')

        cat_results = {
            'acceptance_by_step': {k: [] for k in range(args.max_k + 1)},
            'confidence_by_step': {k: [] for k in range(args.max_k + 1)},
            'entropy_by_step': {k: [] for k in range(args.max_k + 1)},
        }

        for sample in tqdm(eval_samples, desc=cat):
            r = evaluate_sample(model, tokenizer, sample, max_k=args.max_k)
            for k in range(args.max_k + 1):
                cat_results['acceptance_by_step'][k].append(r[f'accept_{k}'])
                cat_results['confidence_by_step'][k].append(r[f'conf_{k}'])
                cat_results['entropy_by_step'][k].append(r[f'ent_{k}'])

        # Summary
        for k in range(args.max_k + 1):
            accs = cat_results['acceptance_by_step'][k]
            confs = cat_results['confidence_by_step'][k]
            ents = cat_results['entropy_by_step'][k]
            print(f'  Step {k}: acc={np.mean(accs):.4f}, conf={np.mean(confs):.4f}, ent={np.mean(ents):.4f} (n={len(accs)})')

        all_results[cat] = cat_results

    return all_results


def analyze_adaptive(all_results, max_k=3, n_thr=40):
    """Analyze adaptive K: Fixed K vs Scheme A (confidence) vs Scheme B (entropy)."""
    analysis = {}
    for cat, res in all_results.items():
        n_samples = len(res['acceptance_by_step'][0])
        cat_analysis = {'per_step': {}, 'fixed_k': [], 'scheme_a': [], 'scheme_b': []}

        # Per-step stats
        for k in range(max_k + 1):
            accs = res['acceptance_by_step'][k]
            confs = res['confidence_by_step'][k]
            ents = res['entropy_by_step'][k]
            cat_analysis['per_step'][k] = {
                'acc': np.mean(accs), 'conf': np.mean(confs), 'ent': np.mean(ents), 'n': len(accs)
            }

        # Fixed K baseline
        for k in range(max_k + 1):
            all_acc = []
            for j in range(k + 1):
                all_acc.extend(res['acceptance_by_step'][j])
            cat_analysis['fixed_k'].append({'K': k, 'acc': np.mean(all_acc)})

        # Scheme A: Confidence threshold (lower conf -> stop)
        all_conf = [res['confidence_by_step'][k][i]
                    for k in range(max_k + 1) for i in range(n_samples)]
        for thr in np.linspace(np.min(all_conf), np.max(all_conf), n_thr):
            sample_acc = []
            sample_k = []
            for i in range(n_samples):
                stop_k = max_k
                for k in range(1, max_k + 1):
                    if res['confidence_by_step'][k][i] < thr:
                        stop_k = k - 1
                        break
                sample_k.append(stop_k)
                accs_used = [res['acceptance_by_step'][j][i] for j in range(stop_k + 1)]
                sample_acc.append(np.mean(accs_used))
            cat_analysis['scheme_a'].append({
                'thr': float(thr), 'acc': float(np.mean(sample_acc)),
                'mean_k': float(np.mean(sample_k))
            })

        # Scheme B: Entropy threshold (higher entropy -> stop)
        all_ent = [res['entropy_by_step'][k][i]
                   for k in range(max_k + 1) for i in range(n_samples)]
        for thr in np.linspace(np.min(all_ent), np.max(all_ent), n_thr):
            sample_acc = []
            sample_k = []
            for i in range(n_samples):
                stop_k = max_k
                for k in range(1, max_k + 1):
                    if res['entropy_by_step'][k][i] > thr:
                        stop_k = k - 1
                        break
                sample_k.append(stop_k)
                accs_used = [res['acceptance_by_step'][j][i] for j in range(stop_k + 1)]
                sample_acc.append(np.mean(accs_used))
            cat_analysis['scheme_b'].append({
                'thr': float(thr), 'acc': float(np.mean(sample_acc)),
                'mean_k': float(np.mean(sample_k))
            })

        analysis[cat] = cat_analysis
    return analysis


def plot_results(analysis, output_dir):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    cats = sorted(analysis.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cats), 1)))
    max_k = max(max(analysis[c]['per_step'].keys()) for c in cats)

    # ---- 1. Per-step analysis ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric, ylabel, title) in enumerate([
        ('acc', 'Acceptance Rate', 'Acceptance Rate per Draft Step'),
        ('conf', 'Mean Confidence', 'Confidence per Draft Step'),
        ('ent', 'Mean Entropy', 'Entropy per Draft Step'),
    ]):
        ax = axes[i]
        for cat, c in zip(cats, colors):
            steps = sorted(analysis[cat]['per_step'].keys())
            vals = [analysis[cat]['per_step'][s][metric] for s in steps]
            ax.plot(steps, vals, 'o-', color=c, label=cat, lw=2, ms=8)
        ax.set_xlabel('Draft Step K'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(range(max_k + 1))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'per_step.png'), dpi=150); plt.close()

    # ---- 2. Threshold sweep ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (scheme, xlabel) in enumerate([
        ('scheme_a', 'Confidence Threshold (low → stop)'),
        ('scheme_b', 'Entropy Threshold (high → stop)'),
    ]):
        ax = axes[i]
        for cat, c in zip(cats, colors):
            data = analysis[cat][scheme]
            if data:
                xs, ys = [r['thr'] for r in data], [r['acc'] for r in data]
                ax.plot(xs, ys, '-', color=c, label=cat, lw=2)
                best = max(data, key=lambda r: r['acc'])
                ax.scatter(best['thr'], best['acc'], color=c, s=80, marker='*',
                          edgecolors='black', lw=1)
        ax.set_xlabel(xlabel); ax.set_ylabel('Acceptance Rate')
        ax.set_title(f'Scheme {"A" if scheme == "scheme_a" else "B"}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'threshold_sweep.png'), dpi=150); plt.close()

    # ---- 3. Trade-off ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, c in zip(cats, colors):
        fixed = analysis[cat]['fixed_k']
        xf, yf = [r['K'] for r in fixed], [r['acc'] for r in fixed]
        ax.plot(xf, yf, '-', color=c, alpha=0.3, lw=1)
        ax.scatter(xf, yf, c=[c], s=120, marker='*', zorder=5, edgecolors='black', lw=1,
                  label=f'{cat} (fixed K)')

        sa, sb = analysis[cat]['scheme_a'], analysis[cat]['scheme_b']
        if sa: ax.scatter([r['mean_k'] for r in sa], [r['acc'] for r in sa],
                         c=[c], s=20, marker='o', alpha=0.4, zorder=3)
        if sb: ax.scatter([r['mean_k'] for r in sb], [r['acc'] for r in sb],
                         c=[c], s=20, marker='^', alpha=0.4, zorder=3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='*', color='gray', label='Fixed K', ms=12, ls='None'),
               Line2D([0],[0], marker='o', color='gray', label='Scheme A (conf)', ms=8, ls='None'),
               Line2D([0],[0], marker='^', color='gray', label='Scheme B (ent)', ms=8, ls='None')]
    for cat, c in zip(cats, colors):
        handles.append(Line2D([0],[0], color=c, label=cat, lw=2))
    ax.set_xlabel('Mean K Used'); ax.set_ylabel('Acceptance Rate')
    ax.set_title('Trade-off: Draft Steps vs Acceptance Rate'); ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'tradeoff.png'), dpi=150); plt.close()

    # ---- 4. Bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats)); width = 0.2
    for k in range(max_k + 1):
        accs = [analysis[c]['per_step'].get(k, {}).get('acc', 0) for c in cats]
        ax.bar(x + k * width, accs, width, label=f'Step {k}')
    ax.set_xlabel('Task'); ax.set_ylabel('Acceptance Rate'); ax.set_title('Per-Step Acceptance by Task')
    ax.set_xticks(x + width * max_k / 2); ax.set_xticklabels(cats); ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'bar.png'), dpi=150); plt.close()

    print(f'Plots saved to {output_dir}/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/ssd/yangxw/FastMTP/model/FastMTP')
    parser.add_argument('--data-dir', default='/ssd/yangxw/FastMTP/evaluation')
    parser.add_argument('--output-dir', default='/ssd/yangxw/FastMTP/evaluation/adaptive_k/results_v2')
    parser.add_argument('--max-samples', type=int, default=80)
    parser.add_argument('--max-k', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f'Loading model from {args.model_path} on GPU {args.gpu}...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='cuda:0',
        trust_remote_code=False, low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    print(f'Loaded. dtype={model.dtype}')

    # Load datasets
    spec_dir = Path(args.data_dir) / 'spec_bench'
    datasets = {}
    for fp in sorted(spec_dir.glob('*.jsonl')):
        if 'answer' in fp.name or 'result' in fp.name:
            continue
        samples = [json.loads(l) for l in open(fp)]
        if samples:
            cat = samples[0].get('category', fp.stem)
            datasets[cat] = samples
            print(f'Loaded {len(samples)}: {fp.name} ({cat})')

    # Evaluate
    all_results = run_evaluation(model, tokenizer, datasets, args)

    # Analyze
    print(f'\n{"="*60}\nAdaptive K Analysis\n{"="*60}')
    analysis = analyze_adaptive(all_results, max_k=args.max_k)

    for cat, a in analysis.items():
        print(f'\n--- {cat} ---')
        print('Fixed K:', ', '.join(f'K={r["K"]}:{r["acc"]:.4f}' for r in a['fixed_k']))
        for scheme in ['scheme_a', 'scheme_b']:
            data = a[scheme]
            if data:
                best = max(data, key=lambda r: r['acc'])
                print(f'{scheme} best: thr={best["thr"]:.4f} acc={best["acc"]:.4f} K={best["mean_k"]:.2f}')

    # Plots & save
    plot_results(analysis, args.output_dir)
    serializable = {}
    for cat, a in analysis.items():
        serializable[cat] = {k: (v if k not in ('per_step',) else {str(kk): vv for kk, vv in v.items()})
                            for k, v in a.items()}
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'\nResults → {args.output_dir}/')


if __name__ == '__main__':
    main()
