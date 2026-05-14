"""
Adaptive K MTP Evaluation — Scheme A (confidence) & Scheme B (entropy).

Correct metric: Expected Acceptance Length (EAL).
  EAL(K) = 1 + sum_{j=1}^{K} prod_{i=1}^{j} accept_i
  where accept_i = 1 if MTP step i draft matches target, else 0.
  In speculative decoding, step j is only verified if all prior steps accepted.

For adaptive K, the decision at each step is:
  Scheme A: if confidence < threshold → stop
  Scheme B: if entropy > threshold → stop
"""
import json, os, argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_signal(logits_last_pos):
    probs = torch.softmax(logits_last_pos.float(), dim=-1)
    conf = probs.max(dim=-1).values.item()
    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
    draft = logits_last_pos.argmax(dim=-1).item()
    return draft, conf, ent


def evaluate_sample(model, prompt_ids, max_k=3):
    """Get per-step draft, acceptance, confidence, entropy at prompt boundary."""
    L = len(prompt_ids)
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=prompt_ids.unsqueeze(0),
            max_new_tokens=max_k + 1, do_sample=False, num_beams=1,
            pad_token_id=0, eos_token_id=-1,
        )
    full_ids = gen_out[0]
    cont_ids = full_ids[L:]

    with torch.no_grad():
        outputs = model.model(input_ids=full_ids.unsqueeze(0), use_cache=False)
        main_logit = model.lm_head(outputs.last_hidden_state[:, L-1:L, :])
        draft_0, conf_0, ent_0 = compute_signal(main_logit.squeeze(0).squeeze(0))
        mtp_info = []
        for k in range(max_k):
            mtp_logit = model.lm_head(outputs.hidden_states_mtp[k][:, L-1:L, :])
            d, c, e = compute_signal(mtp_logit.squeeze(0).squeeze(0))
            mtp_info.append({'draft': d, 'conf': c, 'ent': e})

    drafts = [draft_0] + [m['draft'] for m in mtp_info]
    confs  = [conf_0]  + [m['conf']  for m in mtp_info]
    ents   = [ent_0]   + [m['ent']   for m in mtp_info]

    result = {}
    for k in range(max_k + 1):
        accepted = 1 if (k < len(cont_ids) and k < len(drafts)
                         and drafts[k] == cont_ids[k].item()) else 0
        result[f'accept_{k}'] = accepted
        result[f'conf_{k}']    = confs[k] if k < len(confs) else 0
        result[f'ent_{k}']     = ents[k]  if k < len(ents)  else 0
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_acceptance_length(sample, K_used):
    """
    Compute EAL for using MTP up to K_used steps.
    EAL = 1 + accept_1 + accept_1*accept_2 + ... + prod(accept_1..accept_K)
    """
    eal = 1.0  # step 0 always accepted
    prod = 1.0
    for k in range(1, K_used + 1):
        prod *= sample[f'accept_{k}']
        eal += prod
    return eal


def simulate_adaptive_confidence(sample, threshold, max_k=3):
    """Scheme A: stop at step k if confidence < threshold."""
    stopped = max_k
    for k in range(1, max_k + 1):
        if sample[f'conf_{k}'] < threshold:
            stopped = k - 1
            break
    return stopped, expected_acceptance_length(sample, stopped)


def simulate_adaptive_entropy(sample, threshold, max_k=3):
    """Scheme B: stop at step k if entropy > threshold."""
    stopped = max_k
    for k in range(1, max_k + 1):
        if sample[f'ent_{k}'] > threshold:
            stopped = k - 1
            break
    return stopped, expected_acceptance_length(sample, stopped)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def threshold_sweep(samples, scheme, max_k, n_thr=50):
    if scheme == 'confidence':
        sim_fn = simulate_adaptive_confidence
        all_vals = [s[f'conf_{k}'] for s in samples for k in range(1, max_k + 1)]
    else:
        sim_fn = simulate_adaptive_entropy
        all_vals = [s[f'ent_{k}'] for s in samples for k in range(1, max_k + 1)]

    thr_range = np.linspace(np.min(all_vals), np.max(all_vals), n_thr)
    results = []
    for thr in thr_range:
        eals, ks = [], []
        for s in samples:
            stopped, eal = sim_fn(s, float(thr), max_k)
            eals.append(eal)
            ks.append(stopped)
        results.append({'threshold': float(thr), 'eal': float(np.mean(eals)),
                        'mean_k': float(np.mean(ks))})
    return results


def compute_fixed_k_baseline(samples, max_k):
    """Compute EAL for fixed K values."""
    baseline = []
    for k in range(max_k + 1):
        eals = [expected_acceptance_length(s, k) for s in samples]
        baseline.append({'K': k, 'eal': float(np.mean(eals))})
    return baseline


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(task_results, analysis, output_dir, max_k):
    os.makedirs(output_dir, exist_ok=True)
    tasks = sorted(task_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tasks), 1)))

    # ---- 1. Per-step acceptance, confidence, entropy ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (metric, ylabel) in enumerate([
        ('accept', 'Acceptance Rate'), ('conf', 'Mean Confidence'), ('ent', 'Mean Entropy')
    ]):
        ax = axes[i]
        for task, c in zip(tasks, colors):
            vals = [np.mean([s[f'{metric}_{k}'] for s in task_results[task]]) for k in range(max_k + 1)]
            ax.plot(range(max_k + 1), vals, 'o-', color=c, label=task, lw=2, ms=8)
        ax.set_xlabel('Draft Step K'); ax.set_ylabel(ylabel); ax.set_title(f'Per-Step {ylabel}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(range(max_k + 1))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'per_step.png'), dpi=150); plt.close()

    # ---- 2. Fixed K: EAL vs K ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, c in zip(tasks, colors):
        fk = analysis[f'{task}_fixed_k']
        ks = [r['K'] for r in fk]
        eals = [r['eal'] for r in fk]
        ax.plot(ks, eals, 'o-', color=c, label=task, lw=2, ms=8)
        # Annotate EAL values
        for k, eal in zip(ks, eals):
            ax.annotate(f'{eal:.3f}', (k, eal), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7, color=c)
    ax.set_xlabel('Fixed K'); ax.set_ylabel('Expected Acceptance Length')
    ax.set_title('Fixed K Baseline: EAL vs K')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(range(max_k + 1))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'fixed_k_eal.png'), dpi=150); plt.close()

    # ---- 3. Threshold sweep: EAL vs threshold ----
    for scheme, slabel, xlabel_extra in [
        ('confidence', 'A: Confidence', '\n(lower → stop earlier)'),
        ('entropy', 'B: Entropy', '\n(higher → stop earlier)'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # All tasks combined
        ax = axes[0]
        ar = analysis[f'all_{scheme}']
        xs, ys = [r['threshold'] for r in ar], [r['eal'] for r in ar]
        ax.plot(xs, ys, 'b-', lw=2)
        best = max(ar, key=lambda r: r['eal'])
        ax.scatter(best['threshold'], best['eal'], c='red', s=100, marker='*',
                  edgecolors='black', lw=1, zorder=5,
                  label=f'Best: thr={best["threshold"]:.3f}\nEAL={best["eal"]:.4f}, K={best["mean_k"]:.1f}')
        ax.set_xlabel(f'Threshold{xlabel_extra}'); ax.set_ylabel('EAL')
        ax.set_title(f'Scheme {slabel} — All Tasks'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # Per-task
        ax = axes[1]
        for task, c in zip(tasks, colors):
            tr = analysis[f'{task}_{scheme}']
            xs_t, ys_t = [r['threshold'] for r in tr], [r['eal'] for r in tr]
            ax.plot(xs_t, ys_t, '-', color=c, label=task, lw=2)
            best_t = max(tr, key=lambda r: r['eal'])
            ax.scatter(best_t['threshold'], best_t['eal'], color=c, s=60,
                      marker='*', edgecolors='black', lw=1, zorder=5)
        ax.set_xlabel(f'Threshold{xlabel_extra}'); ax.set_ylabel('EAL')
        ax.set_title(f'Scheme {slabel} — Per Task'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'threshold_{scheme}.png'), dpi=150); plt.close()

    # ---- 4. EAL vs Mean K trade-off ----
    fig, ax = plt.subplots(figsize=(11, 7))
    for task, c in zip(tasks, colors):
        fk = analysis[f'{task}_fixed_k']
        ax.plot([r['K'] for r in fk], [r['eal'] for r in fk], '--', color=c, alpha=0.3, lw=1)
        ax.scatter([r['K'] for r in fk], [r['eal'] for r in fk], color=c, s=150,
                  marker='*', zorder=6, edgecolors='black', lw=1.5, label=f'{task} (fixed)')
    for scheme, marker, alpha in [('confidence', 'o', 0.4), ('entropy', 's', 0.4)]:
        for task, c in zip(tasks, colors):
            tr = analysis[f'{task}_{scheme}']
            ax.scatter([r['mean_k'] for r in tr], [r['eal'] for r in tr],
                      color=c, s=18, marker=marker, alpha=alpha, zorder=3)
    # Best adaptive points
    for task, c in zip(tasks, colors):
        for scheme, marker in [('confidence', 'D'), ('entropy', '^')]:
            tr = analysis[f'{task}_{scheme}']
            best = max(tr, key=lambda r: r['eal'])
            ax.scatter(best['mean_k'], best['eal'], color=c, s=120,
                      marker=marker, edgecolors='black', lw=2, zorder=8)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='*', color='gray', label='Fixed K baseline', ms=12, ls='None'),
        Line2D([0],[0], marker='D', color='gray', label='Best adaptive (conf)', ms=10, ls='None'),
        Line2D([0],[0], marker='^', color='gray', label='Best adaptive (ent)', ms=10, ls='None'),
    ]
    for task, c in zip(tasks, colors):
        handles.append(Line2D([0],[0], color=c, label=task, lw=2))
    ax.set_xlabel('Mean K Used'); ax.set_ylabel('Expected Acceptance Length (EAL)')
    ax.set_title('Adaptive K Trade-off: Draft Steps vs EAL')
    ax.legend(handles=handles, fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'tradeoff.png'), dpi=150); plt.close()

    # ---- 5. Summary table ----
    fig, ax = plt.subplots(figsize=(11, 5)); ax.axis('off')
    rows = []; cell_colors = []
    for task in tasks:
        fk = analysis[f'{task}_fixed_k']
        fk_k0, fk_k3 = fk[0]['eal'], fk[3]['eal']
        row_bg = []
        for scheme, label in [('confidence', 'Conf'), ('entropy', 'Ent')]:
            tr = analysis[f'{task}_{scheme}']
            best = max(tr, key=lambda r: r['eal'])
            delta = best['eal'] - fk_k3
            color = '#c8e6c9' if delta > 0 else '#ffcdd2'
            rows.append([task, label, f'{best["threshold"]:.4f}', f'{best["mean_k"]:.2f}',
                        f'{best["eal"]:.4f}', f'{fk_k0:.4f}', f'{fk_k3:.4f}', f'{delta:+.4f}'])
            row_bg.append(color)

    tbl = ax.table(cellText=rows,
                   colLabels=['Task', 'Scheme', 'Best Thr', 'Mean K', 'Adaptive EAL',
                              'K=0 EAL', 'Fixed K=3 EAL', 'Δ vs K=3'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.15, 1.8)
    # Color rows
    for i, row_color in enumerate([c for pair in zip(row_bg, row_bg) for c in pair]):
        pass  # row coloring a bit complex with matplotlib, skip for now
    ax.set_title('Adaptive K Summary (Expected Acceptance Length)', fontsize=14, pad=20)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150); plt.close()

    # ---- 6. Bar chart: acceptance rate per step per task ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks)); w = 0.2
    for k in range(max_k + 1):
        vals = [np.mean([s[f'accept_{k}'] for s in task_results[t]]) for t in tasks]
        ax.bar(x + k * w, vals, w, label=f'Step {k}')
    ax.set_xticks(x + w * max_k / 2); ax.set_xticklabels(tasks)
    ax.set_ylabel('Acceptance Rate'); ax.set_title('Per-Step Acceptance Rate by Task')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'bar_accept.png'), dpi=150); plt.close()

    print(f'Plots → {output_dir}/')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/ssd/yangxw/FastMTP/model/FastMTP')
    parser.add_argument('--data-dir', default='/ssd/yangxw/FastMTP/evaluation')
    parser.add_argument('--output-dir', default='/ssd/yangxw/FastMTP/evaluation/adaptive_k/results')
    parser.add_argument('--max-samples', type=int, default=80)
    parser.add_argument('--max-k', type=int, default=3)
    parser.add_argument('--n-thresholds', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f'Loading model (GPU {args.gpu})...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='cuda:0',
        trust_remote_code=False, low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    print(f'Loaded. dtype={model.dtype}')

    data_dir = Path(args.data_dir)

    # Dataset definitions: (subdir, glob_pattern, prompt_key, category_key)
    dataset_specs = [
        ('spec_bench', '*.jsonl', 'turns', 'category'),
        ('math_500', '*.jsonl', 'problem', 'subject'),
        ('mt_bench', '*.jsonl', 'turns', 'category'),
        ('livecodebench_v6', '*.jsonl', 'question_content', 'difficulty'),
        ('c_eval', '*.jsonl', 'question', 'category'),
    ]

    datasets = {}
    for subdir, pattern, prompt_key, cat_key in dataset_specs:
        sd = data_dir / subdir
        if not sd.exists():
            continue
        for fp in sorted(sd.glob(pattern)):
            if 'answer' in fp.name or 'result' in fp.name:
                continue
            # Collect samples, group by category within file
            all_samples = []
            with open(fp) as f:
                for line in f:
                    obj = json.loads(line)
                    prompt_val = obj.get(prompt_key, '')
                    if isinstance(prompt_val, list):
                        prompt_val = prompt_val[0] if prompt_val else ''
                    if not prompt_val:
                        continue
                    obj['_prompt'] = prompt_val
                    all_samples.append(obj)

            # Group by category for multi-category files
            from collections import defaultdict
            grouped = defaultdict(list)
            for s in all_samples:
                cat_raw = s.get(cat_key, 'unknown')
                grouped[str(cat_raw)].append(s)

            for cat_raw, samples in grouped.items():
                if len(samples) < 5:  # skip tiny categories
                    continue
                cat = f'{subdir}/{cat_raw}'
                datasets[cat] = samples
                print(f'Loaded: {subdir}/{fp.name} → "{cat_raw}": {len(samples)} samples')

    task_results, all_samples = {}, []
    for task, samples in datasets.items():
        evals = []
        for sample in tqdm(samples[:args.max_samples], desc=task):
            enc = tokenizer(sample['_prompt'], return_tensors='pt', truncation=True, max_length=2048)
            evals.append(evaluate_sample(model, enc['input_ids'][0].to(model.device), max_k=args.max_k))
        task_results[task] = evals
        all_samples.extend(evals)

    # ---- Print per-step ----
    print(f'\n{"="*60}\nPer-Step Acceptance\n{"="*60}')
    print(f'{"Task":<16}', *(f'Step{k:>8}' for k in range(args.max_k + 1)))
    print('-' * (16 + 9 * (args.max_k + 1)))
    for task in sorted(task_results):
        accs = [np.mean([s[f'accept_{k}'] for s in task_results[task]]) for k in range(args.max_k + 1)]
        print(f'{task:<16}', *(f'{a:>8.4f}' for a in accs))

    # ---- Analysis ----
    print(f'\n{"="*60}\nAdaptive K Analysis (EAL metric)\n{"="*60}')
    analysis = {}

    # Fixed K baseline
    for task in sorted(task_results):
        analysis[f'{task}_fixed_k'] = compute_fixed_k_baseline(task_results[task], args.max_k)
    analysis['all_fixed_k'] = compute_fixed_k_baseline(all_samples, args.max_k)

    # Threshold sweeps
    for scheme in ['confidence', 'entropy']:
        analysis[f'all_{scheme}'] = threshold_sweep(all_samples, scheme, args.max_k, args.n_thresholds)
        best = max(analysis[f'all_{scheme}'], key=lambda r: r['eal'])
        print(f'\nAll tasks, Scheme {"A" if scheme=="confidence" else "B"}: '
              f'best thr={best["threshold"]:.4f} → EAL={best["eal"]:.4f}, K={best["mean_k"]:.2f}')

        for task in sorted(task_results):
            analysis[f'{task}_{scheme}'] = threshold_sweep(
                task_results[task], scheme, args.max_k, args.n_thresholds)

    # Per-task summary
    print(f'\n{"="*60}\nPer-Task Summary\n{"="*60}')
    header = f'{"Task":<16} {"Fixed K=0":>10} {"Fixed K=3":>10} {"Best Adaptive":>14} {"Scheme":>8} {"Δ vs K=3":>10}'
    print(header); print('-' * len(header))
    for task in sorted(task_results):
        fk = analysis[f'{task}_fixed_k']
        for scheme, slabel in [('confidence', 'Conf'), ('entropy', 'Ent')]:
            tr = analysis[f'{task}_{scheme}']
            best = max(tr, key=lambda r: r['eal'])
            delta = best['eal'] - fk[3]['eal']
            print(f'{task:<16} {fk[0]["eal"]:>10.4f} {fk[3]["eal"]:>10.4f} '
                  f'{best["eal"]:>10.4f} (K={best["mean_k"]:.1f}) {slabel:>8} {delta:>+10.4f}')

    # ---- Plots ----
    print(f'\nGenerating plots...')
    plot_results(task_results, analysis, args.output_dir, args.max_k)

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    save = {'per_step_accept': {t: {str(k): float(np.mean([s[f'accept_{k}'] for s in task_results[t]]))
                                    for k in range(args.max_k + 1)} for t in sorted(task_results)},
            'fixed_k_eal': {key: val for key, val in analysis.items() if 'fixed_k' in key},
            'adaptive': {key: val for key, val in analysis.items() if 'fixed_k' not in key}}
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f'\nSaved → {args.output_dir}/')


if __name__ == '__main__':
    main()
