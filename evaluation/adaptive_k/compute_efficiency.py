"""Compute efficiency metrics from existing analysis.json."""
import json, numpy as np

with open('/ssd/yangxw/FastMTP/evaluation/adaptive_k/results/analysis.json') as f:
    data = json.load(f)

header = (f'{"Task":<35} {"Acc1":>6} {"Acc2":>6} {"Acc3":>6} | '
          f'{"K=0":>6}  {"K=3":>8}  {"G/step":>7} | '
          f'{"EAL":>8} {"K":>5} {"Gain":>7} {"G/K":>7} | '
          f'{"EAL":>8} {"K":>5} {"Gain":>7} {"G/K":>7} | '
          f'{"K90%":>5} {"K90%":>5}')
print(header)
subh = (f'{"":35} {"":6} {"":6} {"":6} | '
        f'{"EAL":>6}  {"EAL":>8}  {"(E-1)/3":>7} | '
        f'{"(A:maxEAL)":>8} {"":5} {"EAL-1":>7} {"":7} | '
        f'{"(B:maxEAL)":>8} {"":5} {"EAL-1":>7} {"":7} | '
        f'{"fixed":>5} {"A_conf":>5}')
print(subh)
print('-' * 135)

rows = []
for task in sorted(data.get('per_step_accept', {}).keys()):
    acc = data['per_step_accept'][task]
    fk = data.get('fixed_k_eal', {}).get(f'{task}_fixed_k', [])
    fk_eal = {r['K']: r['eal'] for r in fk} if fk else {}
    eal0, eal3 = fk_eal.get(0, 1.0), fk_eal.get(3, 0)
    gain3 = eal3 - 1.0
    gs3 = gain3 / 3.0 if gain3 > 0 else 0
    target_gain = 0.9 * gain3

    # Fixed K 90%
    k90_fixed = 3
    for k in range(4):
        if fk_eal.get(k, 1.0) - 1.0 >= target_gain:
            k90_fixed = k
            break

    row = {'task': task, 'acc1': acc['1'], 'acc2': acc['2'], 'acc3': acc['3'],
           'eal0': eal0, 'eal3': eal3, 'gs3': gs3, 'k90_fixed': k90_fixed}

    for sk, sl in [('confidence', 'A'), ('entropy', 'B')]:
        ad = data.get('adaptive', {}).get(f'{task}_{sk}', [])
        if not ad:
            continue
        best_eal = max(ad, key=lambda r: r['eal'])
        valid = [r for r in ad if r['mean_k'] >= 0.1]
        best_eff = max(valid, key=lambda r: (r['eal'] - 1.0) / r['mean_k']) if valid else best_eal

        # Find threshold for 90% gain
        k90 = None
        for r in ad:
            if r['eal'] - 1.0 >= target_gain:
                k90 = r['mean_k']
                break

        row[f'{sl}_eal'] = best_eal['eal']
        row[f'{sl}_K'] = best_eal['mean_k']
        row[f'{sl}_gain'] = best_eal['eal'] - 1.0
        row[f'{sl}_gk'] = (best_eal['eal'] - 1.0) / max(best_eal['mean_k'], 0.01)
        row[f'{sl}_k90'] = k90

    rows.append(row)

for r in sorted(rows, key=lambda r: -r['acc1']):
    k90a = f'{r.get("A_k90", "-"):.2f}' if r.get('A_k90') is not None else '  N/A'
    k90b = f'{r.get("B_k90", "-"):.2f}' if r.get('B_k90') is not None else '  N/A'
    print(f'{r["task"]:<35} {r["acc1"]:>6.4f} {r["acc2"]:>6.4f} {r["acc3"]:>6.4f} | '
          f'{r["eal0"]:>6.4f}  {r["eal3"]:>8.4f}  {r["gs3"]:>7.4f} | '
          f'{r.get("A_eal", 0):>8.4f} {r.get("A_K", 0):>5.2f} {r.get("A_gain", 0):>7.4f} {r.get("A_gk", 0):>7.4f} | '
          f'{r.get("B_eal", 0):>8.4f} {r.get("B_K", 0):>5.2f} {r.get("B_gain", 0):>7.4f} {r.get("B_gk", 0):>7.4f} | '
          f'{r["k90_fixed"]:>5d} {k90a:>5}')

print()
print('--- 指标说明 ---')
print('Acc1-3      = MTP step 1-3 边际接受率')
print('K=0/K=3 EAL = 固定K的 Expected Acceptance Length')
print('G/step      = (EAL_K3 - 1) / 3 : 固定K=3每步平均增益')
print('Scheme A/B (maxEAL):')
print('  EAL, K    = 最优EAL自适应的EAL值和平均步数')
print('  Gain      = EAL - 1 (MTP带来的额外接受token)')
print('  G/K       = Gain / K (每步的增量效率 — 越高越好)')
print('K90%        = 达到固定K=3的90%增益所需步数')
print()
print('关键结论:')
print('- 若自适应 G/K > 固定 G/step, 说明自适应每步更高效')
print('- K90%越小, 说明用更少步数即可达到接近固定K=3的效果')
