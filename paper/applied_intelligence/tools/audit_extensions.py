"""CPU recheck of the external-baseline, seed and ChEMU numbers from per-document outcomes.

Every count, interval, paired test and Holm adjustment is recomputed from the saved
document_outcomes.csv files and compared with the queue's own summary.json. Writes
evidence/extensions_audit.json and exits non-zero on any disagreement.
"""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parents[1] / 'evidence'
CAP = '8'
problems = []


def flag(value):
    return value == 'True'


def load(path, key=('arm', 'budget')):
    table = defaultdict(dict)
    for row in csv.DictReader(path.open(encoding='utf-8')):
        table[tuple(row[k] for k in key)][row['document']] = row
    return table


def wilson(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def mcnemar(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def arm_stats(rows, seconds=False):
    accepted = [r['accepted'] for r in rows.values()]
    confirmed = sum(flag(r['confirmed']) for r in rows.values())
    out = {'documents': len(rows), 'confirmed': confirmed,
           'rate_percent': round(100 * confirmed / len(rows), 1), 'wilson95': wilson(confirmed, len(rows))}
    if all(a != '' for a in accepted):
        acc = sum(flag(a) for a in accepted)
        out |= {'accepted': acc, 'gap_percent': round(100 * (acc - confirmed) / acc, 1)}
    if 'output_tokens' in next(iter(rows.values())):
        out |= {key: sum(int(r[key]) for r in rows.values()) for key in ('calls', 'input_tokens', 'output_tokens')}
    if seconds and 'generation_seconds' in next(iter(rows.values())):
        out['generation_seconds'] = round(sum(float(r['generation_seconds']) for r in rows.values()), 1)
    return out


def paired(reference, arm):
    gains = sum(flag(reference[d]['confirmed']) and not flag(arm[d]['confirmed']) for d in reference)
    losses = sum(flag(arm[d]['confirmed']) and not flag(reference[d]['confirmed']) for d in reference)
    return {'gains': gains, 'losses': losses, 'exact_p': mcnemar(gains, losses)}


def family(table, reference, arms):
    tests = {a: paired(table[(reference, CAP)], table[(a, CAP)]) for a in arms}
    for a, p in zip(arms, holm([tests[a]['exact_p'] for a in arms])):
        tests[a]['holm_p'] = p
    return tests


def check_summary(name, summary, table):
    for row in summary['pooled']:
        mine = table.get((row['arm'], str(row['budget'])))
        if mine is None:
            problems.append(f'{name}: {row["arm"]}@{row["budget"]} missing from outcomes')
            continue
        stats = arm_stats(mine)
        for key in ('confirmed', 'accepted', 'calls', 'input_tokens', 'output_tokens'):
            if row.get(key) is not None and key in stats and stats[key] != row[key]:
                problems.append(f'{name}: {row["arm"]}@{row["budget"]} {key} {stats[key]} != summary {row[key]}')
    for row in summary.get('comparisons_at_cap8', []):
        mine = paired(table[(row['reference'], CAP)], table[(row['arm'], CAP)])
        if (mine['gains'], mine['losses']) != (row['reference_only'], row['arm_only']):
            problems.append(f'{name}: {row["reference"]} vs {row["arm"]} discordance differs')


def external(backend, arms):
    base = ROOT / 'results/external_baselines' / backend
    table = load(base / 'document_outcomes.csv')
    check_summary(backend, json.loads((base / 'summary.json').read_text(encoding='utf-8')), table)
    names = sorted({a for a, b in table if b == CAP})
    return {'arms_at_cap8': {a: arm_stats(table[(a, CAP)], seconds=True) for a in names},
            'P_family': family(table, 'P_producer_feedback', arms),
            'versus_D': {a: paired(table[(a, CAP)], table[('D_deterministic', CAP)]) for a in arms},
            'P_by_ceiling': {b: arm_stats(table[('P_producer_feedback', b)]) for b in ('1', '2', '4', '8')},
            'arms_by_ceiling': {a: {b: arm_stats(table[(a, b)])['confirmed'] for b in ('1', '2', '4', '8')} for a in arms}}


def generation_events(paths):
    events = [e for p in paths for e in json.loads(p.read_text(encoding='utf-8'))['events']]
    return {'sequences': len(events),
            'hit_token_limit': sum(bool(e['info'].get('hit_token_limit')) for e in events),
            'unparsed': sum(not e.get('parsed', e.get('operator') is not None) for e in events),
            'errors': sum(bool(e['info'].get('error')) for e in events)}


def self_refine_feedback():
    events = [e for p in sorted((ROOT / 'results/external_baselines/deepseek-flash').glob('fold*/self_refine.json'))
              for e in json.loads(p.read_text(encoding='utf-8'))['events'] if e['kind'] == 'feedback']
    return {'feedback_turns': len(events), 'declared_done': sum(bool(e['done']) for e in events),
            'ten_issues': sum(len(e.get('issues') or []) == 10 for e in events)}


def seeds():
    table = load(ROOT / 'results/seeds/document_outcomes.csv', key=('seed', 'arm', 'budget'))
    summary = json.loads((ROOT / 'results/seeds/summary.json').read_text(encoding='utf-8'))
    out, values = {}, defaultdict(list)
    for seed in sorted({s for s, _, _ in table}):
        arm = {a: table[(seed, a, CAP)] for a in ('F_first_pass', 'D_deterministic', 'producer_feedback', 'untuned_producer')}
        counts = {a: sum(flag(r['confirmed']) for r in rows.values()) for a, rows in arm.items()}
        out[seed] = {'confirmed': counts,
                     'P_vs_D': paired(arm['producer_feedback'], arm['D_deterministic']),
                     'P_vs_U': paired(arm['producer_feedback'], arm['untuned_producer']),
                     'D_vs_F': paired(arm['D_deterministic'], arm['F_first_pass']),
                     'P_by_ceiling': {b: sum(flag(r['confirmed']) for r in table[(seed, 'producer_feedback', b)].values())
                                      for b in ('1', '2', '4', '8')}}
        for a, c in counts.items():
            values[a].append(c)
            if summary['seeds'][seed]['confirmed_at_cap8'][a] != c:
                problems.append(f'seed {seed}: {a} {c} != summary')
    across = {a: {'values': v, 'mean': round(statistics.mean(v), 1), 'sd': round(statistics.stdev(v), 1)}
              for a, v in values.items()}
    return {'per_seed': out, 'across_seeds': across}


def chemu():
    table = load(ROOT / 'results/chemu/document_outcomes.csv')
    check_summary('chemu', json.loads((ROOT / 'results/chemu/summary.json').read_text(encoding='utf-8')), table)
    api = ['deepseek-flash:critic', 'deepseek-flash:clairify', 'deepseek-flash:routing']
    arms = {a: arm_stats(table[(a, CAP)]) for a in ['F_first_pass', 'D_deterministic', 'untuned_producer',
                                                     'producer_feedback', *api]}
    first, det = table[('F_first_pass', CAP)], table[('D_deterministic', CAP)]
    failing_first = [d for d, r in first.items() if flag(r['accepted']) and not flag(r['confirmed'])]
    failing_det = {d for d, r in det.items() if flag(r['accepted']) and not flag(r['confirmed'])}
    rescued = {a: sorted(d for d in failing_first if flag(table[(a, CAP)][d]['confirmed']))
               for a in ['D_deterministic', 'untuned_producer', 'producer_feedback', *api]}
    gains_from_failing = {a: sum(1 for d, r in table[(a, CAP)].items()
                                 if flag(r['confirmed']) and not flag(det[d]['confirmed']) and d in failing_det)
                          for a in ['untuned_producer', 'producer_feedback', *api]}
    gains_vs_D = {a: paired(table[(a, CAP)], det) for a in ['untuned_producer', 'producer_feedback', *api]}
    return {'arms_at_cap8': arms,
            'D_vs_F': paired(det, first),
            'P_family': family(table, 'producer_feedback', ['D_deterministic', 'untuned_producer', *api]),
            'versus_D': gains_vs_D,
            'rejected_after_D': sum(not flag(r['accepted']) for r in det.values()),
            'accepted_failing_after_D': len(failing_det),
            'gains_over_D_from_accepted_failing': gains_from_failing,
            'first_pass_accepted_failing': len(failing_first),
            'first_pass_accepted_failing_confirmed_by_arm': {a: len(v) for a, v in rescued.items()}}


def main():
    report = {'endpoint': 'Full verifier acceptance AND annotation-chain state replay',
              'external_api_deepseek_flash': external('deepseek-flash', ['clairify', 'critic', 'routing', 'self_refine']),
              'external_same_backbone_qwen14_base': external('qwen14-base', ['critic', 'clairify']),
              'self_refine_feedback': self_refine_feedback(),
              'external_generation': {
                  backend: generation_events(sorted((ROOT / 'results/external_baselines' / backend).glob('fold*/*.json')))
                  for backend in ('deepseek-flash', 'qwen14-base')} | {
                  'chemu_api': generation_events(chemu_raw) if (chemu_raw := sorted(
                      (ROOT / 'results/chemu/external').glob('*/*.json'))) else 'not distributed (licensed text)'},
              'seeds': seeds(),
              'chemu': chemu(),
              'problems': problems}
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / 'extensions_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'problems': problems}, indent=2))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
