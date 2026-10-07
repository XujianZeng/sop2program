"""Budget curves for every prespecified arm, from the completed CPU audit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
report = json.loads((P / 'evidence/controlled_audit.json').read_text(encoding='utf-8'))
extension = json.loads((P / 'evidence/violation_omission_audit.json').read_text(encoding='utf-8'))
report['pooled'] += [dict(r, arm='producer_candidate_only') for r in extension['pooled']]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
arms = [('producer_feedback', 'Trained: producer feedback', '#315c46', 'o'),
        ('node_feedback', 'Trained: node feedback', '#557f9a', 's'),
        ('producer_no_feedback', 'Trained: feedback block omitted', '#c2783e', '^'),
        ('untuned_producer', 'Untuned: producer feedback', '#7d638c', 'D'),
        ('producer_candidate_only', 'Follow-up C: violation field omitted', '#ad4950', 'x')]
fig, ax = plt.subplots(figsize=(4.8, 3.7))
for arm, label, color, marker in arms:
    values = sorted([r for r in report['pooled'] if r['arm'] == arm], key=lambda r: r['budget'])
    ax.plot(range(4), [r['confirmed'] for r in values], marker=marker, label=f'{label} ({values[-1]["confirmed"]} at 8)',
            color=color, lw=1.4, ms=4.2)
ax.axhline(142, color='#697278', ls=':', lw=1)
ax.text(.05, 140, 'Deterministic repair: 142', fontsize=7, va='top', color='#4d5559')
ax.set(xticks=range(4), xticklabels=['1', '2', '4', '8'], xlim=(-.12, 3.12), ylim=(135, max(r['confirmed'] for r in report['pooled'])+4),
       xlabel='Maximum generated sequences per document', ylabel='Jointly confirmed programs (of 276)')
ax.yaxis.grid(True, color='#e9edef')
ax.set_axisbelow(True)
fig.legend(loc='upper center', bbox_to_anchor=(.52, 1.02), ncol=1, frameon=False,
           fontsize=7.5, handlelength=2, labelspacing=.35)
fig.tight_layout(rect=(0, 0, 1, .76))
for ext in ('pdf', 'eps', 'png'):
    fig.savefig(P / f'figures/Fig2.{ext}', bbox_inches='tight', dpi=400)
plt.close(fig)
print('Saved main Fig2 PDF/EPS/PNG from all 20 audited endpoints; C is a later follow-up arm.')
