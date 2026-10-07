"""Document-level 5-fold cross-validation splits and per-fold rule libraries.

Identical source documents stay in one fold. The three documents that supply the fixed
few-shot demonstrations are pinned to every fold's training part, so every compiler is
scored on the same 276 documents. For fold k the test part is fold k, the dev part is half
of fold k+1 (rule confirmation only), and everything else trains. Hyperparameters are the
ones chosen on the original dev split; nothing is retuned per fold.
"""
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.induce import mine
from sop2program.ir import Workflow
from sop2program.paths import CORPUS

PINNED={'xwlp_91','xwlp_216','xwlp_212'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='data/cv')
    p.add_argument('--knowledge',default='results/cv')
    args=p.parse_args()
    workflows={};rows=defaultdict(list)
    for split in ('train','dev','test'):
        for w in json.loads((CORPUS/f'{split}_workflows.json').read_text(encoding='utf-8')):workflows[w['id']]=w
        for line in (CORPUS/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);rows[row['doc']].append(line)
    groups=defaultdict(list)
    for doc,w in workflows.items():groups[w['metadata']['source_sha256']].append(doc)
    pinned={g for g,docs in groups.items() if PINNED&set(docs)}
    order=sorted(set(groups)-pinned);random.Random(args.seed).shuffle(order)
    folds=[order[k::args.folds] for k in range(args.folds)]
    summary={'seed':args.seed,'pinned_training_documents':sorted(PINNED),'folds':[]}
    for k in range(args.folds):
        nxt=folds[(k+1)%args.folds]
        parts={'test':folds[k],'dev':nxt[:len(nxt)//2]}
        held={g for gs in parts.values() for g in gs}
        parts['train']=[g for g in groups if g not in held]
        data=ROOT/args.output/f'fold{k}';data.mkdir(parents=True,exist_ok=True)
        manifest={'fold':k,'documents':{},'counts':{}}
        for split,gs in parts.items():
            docs=sorted(d for g in gs for d in groups[g])
            manifest['documents'][split]=docs
            (data/f'{split}.jsonl').write_text('\n'.join(l for d in docs for l in rows[d])+'\n',encoding='utf-8')
            (data/f'{split}_workflows.json').write_text(json.dumps([workflows[d] for d in docs],ensure_ascii=False),encoding='utf-8')
            manifest['counts'][split]={'documents':len(docs),'operators':sum(len(rows[d]) for d in docs)}
        (data/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        load=lambda split:[Workflow.model_validate(workflows[d]) for d in manifest['documents'][split]]
        rules,audit=mine(load('train'),holdout=load('dev'))
        knowledge=ROOT/args.knowledge/f'fold{k}'/'knowledge';knowledge.mkdir(parents=True,exist_ok=True)
        (knowledge/'full_rules.json').write_text(json.dumps(rules,indent=2,ensure_ascii=False),encoding='utf-8')
        (knowledge/'full_candidates.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
        families={f:sum(r['family']==f for r in rules) for f in sorted({r['family'] for r in rules})}
        summary['folds'].append({**manifest['counts'],'rules':len(rules),'families':families})
        print(k,manifest['counts'],len(rules),families,flush=True)
    (ROOT/args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
