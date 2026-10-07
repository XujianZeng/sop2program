"""Trigger -> attested action inventory, mined from the training split only.

Algorithm C may correct an action type only to one this corpus evidence supports,
so the repairer cannot search the closed action inventory for a passing label.
"""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sop2program.ir import norm
from sop2program.paths import DATA,KNOWLEDGE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',default='train');p.add_argument('--min-count',type=int,default=2)
    p.add_argument('--min-share',type=float,default=.1)
    p.add_argument('--output',default=str(KNOWLEDGE/'trigger_lexicon.json'))
    args=p.parse_args()
    data=DATA/f'{args.split}.jsonl'
    counts=defaultdict(Counter)
    for line in data.read_text(encoding='utf-8').splitlines():
        row=json.loads(line);target=row['target']
        counts[norm(target['trigger'])][target['action']]+=1
    lexicon={}
    for trigger,actions in counts.items():
        total=sum(actions.values())
        kept=sorted(a for a,n in actions.items() if n>=args.min_count and n/total>=args.min_share)
        if len(kept)>1:lexicon[trigger]=kept
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    payload={'split':args.split,'min_count':args.min_count,'min_share':args.min_share,
             'source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
             'triggers':len(counts),'ambiguous_triggers':len(lexicon),
             'scope':'Training-split trigger/action co-occurrence; no test data used.',
             'lexicon':dict(sorted(lexicon.items()))}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{len(lexicon)} ambiguous triggers of {len(counts)} -> {out}')

if __name__=='__main__':main()
