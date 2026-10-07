"""Mentions the training gold labels as a setting, mined from the training split only.

A site/usage argument that is neither in the initial inventory nor produced by any step
never occurs in the training gold; such a mention is relabelled as a setting only when
this corpus attests that reading, and is dropped only when the corpus never produces it
(a commonly produced object signals a missing producer instead).
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sop2program.ir import Workflow,norm,physical
from sop2program.paths import DATA,KNOWLEDGE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',default='train');p.add_argument('--min-count',type=int,default=2)
    p.add_argument('--output',default=str(KNOWLEDGE/'setting_lexicon.json'))
    args=p.parse_args()
    data=DATA/f'{args.split}_workflows.json'
    gold=[Workflow.model_validate(w) for w in json.loads(data.read_text(encoding='utf-8'))]
    counts=Counter(norm(a.text) for w in gold for s in w.steps for a in s.operator.arguments if a.role=='setting')
    settings=sorted(t for t,n in counts.items() if n>=args.min_count)
    made=Counter(norm(x) for w in gold for s in w.steps for x in s.operator.add)
    products=sorted(t for t,n in made.items() if n>=args.min_count)
    unavailable=sum(1 for w in gold for s in w.steps for a in s.operator.arguments
                    if physical(a) and a.role in ('site','usage') and norm(a.text) not in {norm(x) for x in w.initial}
                    and norm(a.text) not in {norm(x) for t in w.steps for x in t.operator.add})
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    payload={'split':args.split,'min_count':args.min_count,'source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
             'gold_unavailable_site_or_usage':unavailable,
             'scope':'Training-split setting and product mentions; no dev or test data used.',
             'settings':settings,'products':products}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{len(settings)} setting mentions, {len(products)} products; '
          f'gold site/usage arguments with no source object: {unavailable} -> {out}')

if __name__=='__main__':main()
