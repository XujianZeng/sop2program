"""Head-word transitions attested by X-WLP coreference on the training split.

A co_ref_of event a -> b whose head words differ ("culture" -> "cells") is evidence
that a later mention with head(b) may continue an earlier object with head(a).
Only transitions seen in at least --min-count training documents are kept.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.identity import head


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--min-count',type=int,default=2)
    p.add_argument('--output',default='results/symbolic/identity_lexicon.json')
    args=p.parse_args()
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    train={d for d in manifest['documents']['train']}
    docs=Counter();events=0
    for path in sorted((ROOT/'data/raw/xwlp/data').glob('*/*.peg')):
        raw=json.loads(path.read_text())
        if 'xwlp_'+str(raw['graph']['proto_idx']) not in train:continue
        nodes={x['id']:x['data'] for x in raw['nodes']};seen=set()
        for n in nodes.values():
            if n['parent_type']!='im_ev' or n['type']!='co_ref_of':continue
            a,b=(nodes.get(n['inputs'].get(k)) for k in ('a','b'))
            if not a or not b:continue
            pair=(head(a['display_name']),head(b['display_name']))
            if pair[0]!=pair[1] and all(pair):seen.add(pair);events+=1
        docs.update(seen)
    kept=sorted([[a,b,c] for (a,b),c in docs.items() if c>=args.min_count],key=lambda x:(-x[2],x))
    out=ROOT/args.output
    out.write_text(json.dumps({'source':'train split co_ref_of events, head words differ','min_documents':args.min_count,
                               'events':events,'transitions':kept},indent=1,ensure_ascii=False),encoding='utf-8')
    print(len(kept),'transitions from',events,'events; top',kept[:25])


if __name__=='__main__':main()
