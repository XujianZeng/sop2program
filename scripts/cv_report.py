"""Pool 5-fold cross-validation repairs under annotated identity.

Each document is tested in exactly one fold, so per-document outcomes from the five folds
form one paired sample of 276 documents per compiler. Rechecks run per fold (the fold's
data and knowledge are bound at import), then outcomes are pooled for exact McNemar tests.
"""
import json,os,subprocess,sys
from math import comb
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMPILERS=['zeroshot','fewshot3','lora3b','qwen14']
ARMS={'deterministic':'deterministic_repair','trained':'cascade','untuned':'producer_feedback'}


def mcnemar(b,c):
    n=b+c
    return min(1.,2*sum(comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.


def recheck(k,compiler,kind):
    repair=ROOT/f'results/cv/fold{k}/repair_{kind}_{compiler}'
    out=ROOT/f'results/cv/fold{k}/recheck_{kind}_{compiler}.json'
    if not (repair/'summary.json').exists():return None
    if not out.exists():
        env=dict(os.environ,PYTHONUTF8='1',SOP_DATA=f'data/cv/fold{k}',SOP_KNOWLEDGE=f'results/cv/fold{k}/knowledge')
        subprocess.run([sys.executable,'-X','utf8','scripts/recheck_annotation_identity.py','--arm',ARMS[kind],
                        '--repairs',repair.relative_to(ROOT).as_posix(),'--output',out.relative_to(ROOT).as_posix()],
                       cwd=ROOT,env=env,check=True,capture_output=True)
    return json.loads(out.read_text(encoding='utf-8'))[0]


def main():
    report={}
    for compiler in COMPILERS:
        pooled={kind:{} for kind in ARMS};text={kind:0 for kind in ARMS};folds={kind:0 for kind in ARMS}
        for k in range(5):
            for kind in ARMS:
                r=recheck(k,compiler,kind)
                if r is None:continue
                pooled[kind].update(r['chain_identity_by_document']);text[kind]+=r['accepted_text_identity'];folds[kind]+=1
        if not pooled['deterministic']:continue
        docs=pooled['deterministic']
        first={d:v[0] for d,v in docs.items()}
        final={kind:{d:v[1] for d,v in p.items()} for kind,p in pooled.items() if p}
        entry={'folds':folds,'documents':len(docs),'first_pass_chain':sum(first.values())}
        for kind,f in final.items():
            entry[kind]={'accepted_text':text[kind],'confirmed_chain':sum(f.values()),
                         'unconfirmed_share':1-sum(f.values())/text[kind] if text[kind] else None}
        def paired(a,b):
            common=[d for d in a if d in b]
            gain=sum(b[d] and not a[d] for d in common);loss=sum(a[d] and not b[d] for d in common)
            return {'documents':len(common),'gained':gain,'lost':loss,'exact_mcnemar_p':mcnemar(gain,loss)}
        entry['deterministic_vs_first']=paired(first,final['deterministic'])
        for kind in ('trained','untuned'):
            if kind in final:entry[f'{kind}_vs_deterministic']=paired(final['deterministic'],final[kind])
        report[compiler]=entry
        print(compiler,json.dumps(entry),flush=True)
    out=ROOT/'results/cv/report.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
