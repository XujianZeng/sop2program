"""Resumable 5-fold cross-validation queue; every GPU step runs alone, in order.

Phase A completes first-pass few-shot outputs for the documents not yet covered.
Phase B, per fold: train the 14B compiler, the 3B compiler and the 3B feedback repairer
on the fold's training part with the original settings, evaluate both compilers on the
fold's test part, then repair every compiler's output (model-free, trained cascade,
untuned few-shot proposer). Phase C adds the zero-shot compiler. A step whose completion
marker exists is skipped, so the queue can be restarted after any interruption.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
LOGS=ROOT/'results/cv/logs'
FOLDS=range(5)
Q14='models/Qwen3-14B-FP8';Q3='models/Qwen2.5-3B-Instruct'
SHARED={'fewshot3':{'train':'results/cv/shared/fewshot3_train','dev':'results/cv/shared/fewshot3_dev',
                    'test':'results/evaluation_qwen14_fewshot3_test/base'},
        'zeroshot':{'train':'results/cv/shared/zeroshot_train','dev':'results/evaluation_qwen14_zeroshot_dev/base',
                    'test':'results/evaluation_qwen14_zeroshot_test/base'}}
EVAL=['--batch-size','32','--attention','sdpa','--pin-trigger']
TRAIN=['--epochs','1','--batch-size','1','--gradient-accumulation','4','--save-every','10']


def done(marker):
    path=ROOT/marker
    if not path.exists():return False
    if path.name in ('config.json','efficiency.json'):
        return json.loads(path.read_text(encoding='utf-8')).get('status')=='complete'
    return True


def step(name,args,marker,fold=None):
    if done(marker):return
    env=dict(os.environ,PYTHONUTF8='1')
    if fold is not None:env|={'SOP_DATA':f'data/cv/fold{fold}','SOP_KNOWLEDGE':f'results/cv/fold{fold}/knowledge'}
    LOGS.mkdir(parents=True,exist_ok=True)
    started=time.time();print(time.strftime('%H:%M:%S'),'START',name,flush=True)
    with (LOGS/f'{name}.log').open('a',encoding='utf-8') as log:
        code=subprocess.run([PY,'-X','utf8','-u',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    print(time.strftime('%H:%M:%S'),'END',name,'exit',code,f'{(time.time()-started)/60:.1f} min',flush=True)
    if code!=0 or not done(marker):raise SystemExit(f'{name} failed (exit {code}); see {LOGS/name}.log')


def train(name,output,data,fold,extra):
    resume=['--resume'] if (ROOT/output/'checkpoint.pt').exists() else []
    step(name,['scripts/train.py',*TRAIN,'--train-data',data,'--output',output,*extra,*resume],f'{output}/config.json',fold)


def fold_run(compiler,k):
    """Shared first-pass outputs restricted to fold k's test documents."""
    out=ROOT/f'results/cv/fold{k}/eval_{compiler}'
    if (out/'predictions.jsonl').exists():return out.relative_to(ROOT).as_posix()
    docs=set(json.loads((ROOT/f'data/cv/fold{k}/manifest.json').read_text())['documents']['test'])
    lines=[l for src in SHARED[compiler].values() for l in (ROOT/src/'predictions.jsonl').read_text(encoding='utf-8').splitlines()
           if json.loads(l)['id'].split(':')[0] in docs]
    rows=sum(1 for _ in open(ROOT/f'data/cv/fold{k}/test.jsonl',encoding='utf-8'))
    if len(lines)!=rows:raise SystemExit(f'{compiler} fold {k}: {len(lines)} predictions for {rows} rows')
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'run_config.json').write_text(json.dumps({'split':'test','sources':SHARED[compiler]},indent=2),encoding='utf-8')
    return out.relative_to(ROOT).as_posix()


def repairs(k,compiler,run,untuned):
    base=f'results/cv/fold{k}'
    step(f'f{k}_deterministic_{compiler}',['scripts/deterministic_repair.py','--run',run,'--split','test',
         '--output',f'{base}/repair_deterministic_{compiler}'],f'{base}/repair_deterministic_{compiler}/summary.json',k)
    common=['scripts/producer_repair.py','--split','test','--model',Q14,'--run',run,'--pin-trigger',
            '--no-retype-producers','--skip-node-feedback']
    step(f'f{k}_trained_{compiler}',[*common,'--adapter',f'{base}/qwen14/adapter','--fallback-model',Q3,
         '--fallback-adapter',f'{base}/lora3b_repair/adapter','--output',f'{base}/repair_trained_{compiler}'],
         f'{base}/repair_trained_{compiler}/summary.json',k)
    if untuned:
        shots=['--few-shot','3'] if compiler=='fewshot3' else []
        step(f'f{k}_untuned_{compiler}',[*common,'--adapter','none',*shots,'--output',f'{base}/repair_untuned_{compiler}'],
             f'{base}/repair_untuned_{compiler}/summary.json',k)


def shared_eval(compiler,split):
    shots=['--few-shot','3'] if compiler=='fewshot3' else []
    out=SHARED[compiler][split]
    step(f'shared_{compiler}_{split}',['scripts/evaluate.py','--method','base','--model',Q14,'--split',split,*shots,*EVAL,
         '--output',out],f'{out}/efficiency.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases',default='ABC')
    args=p.parse_args()
    if 'A' in args.phases:
        for split in ('train','dev'):shared_eval('fewshot3',split)
    if 'B' in args.phases:
        for k in FOLDS:
            base=f'results/cv/fold{k}';data=f'data/cv/fold{k}'
            train(f'f{k}_train_qwen14',f'{base}/qwen14',f'{data}/train_repair_fb.jsonl',k,['--model',Q14,'--empty-cache-every','1'])
            train(f'f{k}_train_lora3b',f'{base}/lora3b',f'{data}/train_repair_fb.jsonl',k,[])
            train(f'f{k}_train_lora3b_repair',f'{base}/lora3b_repair',f'{data}/train_feedback_only.jsonl',k,
                  ['--init-adapter',f'{base}/lora3b','--learning-rate','1e-4'])
            for name,model,adapter in (('qwen14',Q14,f'{base}/qwen14/adapter'),('lora3b',Q3,f'{base}/lora3b/adapter')):
                out=f'{base}/eval_{name}'
                step(f'f{k}_eval_{name}',['scripts/evaluate.py','--method','lora','--model',model,'--adapter',adapter,
                     '--split','test',*EVAL,'--output',out],f'{out}/efficiency.json',k)
            repairs(k,'qwen14',f'{base}/eval_qwen14',False)
            repairs(k,'lora3b',f'{base}/eval_lora3b',False)
            repairs(k,'fewshot3',fold_run('fewshot3',k),True)
    if 'C' in args.phases:
        shared_eval('zeroshot','train')
        for k in FOLDS:repairs(k,'zeroshot',fold_run('zeroshot',k),True)
    print('CV_ALL_DONE',flush=True)


if __name__=='__main__':main()
