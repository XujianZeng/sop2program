"""Run evaluate.py for each method, resuming after GPU driver resets, then score and report.

evaluate.py keeps finished predictions in predictions.jsonl; a retry continues from there.
Stops when several consecutive attempts add no predictions.
"""
import argparse,json,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def acquire_lock(directory):
    """The OS releases this lock on crashes; another launcher cannot share the cache."""
    directory.mkdir(parents=True,exist_ok=True)
    stream=(directory/'.supervisor.lock').open('a+b')
    if stream.seek(0,2)==0:stream.write(b' ');stream.flush()
    stream.seek(0)
    try:
        if sys.platform=='win32':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f'Another evaluation supervisor owns {directory}; refusing a duplicate writer')
    return stream

def count(path):
    return sum(1 for _ in path.open('rb')) if path.exists() else 0

def call(log,*command):
    with log.open('a',encoding='utf-8') as stream:
        stream.write(f'\n=== {datetime.now().isoformat(timespec="seconds")} {" ".join(map(str,command))} ===\n');stream.flush()
        return subprocess.run([sys.executable,'-X','utf8','-u',*map(str,command)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',default=['base','lora','no_state','unconstrained'])
    p.add_argument('--training-dir',default='results/lora3b_full');p.add_argument('--evaluation-dir',default='results/evaluation')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-docs',type=int,default=0)
    p.add_argument('--max-stalled',type=int,default=3);p.add_argument('--cooldown',type=int,default=60)
    p.add_argument('--skip-report',action='store_true',help='Leave report generation to the full completion pipeline')
    p.add_argument('--compile-cache',action='store_true')
    p.add_argument('--attention', choices=['eager','sdpa'], default='eager')
    args=p.parse_args()
    try:lock=acquire_lock(ROOT/args.evaluation_dir)
    except RuntimeError as error:raise SystemExit(str(error))
    try:run(args)
    finally:lock.close()

def run(args):
    log=ROOT/'results/evaluation_auto.log';paths=[]
    for method in args.methods:
        out=Path(args.evaluation_dir)/method;stalled=0
        while True:
            if (ROOT/args.evaluation_dir/'STOP').exists():
                print('Evaluation STOP file present; leaving the run retired',flush=True);return
            efficiency=ROOT/out/'efficiency.json';before=count(ROOT/out/'predictions.jsonl')
            efficiency.unlink(missing_ok=True)
            code=call(log,'scripts/evaluate.py','--method',method,'--adapter',Path(args.training_dir)/'adapter',
                      '--output',out,'--batch-size',args.batch_size,'--max-docs',args.max_docs,
                      '--attention',args.attention,
                      *(['--compile-cache'] if args.compile_cache else []))
            after=count(ROOT/out/'predictions.jsonl')
            if code==0 and efficiency.exists() and json.loads(efficiency.read_text())['status']=='complete':break
            stalled=0 if after>before else stalled+1
            print(f'{method}: exit {code}, predictions {before} -> {after}, stalled={stalled}',flush=True)
            if stalled>=args.max_stalled:raise SystemExit(f'{method}: no progress in {stalled} consecutive attempts')
            time.sleep(args.cooldown)
        if call(log,'scripts/score.py','--runs',out)!=0:raise SystemExit(f'{method}: scoring failed')
        print(f'{method}: complete',flush=True);paths.append(out)
    if not args.skip_report:
        if call(log,'scripts/report.py','--runs',*paths,'--training-dir',args.training_dir)!=0:raise SystemExit('report failed')
    print('EVALUATION_PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':main()
