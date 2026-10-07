"""Resume train.py after GPU driver resets until the run completes.

Stops when training completes, when several consecutive attempts fail without the
checkpoint advancing, or when a STOP file appears in the output directory. The STOP
file is the only way to retire a supervised run whose launching shell restarts it:
killing the processes just triggers another relaunch.
"""
import argparse,json,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def position(out):
    checkpoint=out/'checkpoint.pt'
    if not checkpoint.exists():return None
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    return saved['epoch'],saved['next_index']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/lora3b_full');p.add_argument('--log',default='results/train_resume_auto.log')
    p.add_argument('--max-stalled',type=int,default=3);p.add_argument('--cooldown',type=int,default=60)
    p.add_argument('train_args',nargs=argparse.REMAINDER,help='extra arguments after --, passed to train.py')
    args=p.parse_args()
    extra=[a for a in args.train_args if a!='--']
    out=ROOT/args.output;log_path=ROOT/args.log;status_path=ROOT/'results/task_status.json'
    stalled=0;attempt=0
    while True:
        if (out/'STOP').exists():
            print(f'STOP file present in {args.output}; leaving the run retired',flush=True);return
        attempt+=1;before=position(out)
        command=[sys.executable,'-X','utf8',str(ROOT/'scripts/train.py'),'--output',args.output,*extra]
        if (out/'checkpoint.pt').exists():command.append('--resume')
        with log_path.open('a',encoding='utf-8') as log:
            log.write(f'\n=== ATTEMPT {attempt} {datetime.now().isoformat(timespec="seconds")} from {before} ===\n');log.flush()
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        after=position(out)
        config=json.loads((out/'config.json').read_text(encoding='utf-8')) if (out/'config.json').exists() else {}
        status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        status.update(last_saved_example=after[1] if after else None,last_attempt=attempt,last_exit_code=code,
                      last_attempt_end=datetime.now().isoformat(timespec='seconds'),training_output=args.output,
                      notes=f'Auto-resume log: {args.log}')
        if code==0 and config.get('status')=='complete':
            status.update(phase='training_complete',last_run_status='complete')
            status_path.write_text(json.dumps(status,indent=2),encoding='utf-8');print('TRAINING_COMPLETE',flush=True);return
        stalled=0 if after!=before else stalled+1
        status.update(phase='training_recovery',last_run_status=f'attempt {attempt} exited {code}; stalled={stalled}')
        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        print(f'attempt {attempt} exit {code}: {before} -> {after}, stalled={stalled}',flush=True)
        if stalled>=args.max_stalled:raise SystemExit(f'No checkpoint progress in {stalled} consecutive attempts')
        time.sleep(args.cooldown)

if __name__=='__main__':main()
