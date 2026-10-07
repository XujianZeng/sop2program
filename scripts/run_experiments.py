"""Reproducible local pipeline. Stops on failed stages; resumes saved work safely."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir',default='results/lora3b_full')
    parser.add_argument('--evaluation-dir',default='results/evaluation')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--max-docs',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=8,help='Inference batch size across SOPs')
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--methods',nargs='+',default=['base','lora','no_state','unconstrained'],
                        choices=['base','lora','no_state','unconstrained'])
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    def run(script,*options):
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,options)]
        print('RUN', ' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    manifest=ROOT/'models/Qwen2.5-3B-Instruct/download_manifest.json'
    if not manifest.exists() or not json.loads(manifest.read_text()).get('verified'):
        raise RuntimeError('Run scripts/download_model.py and verify the model first')
    run('audit_data.py')
    run('symbolic_experiments.py')
    training=ROOT/args.training_dir
    config=json.loads((training/'config.json').read_text()) if (training/'config.json').exists() else {}
    if config.get('status')!='complete':
        if args.skip_training:raise RuntimeError('Training has not completed')
        options=['--output',args.training_dir,'--epochs',args.epochs]
        if (training/'checkpoint.pt').exists():options+=['--resume']
        run('train.py',*options)
    paths=[]
    for method in args.methods:
        output=str(Path(args.evaluation_dir)/method)
        run('evaluate.py','--method',method,'--adapter',str(Path(args.training_dir)/'adapter'),
            '--output',output,'--max-docs',args.max_docs,'--batch-size',args.batch_size)
        run('score.py','--runs',output)
        paths.append(output)
    run('report.py','--runs',*paths,'--training-dir',args.training_dir)
    print('PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':main()
