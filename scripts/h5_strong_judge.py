"""H5 with a stronger text-only judge: untuned Qwen3-14B-FP8, zero-shot, thinking disabled.

Same near-miss cases, judge prompt, payload/token budget, parsing and paired tests as the
original H5 run; only the judge model changes. The verifier side is recomputed unchanged.
"""
import argparse,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer,set_seed
import scripts.hypothesis_tests as ht
from sop2program.fp8 import load_fp8_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/Qwen3-14B-FP8')
    p.add_argument('--rules',default='results/symbolic/full_rules.json')
    p.add_argument('--output',default='results/h5_qwen3_14b_judge.json')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    set_seed(42)
    rules=json.loads((ROOT/args.rules).read_text(encoding='utf-8'))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/args.model,local_files_only=True)
    original=tokenizer.apply_chat_template
    tokenizer.apply_chat_template=lambda *a,**k:original(*a,enable_thinking=False,**k)
    model=load_fp8_model(ROOT/args.model,'sdpa').eval()
    def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    report={'judge_model':args.model,'judge_adapter':None,'thinking':False,
            'model_manifest_sha256':digest(Path(args.model)/'download_manifest.json'),
            'rules_sha256':digest(args.rules),'test_workflows_sha256':digest('data/processed/test_workflows.json'),
            'hypothesis_code_sha256':digest('scripts/hypothesis_tests.py'),'script_sha256':digest('scripts/h5_strong_judge.py'),
            'judge_batch_size':4,'judge_max_new_tokens':48,'judge_max_input_tokens':3584,
            'judge_max_payload_characters':12000,'seed':42,'limit':args.limit}
    cache=ht.CaseCache(ROOT/(args.output+'.checkpoints'),report)
    report['H5']=ht.run_h5(model,tokenizer,rules,args.limit,cache)
    report['H5']['scope']=report['H5']['scope'].replace('the trained compiler','untuned Qwen3-14B-FP8 (thinking disabled)')
    report['status']='complete'
    ht.atomic_json(ROOT/args.output,report)
    h=report['H5']
    print('verifier',h['verifier'],'\njudge',{k:v for k,v in h['llm_judge'].items()},
          '\ndetection p',h['paired_detection_test']['p_value'],'localization p',h['paired_localization_test']['p_value'],flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
