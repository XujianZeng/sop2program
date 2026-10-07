"""Compile a UTF-8 SOP with user-supplied action spans and initial inventory."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sop2program.pipeline import compile_workflow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='UTF-8 SOP text')
    parser.add_argument('--anchors',required=True,help='JSON list of {start,end} source offsets, in execution order')
    parser.add_argument('--inventory',required=True,help='JSON list of initially available object names')
    parser.add_argument('--adapter',default='results/lora3b_full/adapter')
    parser.add_argument('--rules',default='results/symbolic/full_rules.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--llm-repair',action='store_true',help='Ask the model for up to four local patches; verify each externally')
    args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from sop2program.compiler import LocalCompiler
    torch.set_num_threads(6)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-3B-Instruct',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen2.5-3B-Instruct',dtype=torch.bfloat16,
                                             device_map='cuda',attn_implementation='eager',local_files_only=True)
    model=PeftModel.from_pretrained(model,ROOT/args.adapter).merge_and_unload().eval()
    model.config.use_cache=True
    result=compile_workflow((ROOT/args.input).read_text(encoding='utf-8'),
                            json.loads((ROOT/args.inventory).read_text(encoding='utf-8')),
                            json.loads((ROOT/args.anchors).read_text(encoding='utf-8')),
                            LocalCompiler(model,tokenizer),
                            rules=json.loads((ROOT/args.rules).read_text(encoding='utf-8')),
                            workflow_id=Path(args.input).stem,llm_repair=args.llm_repair)
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(result['status'],str(path))


if __name__=='__main__':main()
