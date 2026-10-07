"""GPU LoRA SFT with auxiliary semantic losses; deterministic document split."""
import argparse,json,math,random,time,sys,os,hashlib,platform
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['TOKENIZERS_PARALLELISM']='false'
# Serialized launches attribute a fault to the op that raised it, but this machine's
# driver resets kept happening with it on, so it buys diagnosis rather than stability
# and costs roughly half the throughput. It must be decided before torch is imported,
# hence the argv scan; argparse still declares the flag so --help and config record it.
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','1' if '--cuda-launch-blocking' in sys.argv else '0')
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from sop2program.compiler import prompt
from sop2program.fp8 import bf16_adapters,is_fp8_checkpoint,load_fp8_model
from sop2program.ir import ACTIONS,TYPES,ROLES

def mention_tokens(text,offsets,needle,lo,hi,boundary):
    """Prompt token range of `needle`'s first occurrence inside [lo,hi); None if absent."""
    if not needle.strip():return None
    pos=text.find(needle,lo,hi)
    if pos<0:return None
    ids=[i for i,(s,e) in enumerate(offsets[:boundary]) if e>pos and s<pos+len(needle)]
    return (ids[0],ids[-1]) if ids else None

class SemanticHeads(nn.Module):
    """Eq. (5)-(9): per-entity type, biaffine role, state-conditioned pre/effect, evidence pointer."""
    def __init__(self,dim,width=96):
        super().__init__()
        self.action=nn.Linear(dim,len(ACTIONS))
        self.type_head=nn.Linear(dim,len(TYPES))
        self.rel_action=nn.Linear(dim,width);self.rel_object=nn.Linear(dim,width)
        self.biaffine=nn.Parameter(torch.empty(len(ROLES),width,width));nn.init.xavier_uniform_(self.biaffine)
        self.rel_pair=nn.Linear(2*width,len(ROLES))
        # z(S_i) is pooled from the serialized state span, so the pre/effect heads
        # are conditioned on the committed world rather than on the text alone.
        self.pre=nn.Linear(2*dim,1);self.effect=nn.Linear(2*dim,3)
        self.pointer=nn.Linear(dim,2)

    def loss(self,h,boundary,record,tokenizer,text,offsets):
        """Returns the weighted auxiliary loss, its component names, and the undetached
        component stack. The caller copies the stack to the host once per step rather
        than calling .item() per component, which would synchronize six times per example."""
        target=record['target'];hidden=h[0,:boundary].float()
        v=hidden[boundary-1]
        zero=v.new_zeros(())
        context_start=text.index('SOP context:\n')+len('SOP context:\n')
        context_end=text.index('\nMarked action: ',context_start)
        state_start=text.index('\nCurrent state: ',context_end)
        state_span=mention_tokens(text,offsets,text[state_start:state_start+16],state_start,len(text),boundary)
        z=hidden[state_span[0]:].mean(0) if state_span else v
        # Evidence pointer over prompt tokens only; the target JSON is never a pointer target.
        source_end=record['source'].find('\n',record['offset']+len(record['trigger']))
        if source_end<0:source_end=len(record['source'])
        shift=context_start-(source_end-len(record['context']))
        trigger=mention_tokens(text,offsets,record['trigger'],context_start+record['offset']+shift-2,context_end,boundary)
        spans=[trigger] if trigger else []
        # Locate every argument on the host first, so each head runs as one batched kernel
        # over all arguments and every integer label crosses the bus in a single copy.
        embeddings=[];types=[];roles=[];pre=[];effect=[]
        for argument in target['arguments']:
            span=mention_tokens(text,offsets,argument['text'],context_start,context_end,boundary)
            if span is None:continue
            spans.append(span)
            embeddings.append(hidden[span[0]:span[1]+1].mean(0))
            types.append(TYPES.index(argument['type']));roles.append(ROLES.index(argument['role']))
            pre.append(float(argument['text'] in target['pre']))
            effect+=[float(argument['text'] in target['add']),float(argument['text'] in target['delete']),
                     float(any(m['object']==argument['text'] for m in target['modify']))]
        located=len(embeddings);width=len(spans)
        index=torch.tensor([ACTIONS.index(target['action'])]+types+roles+[s for s,_ in spans]+[e for _,e in spans],
                           device=v.device)
        losses={'action':nn.functional.cross_entropy(self.action(v)[None],index[:1])}
        if located:
            e=torch.stack(embeddings);conditioned=torch.cat([e,z.expand(located,-1)],1)
            a=self.rel_action(v);o=self.rel_object(e)
            pair=torch.einsum('i,rij,nj->nr',a,self.biaffine,o)+self.rel_pair(torch.cat([a.expand(located,-1),o],1))
            targets=torch.tensor(pre+effect,device=v.device,dtype=e.dtype)
            losses['type']=nn.functional.cross_entropy(self.type_head(e),index[1:1+located])
            losses['relation']=nn.functional.cross_entropy(pair,index[1+located:1+2*located])
            losses['pre']=nn.functional.binary_cross_entropy_with_logits(self.pre(conditioned).squeeze(1),
                                                                         targets[:located])
            losses['effect']=nn.functional.binary_cross_entropy_with_logits(self.effect(conditioned),
                                                                            targets[located:].view(located,3))
        else:
            losses|={'type':zero,'relation':zero,'pre':zero,'effect':zero}
        if width:
            # Equivalent to averaging cross_entropy(pointer_logits,[start,end]) over spans,
            # but shares one log-softmax instead of recomputing it per span.
            logp=nn.functional.log_softmax(self.pointer(hidden).T,dim=-1)
            starts=index[1+2*located:1+2*located+width];ends=index[1+2*located+width:]
            losses['evidence']=-(logp[0,starts].sum()+logp[1,ends].sum())/(2*width)
        else:
            losses['evidence']=zero
        names=list(losses);values=torch.stack([losses[name] for name in names])
        return .1*values.sum(),names,values.detach(),{'located_arguments':located,
                                                      'gold_arguments':len(target['arguments'])}

def counterfactual(target):
    """Single-factor near-miss: flip one state or parameter fact, keep the rest intact."""
    import copy,re
    near=copy.deepcopy(target)
    if near['pre']:
        near['pre']=near['pre'][1:];return near,'drop_precondition'
    if near['add']:
        near['delete']=sorted(set(near['delete'])|{near['add'][0]});near['add']=near['add'][1:]
        return near,'add_to_delete'
    if near['modify']:
        near['modify']=near['modify'][1:];return near,'drop_modify'
    for argument in near['arguments']:
        match=re.search(r'\d+(?:\.\d+)?',argument['text'])
        if match:
            scaled=f"{float(match.group())*10:g}"
            argument['text']=argument['text'][:match.start()]+scaled+argument['text'][match.end():]
            return near,'scale_parameter'
    for argument in near['arguments']:
        alternatives=[t for t in TYPES if t!=argument['type']]
        argument['type']=alternatives[0];return near,'swap_type'
    return None,None

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=1);p.add_argument('--max-examples',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--no-state',action='store_true');p.add_argument('--output',default='results/lora3b')
    p.add_argument('--max-length',type=int,default=2048);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--resume',action='store_true');p.add_argument('--gradient-accumulation',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=2)
    # Defaults follow results/benchmark/summary.json on this card. SDPA runs but is 2.9x
    # slower than eager here, and the desktop already holds about 8 GB, so dropping
    # activation recomputation or widening the batch spills into host memory and crawls.
    p.add_argument('--attention-backend',choices=['math','auto','eager'],default='eager')
    p.add_argument('--no-gradient-checkpointing',dest='gradient_checkpointing',action='store_false',
                   help='keep activations instead of recomputing them; needs headroom this card lacks')
    p.add_argument('--cuda-launch-blocking',action='store_true',help='serialize kernel launches for fault attribution; halves throughput')
    p.add_argument('--empty-cache-every',type=int,default=0,help='release the caching allocator every Nth microbatch; 0 disables')
    p.add_argument('--cf-every',type=int,default=4,help='counterfactual ranking on every Nth example; 0 disables')
    p.add_argument('--cf-margin',type=float,default=.5);p.add_argument('--cf-weight',type=float,default=.1)
    p.add_argument('--train-data',default='data/processed/train.jsonl')
    p.add_argument('--init-adapter',help='Start from this trained run directory (adapter and semantic heads); optimizer starts fresh')
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--model',default='models/Qwen2.5-3B-Instruct',help='Local base model directory')
    args=p.parse_args()
    if min(args.epochs,args.max_length,args.save_every,args.gradient_accumulation,args.batch_size)<1:raise ValueError('Training counts must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this training script')
    set_seed(args.seed);torch.set_num_threads(6)
    if args.attention_backend=='math':
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'training.jsonl').exists() and not args.resume:raise FileExistsError('Use --resume or a new --output to preserve prior training')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/args.model,local_files_only=True)
    attention='eager' if args.attention_backend=='eager' else 'sdpa'
    fp8=is_fp8_checkpoint(ROOT/args.model)
    if fp8:model=load_fp8_model(ROOT/args.model,attention)
    else:model=AutoModelForCausalLM.from_pretrained(ROOT/args.model,dtype=torch.bfloat16,device_map='cuda',attn_implementation=attention,local_files_only=True)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    if fp8:bf16_adapters(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    model.config.use_cache=False
    heads=SemanticHeads(model.config.hidden_size).cuda()
    if args.init_adapter and not args.resume:
        from safetensors.torch import load_file
        start_from=ROOT/args.init_adapter
        set_peft_model_state_dict(model,load_file(start_from/'adapter/adapter_model.safetensors'))
        heads.load_state_dict(torch.load(start_from/'semantic_heads.pt',map_location='cuda'))
    trainable=[p for p in model.parameters() if p.requires_grad]+list(heads.parameters())
    opt=torch.optim.AdamW(trainable,lr=args.learning_rate)
    rows=[json.loads(l) for l in (ROOT/args.train_data).read_text(encoding='utf-8').splitlines()]
    random.Random(args.seed).shuffle(rows)
    if args.max_examples:rows=rows[:args.max_examples]
    data=[];skipped=[]
    for row in rows:
        text=prompt(tokenizer,row,not args.no_state)
        prefix=tokenizer(text)['input_ids'];target=tokenizer(json.dumps(row['target'],ensure_ascii=False,separators=(',',':'))+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        if len(prefix)+len(target)>args.max_length:skipped.append(row['id']);continue
        near,kind=counterfactual(row['target']) if args.cf_every else (None,None)
        near_ids=tokenizer(json.dumps(near,ensure_ascii=False,separators=(',',':'))+tokenizer.eos_token,add_special_tokens=False)['input_ids'] if near else None
        if near_ids and len(prefix)+len(near_ids)>args.max_length:near_ids=None
        offsets=tokenizer(text,return_offsets_mapping=True)['offset_mapping']
        data.append((row,text,prefix,target,offsets,near_ids,kind))
    if not data:raise ValueError('No training examples fit the length limit')
    fingerprint=hashlib.sha256((ROOT/args.train_data).read_bytes()).hexdigest()
    config=vars(args)|{'model':json.loads((ROOT/args.model/'download_manifest.json').read_text(encoding='utf-8'))['repo'],
                       'model_manifest_sha256':hashlib.sha256((ROOT/args.model/'download_manifest.json').read_bytes()).hexdigest(),'lora_rank':8,'lora_alpha':16,'precision':'bfloat16 compute and adapters',
                       'quantization':'frozen float8_e4m3fn base, 128x128 block scales' if fp8 else None,'learning_rate':args.learning_rate,
                         'init_adapter_sha256':hashlib.sha256((ROOT/args.init_adapter/'adapter/adapter_model.safetensors').read_bytes()).hexdigest() if args.init_adapter else None,'max_sequence_length':args.max_length,'aux_weight':.1,'examples':len(data),'skipped':skipped,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'label_quality':'gold PEG structure + template weak states; gold action mention and physical inventory inputs','seed':args.seed,'train_sha256':fingerprint,'status':'running','lm_head':'frozen; LoRA and auxiliary heads trained','cuda_launch_blocking':os.environ['CUDA_LAUNCH_BLOCKING'],'lm_loss_weighting':'mean over target tokens in a microbatch','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start_epoch=start_index=updates=0;previous_elapsed=0
    config['software_versions']={name:version(name) for name in ['torch','transformers','peft','accelerate']}
    config['runtime']={'platform':platform.platform(),'python':sys.version,'executable':sys.executable,'cuda':torch.version.cuda}
    checkpoint=out/'checkpoint.pt'
    if args.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        bad=[name for name,t in list(saved['adapter'].items())+list(saved['heads'].items()) if t.is_floating_point() and not torch.isfinite(t).all()]
        if bad:raise FloatingPointError(f'Checkpoint contains nonfinite tensors: {bad[:5]}')
        for key in ['train_sha256','seed','no_state','max_examples','max_length']:
            if saved['config'][key]!=config[key]:raise ValueError(f'Cannot resume with changed {key}')
        if saved['config']['batch_size']*saved['config']['gradient_accumulation']!=args.batch_size*args.gradient_accumulation:
            raise ValueError('Resume must preserve effective batch size')
        set_peft_model_state_dict(model,saved['adapter']);heads.load_state_dict(saved['heads']);opt.load_state_dict(saved['optimizer'])
        start_epoch=saved['epoch'];start_index=saved['next_index'];updates=saved['updates'];previous_elapsed=saved['elapsed_s']
        config['resume_history']=saved['config'].get('resume_history',[])
        if saved['config'].get('resumed_from'):config['resume_history'].append(saved['config']['resumed_from'])
        config['resumed_from']={'epoch':start_epoch,'next_index':start_index,'attention_backend':saved['config']['attention_backend'],'batch_size':saved['config']['batch_size'],'gradient_accumulation':saved['config']['gradient_accumulation'],'torch':saved['config']['torch']}
        config['prior_peak_vram_gb']=saved.get('peak_vram_gb',saved['config'].get('prior_peak_vram_gb',0))
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    (out/'config.json').write_text(json.dumps(config,indent=2));model.print_trainable_parameters()
    def target_loss(ids,mask,labels,prefixes):
        """Causal LM loss with vocabulary logits only from the last prompt token onward.

        Prompt labels are already -100, so this is the same mean over target tokens;
        it just never materializes float32 logits for a 2k-token prompt.
        """
        keep=ids.shape[1]-min(map(len,prefixes))+1
        return model(input_ids=ids,attention_mask=mask,labels=labels[:,-keep:],logits_to_keep=keep).loss
    def pack(prefixes,targets):
        """Pad on the host and ship one tensor per field; a copy per row stalls the queue."""
        lengths=[len(p)+len(t) for p,t in zip(prefixes,targets)]
        ids=torch.full((len(prefixes),max(lengths)),tokenizer.eos_token_id,dtype=torch.long)
        labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
        for j,(p,t) in enumerate(zip(prefixes,targets)):
            ids[j,:lengths[j]]=torch.tensor(p+t);mask[j,:lengths[j]]=1
            labels[j,len(p):lengths[j]]=ids[j,len(p):lengths[j]]
        return (ids.to('cuda',non_blocking=True),labels.to('cuda',non_blocking=True),
                mask.to('cuda',non_blocking=True))
    start=time.time();model.train();opt.zero_grad();torch.cuda.reset_peak_memory_stats()
    # Hook only final decoder normalization; avoids retaining every layer output.
    captured={}
    def hook(module,inputs,output):captured['h']=output
    handle=model.base_model.model.model.norm.register_forward_hook(hook)
    def save(epoch,next_index):
        record={'adapter':get_peft_model_state_dict(model),'heads':heads.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'next_index':next_index,'updates':updates,'config':config,'elapsed_s':previous_elapsed+time.time()-start,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'peak_vram_gb':max(config.get('prior_peak_vram_gb',0),torch.cuda.max_memory_allocated()/2**30)}
        temporary=out/'checkpoint.pt.tmp';torch.save(record,temporary);temporary.replace(checkpoint)
        model.save_pretrained(out/'adapter');torch.save(heads.state_dict(),out/'semantic_heads.pt')
        print('CHECKPOINT',epoch,next_index,updates,flush=True)
    # The tokenizer never changes during training, so it is written beside the adapter once
    # instead of being re-serialized at every checkpoint.
    tokenizer.save_pretrained(out/'adapter')
    with (out/'training.jsonl').open('a' if args.resume else 'w',encoding='utf-8') as log:
        for epoch in range(start_epoch,args.epochs):
            first=start_index if epoch==start_epoch else 0
            for i in range(first,len(data),args.batch_size):
                batch=data[i:i+args.batch_size];lengths=[len(entry[2])+len(entry[3]) for entry in batch]
                ids,labels,mask=pack([entry[2] for entry in batch],[entry[3] for entry in batch])
                lm=target_loss(ids,mask,labels,[entry[2] for entry in batch])
                auxiliaries=[heads.loss(captured['h'][j:j+1,:lengths[j]],len(entry[2]),entry[0],tokenizer,entry[1],entry[4])
                             for j,entry in enumerate(batch)]
                aux=torch.stack([a for a,names,values,counts in auxiliaries]).mean()
                batch_index=i//args.batch_size;num_batches=(len(data)+args.batch_size-1)//args.batch_size
                group_start=(batch_index//args.gradient_accumulation)*args.gradient_accumulation
                group_size=min(args.gradient_accumulation,num_batches-group_start)
                loss=lm+aux
                # One device-to-host copy carries the joint loss, the LM term and every
                # auxiliary component, so the finite check and the log share a single
                # synchronization instead of stalling the queue once per reported number.
                scalars=torch.cat([torch.stack([loss.detach(),lm.detach()]).float()]
                                  +[values.float() for a,names,values,counts in auxiliaries]).tolist()
                total,lm_loss=scalars[0],scalars[1]
                component_names=auxiliaries[0][1]
                parts={name:sum(scalars[2+j*len(component_names)+k] for j in range(len(batch)))/len(batch)
                       for k,name in enumerate(component_names)}
                parts|={key:sum(counts[key] for a,names,values,counts in auxiliaries)/len(batch)
                        for key in auxiliaries[0][3]}
                if not math.isfinite(total):
                    failure={'stage':'forward','ids':[entry[0]['id'] for entry in batch],
                             'example':i+len(batch),'updates':updates,'lengths':lengths,
                             'lm_loss':str(lm_loss),'aux':parts,'runtime':config['runtime']}
                    (out/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
                    raise FloatingPointError(f'Nonfinite loss before backward: {failure}')
                gold=lm.detach()
                (loss/group_size).backward()
                # L_cf: a single-factor near miss must score below the gold operator.
                # The gold term is detached and back-propagated separately so the two
                # graphs never occupy the GPU at the same time. The schedule counts
                # examples, not microbatches, so the same fraction of the corpus is
                # contrasted whatever --batch-size is set to.
                cf_batch=[entry for j,entry in enumerate(batch) if entry[5] and args.cf_every and (i+j)%args.cf_every==0]
                if cf_batch:
                    near_ids,near_labels,near_mask=pack([entry[2] for entry in cf_batch],[entry[5] for entry in cf_batch])
                    near_loss=target_loss(near_ids,near_mask,near_labels,[entry[2] for entry in cf_batch])
                    cf=torch.relu(args.cf_margin-(near_loss-gold))
                    cf_value=cf.item()
                    if math.isfinite(cf_value):
                        (args.cf_weight*cf/group_size).backward()
                        total+=args.cf_weight*cf_value
                        parts['counterfactual']=cf_value
                        parts['counterfactual_kinds']=sorted({entry[6] for entry in cf_batch})
                    del near_ids,near_labels,near_mask,near_loss,cf
                del cf_batch,gold
                if (batch_index+1)%args.gradient_accumulation==0 or i+len(batch)==len(data):
                    try:
                        grad_norm=nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
                    except RuntimeError:
                        names=list(model.named_parameters())+[(f'heads.{name}',p) for name,p in heads.named_parameters()]
                        failure={'stage':'backward','ids':[entry[0]['id'] for entry in batch],
                                 'example':i+len(batch),'updates':updates,'lengths':lengths,
                                 'nonfinite_gradients':[name for name,p in names if p.grad is not None and not torch.isfinite(p.grad).all()],
                                 'runtime':config['runtime']}
                        (out/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
                        raise
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    if updates%args.save_every==0:save(epoch,i+len(batch))
                rec={'epoch':epoch,'example':i+len(batch),'ids':[entry[0]['id'] for entry in batch],'updates':updates,'loss':total,'lm_loss':lm_loss,'aux':parts,'elapsed_s':previous_elapsed+time.time()-start}
                log.write(json.dumps(rec)+'\n');log.flush()
                if batch_index%10==0:print(json.dumps(rec),flush=True)
                del lm,loss,aux,ids,labels,mask,auxiliaries;captured.clear()
                # Returning blocks to the driver costs a full synchronization and forces the
                # next step to re-allocate, so it stays off unless a run needs the headroom.
                if args.empty_cache_every and (batch_index+1)%args.empty_cache_every==0:torch.cuda.empty_cache()
            save(epoch+1,0)
    handle.remove();config.update(elapsed_s=previous_elapsed+time.time()-start,peak_vram_gb=max(config.get('prior_peak_vram_gb',0),torch.cuda.max_memory_allocated()/2**30),status='complete',updates=updates)
    (out/'config.json').write_text(json.dumps(config,indent=2));print('TRAINING_COMPLETE',json.dumps(config),flush=True)

if __name__=='__main__':main()
