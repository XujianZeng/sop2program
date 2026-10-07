import json
from .ir import Operator,ACTIONS,TYPES

SYSTEM='''Compile one marked SOP action into a typed state transition. Return only a JSON object with keys action,trigger,arguments,pre,add,delete,modify.
action is one of CREATE TRANSFER DESTROY CONVERT TEMP_TREAT SPIN MEASURE WASH SEAL REMOVE WAIT MIX OTHER.
arguments are {role,text,type}; role is a,b,c,site,setting,usage. Types: material,container,device,seal,setting,measurement,modifier,method,unknown. Copy argument text verbatim from the SOP context. a is the operated input; b/c may be explicit products for SPIN/CONVERT; site is destination. Settings include time/temperature/speed. Do not invent objects.
pre/add/delete are object-name arrays. Require physical inputs to exist except CREATE. Explicit products of SPIN/CONVERT or CREATE go in add. DESTROY input goes in delete. modify is an array of {object,attribute,value}; MIX,TEMP_TREAT,WASH,SPIN,SEAL set last_action; TRANSFER sets location. No other implicit effects. An unknown action uses OTHER. Preserve unsupported preconditions; never silently recreate a discarded object.'''

def user_content(row,state=True):
    world=row.get('state',{}) if state else {'withheld':True}
    # Bounded context is explicit and identical in training and inference.
    if 'exists' in world:
        world={'exists':world['exists'][-32:],'attributes':dict(list(world.get('attributes',{}).items())[-12:])}
    content='SOP context:\n'+row['context']+'\nMarked action: '+row['trigger']+'\nCurrent state: '+json.dumps(world,ensure_ascii=False,separators=(',',':'))
    if row.get('repair_feedback') is not None:
        content+='\nCorrect only this operator using the source and verifier feedback below. Keep the marked trigger. Do not invent inventory or remove a supported action.\n'+json.dumps(row['repair_feedback'],ensure_ascii=False,separators=(',',':'))
    return content

def messages(row,state=True,demonstrations=()):
    turns=[{'role':'system','content':SYSTEM}]
    for demo in demonstrations:
        turns+=[{'role':'user','content':user_content(demo,state)},
                {'role':'assistant','content':json.dumps(demo['target'],ensure_ascii=False,separators=(',',':'))}]
    return turns+[{'role':'user','content':user_content(row,state)}]

def select_demonstrations(rows,k):
    """k short training rows of distinct actions, fixed by a stable order, for in-context baselines."""
    order={'SPIN':0,'TRANSFER':1,'CREATE':2,'DESTROY':3,'CONVERT':4,'WASH':5}
    picked={}
    for row in sorted(rows,key=lambda r:(len(r['context']),r['id'])):
        action=row['target']['action']
        if action in order and action not in picked and len(row['context'])>=200:picked[action]=row
    return [picked[a] for a in sorted(picked,key=order.get)][:k]

def prompt(tokenizer,row,state=True,demonstrations=()):
    # Qwen3 would otherwise open a reasoning block before the JSON; Qwen2.5 ignores the flag.
    return tokenizer.apply_chat_template(messages(row,state,demonstrations),tokenize=False,add_generation_prompt=True,enable_thinking=False)

def inference_batch_capacity(model,prompt_tokens):
    """Bound prefill storage without truncating the source or verifier feedback."""
    if getattr(model.config,'_attn_implementation','eager')=='sdpa':
        # Padded token budget keeps a 3.3k-token H4 prompt at batch 4.
        # Shorter test prompts may use a wider batch up to 32.
        return max(1,min(32,16_384//prompt_tokens))
    return max(1,16_000_000//prompt_tokens**2)

class LocalCompiler:
    def __init__(self,model,tokenizer,constrained=True,max_new_tokens=384,pin_trigger=False,demonstrations=()):
        self.model=model;self.tokenizer=tokenizer;self.constrained=constrained;self.max_new_tokens=max_new_tokens
        self.pin_trigger=pin_trigger;self.demonstrations=list(demonstrations)
        self.static_caches={}
        if constrained:
            from .constrained import tokenizer_data
            self.token_data=tokenizer_data(tokenizer)

    def compile(self,row,state=True):
        return self.compile_batch([row],state)[0]

    def compile_batch(self,rows,state=True):
        """Batch independent documents; the caller preserves order within each document."""
        import torch,time
        if not rows:return []
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        inputs=self.tokenizer([prompt(self.tokenizer,row,state,self.demonstrations) for row in rows],return_tensors='pt',padding=True).to(self.model.device)
        # Bound prefill workspace for long document feedback. This never
        # truncates a prompt and only splits independent generation requests.
        safe_batch=inference_batch_capacity(self.model,inputs.input_ids.shape[1])
        if len(rows)>safe_batch:
            return [output for start in range(0,len(rows),safe_batch)
                    for output in self.compile_batch(rows[start:start+safe_batch],state)]
        kwargs={};cache=None
        if getattr(self.model,'_sop_compile_cache',False) and inputs.input_ids.shape[1]+self.max_new_tokens<=4096:
            from transformers import CompileConfig
            from transformers.cache_utils import StaticCache
            cache=self.static_caches.get(len(rows))
            if cache is None:
                cache=StaticCache(config=self.model.config,max_cache_len=4096)
                self.static_caches[len(rows)]=cache
            kwargs={'past_key_values':cache,'compile_config':CompileConfig(mode='reduce-overhead',fullgraph=True,dynamic=False)}
        from .numerics import FiniteLogitsProcessor
        processors=[FiniteLogitsProcessor()]
        if self.constrained:
            from .constrained import BatchedSchemaLogitsProcessor
            processors.append(BatchedSchemaLogitsProcessor(self.token_data,Operator.model_json_schema()))
        cuda=self.model.device.type=='cuda'
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            if cache is not None:cache.reset()
            out=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.eos_token_id,logits_processor=processors,**kwargs)
        if cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        results=[]
        for i in range(len(rows)):
            tokens=out[i,inputs.input_ids.shape[1]:].tolist()
            stops=self.model.generation_config.eos_token_id
            stops=set(stops if isinstance(stops,list) else [stops])
            stop=next((j for j,t in enumerate(tokens) if t in stops),len(tokens))
            raw=self.tokenizer.decode(tokens[:stop],skip_special_tokens=True)
            try:operator=Operator.model_validate_json(raw);error=None
            except Exception as e:operator=None;error=str(e)
            pinned={}
            if self.pin_trigger and operator is not None:
                # The marked span is an input, so the copy the model wrote is replaced, not trusted.
                pinned={'model_trigger':operator.trigger,'trigger_from_input':True}
                operator=operator.model_copy(update={'trigger':rows[i]['trigger']})
            results.append((operator,{**pinned,'raw':raw,'error':error,'latency_s':elapsed,'amortized_latency_s':elapsed/len(rows),'batch_size':len(rows),'input_tokens':int(inputs.attention_mask[i].sum()),'output_tokens':min(stop+1,len(tokens)),'hit_token_limit':stop==len(tokens) and len(tokens)>=self.max_new_tokens,'compiled_static_cache':cache is not None,'finite_logits_checked':True}))
        return results
