from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

ACTIONS = ['CREATE','TRANSFER','DESTROY','CONVERT','TEMP_TREAT','SPIN','MEASURE','WASH','SEAL','REMOVE','WAIT','MIX','OTHER']
TYPES = ['material','container','device','seal','setting','measurement','modifier','method','unknown']
ROLES = ['a','b','c','site','setting','usage']

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Argument(Strict):
    role: Literal['a','b','c','site','setting','usage']
    text: str
    type: Literal['material','container','device','seal','setting','measurement','modifier','method','unknown']

class Change(Strict):
    object: str
    attribute: str
    value: str

class Operator(Strict):
    action: Literal['CREATE','TRANSFER','DESTROY','CONVERT','TEMP_TREAT','SPIN','MEASURE','WASH','SEAL','REMOVE','WAIT','MIX','OTHER']
    trigger: str
    arguments: list[Argument] = Field(max_length=12)
    pre: list[str] = Field(max_length=12)
    add: list[str] = Field(max_length=12)
    delete: list[str] = Field(max_length=12)
    modify: list[Change] = Field(max_length=12)

class Evidence(Strict):
    start: int
    end: int
    text: str
    provenance: str = 'source'

class Step(Strict):
    id: str
    operator: Operator
    evidence: dict[str, Evidence]
    depends_on: list[str] = []

class Workflow(Strict):
    id: str
    source: str
    initial: list[str]
    steps: list[Step]
    metadata: dict = {}

def norm(s):
    return ' '.join(s.casefold().split())

def triggers_match(predicted,gold):
    """The marked action is supplied text. A trailing newline is still that span."""
    return bool(norm(predicted)) and norm(predicted)==norm(gold)

def physical(arg):
    return arg.type in {'material','container','device','seal'}

def ground(text,source,near=0):
    import re
    # Match the whitespace the verifier already ignores: norm() collapses runs and
    # treats a non-breaking space as a space, so an exact search here would leave a
    # perfectly attested argument unresolved.
    pattern=r'\s+'.join(map(re.escape,text.split()))
    matches=list(re.finditer(pattern,source,re.I)) if text.strip() else []
    if not matches: return Evidence(start=-1,end=-1,text=text,provenance='unresolved')
    m=min(matches,key=lambda m:abs(m.start()-near))
    return Evidence(start=m.start(),end=m.end(),text=source[m.start():m.end()])

def make_step(op,source,index,near=0,history=None):
    evidence={'trigger':ground(op.trigger,source,near)}
    evidence.update({f'arguments.{j}':ground(a.text,source,near) for j,a in enumerate(op.arguments)})
    # State statements are declared template/model inferences, never gold source quotes.
    step=Step(id=f's{index}',operator=op,evidence=evidence)
    for old in history or []:
        if set(map(norm,old.operator.add)) & set(map(norm,op.pre)):
            step.depends_on.append(old.id)
    return step

def required_inventory(steps,supplied=()):
    """S0: materials the protocol consumes but never produces, in first-use order.

    Mirrors the verifier step by step, so an object is stocked only the first time it
    is required and has never existed. A reference to an already destroyed object
    stays a lifecycle violation instead of being silently restocked.
    """
    initial=list(supplied);state=set(map(norm,initial));ever=set(state)
    for step in steps:
        op=step.operator
        added=set(map(norm,op.add))
        required=set(map(norm,op.pre))|set(map(norm,semantic_template(op.action,op.arguments)['pre']))
        required|={norm(m.object) for m in op.modify}-added
        fresh=sorted((required-state)-ever)
        initial+=fresh;state|=set(fresh);ever|=set(fresh)
        state-=set(map(norm,op.delete));state|=added;ever|=added
    return initial

def semantic_template(action,args):
    """Conservative weak labels; no unconditional pellet creation or rpm/g conversion."""
    phys=[a for a in args if physical(a)]
    inputs=[a.text for a in phys if a.role not in ('b','c') or action not in ('SPIN','CONVERT','CREATE')]
    products=[a.text for a in phys if a.role in ('b','c') and action in ('SPIN','CONVERT')]
    pre=sorted(set(inputs)) if action!='CREATE' else []
    add=sorted(set(products + ([a.text for a in phys] if action=='CREATE' else [])))
    delete=[a.text for a in phys if a.role=='a'] if action=='DESTROY' else []
    mods=[]
    target=next((a.text for a in phys if a.role=='a'),'')
    if target and action in ('MIX','TEMP_TREAT','WASH','SPIN','SEAL'):
        mods=[Change(object=target,attribute='last_action',value=action)]
    dest=next((a.text for a in phys if a.role=='site'),'')
    if action=='TRANSFER' and target and dest:
        mods=[Change(object=target,attribute='location',value=dest)]
    return dict(pre=pre,add=add,delete=delete,modify=mods)
