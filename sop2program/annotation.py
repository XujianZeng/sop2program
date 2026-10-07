"""Object identity from the X-WLP gold annotation instead of surface text.

X-WLP gives every object mention an entity id and links ids that denote the same
physical thing across steps with implicit co_ref_of events ("E. coli culture" ->
"cells", one "tube" -> the next "tube"). Joining ids over those events yields entity
chains. Renaming each physical mention to its chain lets the unchanged template
semantics and verifier run on annotated identity, so any difference from the
text-keyed labels is attributable to identity alone.
"""
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from .ir import Argument, Change, Operator, Workflow, make_step, norm, physical, required_inventory, semantic_template, ROLES

TYPE_MAP={'rg':'material','loc':'container','d':'device','sl':'seal','s':'setting','m':'measurement','mod':'modifier','mth':'method'}


def load_peg(path):
    raw=json.loads(path.read_text())
    nodes={x['id']:x['data'] for x in raw['nodes']}
    ops=sorted([n for n in nodes.values() if n['parent_type']=='op' and n.get('spans')],key=lambda n:n['step'])
    return raw,nodes,ops


def aligned_arguments(raw,nodes,op):
    """(Argument, node id) pairs in the order scripts/prepare_data.py emits arguments."""
    pairs=[]
    for role,key in op.get('inputs',{}).items():
        if key not in nodes:continue
        a=nodes[key];r=role if role in ROLES else ('setting' if 'setting' in role else 'usage')
        pairs.append((Argument(role=r,text=a['display_name'],type=TYPE_MAP.get(a['type'],'unknown')),key))
    for edge in raw['links']:
        if edge['target']!=op['name']:continue
        for v in edge.values():
            if not isinstance(v,dict) or v.get('slot') not in ('setting_of','usage_of'):continue
            a=nodes.get(v.get('name',''))
            if a:
                arg=Argument(role='setting' if v['slot']=='setting_of' else 'usage',text=a['display_name'],type=TYPE_MAP.get(a['type'],'unknown'))
                if arg not in [p[0] for p in pairs]:pairs.append((arg,v['name']))
    return pairs


def entity_chains(nodes):
    parent={k:k for k,n in nodes.items() if n['parent_type']=='obj'}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for n in nodes.values():
        if n['parent_type']=='im_ev' and n['type']=='co_ref_of':
            ids=[k for k in [*n.get('inputs',{}).values(),*n.get('outputs',[])] if k in parent]
            for k in ids[1:]:parent[find(k)]=find(ids[0])
    return {k:find(k) for k in parent}


@lru_cache(maxsize=1)
def _gold_workflows():
    root=Path(__file__).resolve().parents[1]/'data/processed'
    return {w['id']:Workflow.model_validate(w) for split in ('train','dev','test')
            for w in json.loads((root/f'{split}_workflows.json').read_text(encoding='utf-8'))}


@lru_cache(maxsize=512)
def gold_identity(doc):
    w=_gold_workflows()[doc]
    return GoldIdentity(w,Path(__file__).resolve().parents[1]/w.metadata['peg_file'])


class OracleResolver:
    """Annotated identity inside the verifier: an upper bound for any identity predictor."""
    def __init__(self,identity,initial):
        self.identity=identity;self.order=list(dict.fromkeys(initial))

    def resolve(self,text,alive,index=None):
        return self.identity.resolve(text,index,alive)

    def commit(self,index,inputs,products):
        return {norm(t):self.identity.resolve(t,index) for t in products}


def oracle_resolver(workflow):
    return OracleResolver(gold_identity(workflow.id),workflow.initial)


class GoldIdentity:
    """Chain names for one gold workflow and a resolver for predicted object strings."""

    def __init__(self,workflow,path):
        raw,nodes,ops=load_peg(path)
        if len(ops)!=len(workflow.steps):raise ValueError(f'{workflow.id}: op count differs from processed workflow')
        self.workflow=workflow;self.nodes=nodes;chain=entity_chains(nodes);self.resolved=Counter()
        self.mentions=[]
        first={}
        for i,(op,step) in enumerate(zip(ops,workflow.steps)):
            pairs=aligned_arguments(raw,nodes,op)
            if [p[0] for p in pairs]!=step.operator.arguments:raise ValueError(f'{workflow.id}:{step.id}: arguments not aligned')
            local=[chain.get(key) if physical(arg) else None for arg,key in pairs]
            self.mentions.append(local)
            for arg,c in zip(step.operator.arguments,local):
                if c is not None:first.setdefault(c,(i,norm(arg.text)))
        order=sorted(first,key=lambda c:first[c][0])
        self.name={c:f'{first[c][1]} #{k}' for k,c in enumerate(order)}
        self.first_step={c:first[c][0] for c in order}
        self.by_text=defaultdict(set)
        for step,local in zip(workflow.steps,self.mentions):
            for arg,c in zip(step.operator.arguments,local):
                if c is not None:self.by_text[norm(arg.text)].add(c)

    def resolve(self,text,index,alive=None):
        """The gold chain a string at step index denotes, or the string itself if none is attested.

        A string anchored to a gold mention of this step takes that mention's chain. Otherwise,
        given the chains alive in the predicted state, any alive candidate is preferred, so an
        ambiguous name is only counted as missing when no reading of it exists.
        """
        key=norm(text)
        if index<len(self.mentions):
            for arg,c in zip(self.workflow.steps[index].operator.arguments,self.mentions[index]):
                if c is not None and norm(arg.text)==key:
                    self.resolved['local']+=1;return self.name[c]
        if index<len(self.mentions):
            # A shortened or extended span of one of this step's mentions ("cell suspension" for
            # "single cell suspension") still names that mention.
            overlapping={c for arg,c in zip(self.workflow.steps[index].operator.arguments,self.mentions[index])
                         if c is not None and key and (key in norm(arg.text) or norm(arg.text) in key)}
            if len(overlapping)==1:
                self.resolved['local_span']+=1;return self.name[overlapping.pop()]
        candidates=self.by_text.get(key)
        if not candidates and index<len(self.mentions) and key:
            # A shortened or extended span of this step's own mention ("cell suspension").
            for arg,c in zip(self.workflow.steps[index].operator.arguments,self.mentions[index]):
                if c is not None and (key in norm(arg.text) or norm(arg.text) in key):
                    self.resolved['local_span']+=1;return self.name[c]
        if not candidates:
            self.resolved['unattested']+=1;return key
        # A chain first mentioned at this step is that step's own product, not an input it can name.
        earlier=[c for c in candidates if self.first_step[c]<index]
        available=[c for c in candidates if alive is not None and self.name[c] in alive]
        if available:pick=max(available,key=self.first_step.get)
        else:pick=max(earlier,key=self.first_step.get) if earlier else min(candidates,key=self.first_step.get)
        self.resolved['document']+=1
        return self.name[pick]

    def rename_operator(self,op,index,retemplate=False,alive=None):
        r=lambda t:self.resolve(t,index,alive)
        args=[Argument(role=a.role,text=r(a.text),type=a.type) if physical(a) else a for a in op.arguments]
        if retemplate:
            labels=semantic_template(op.action,args)
        else:
            labels=dict(pre=[r(t) for t in op.pre],add=[r(t) for t in op.add],delete=[r(t) for t in op.delete],
                        modify=[Change(object=r(m.object),attribute=m.attribute,
                                       value=r(m.value) if m.attribute=='location' else m.value) for m in op.modify])
        return Operator(action=op.action,trigger=op.trigger,arguments=args,**labels)

    def gold(self):
        """The gold workflow on chain identity, with templates and S0 recomputed on chains."""
        steps=[]
        for i,step in enumerate(self.workflow.steps):
            steps.append(make_step(self.rename_operator(step.operator,i,retemplate=True),self.workflow.source,i,
                                   step.evidence['trigger'].start,steps))
        return Workflow(id=self.workflow.id,source=self.workflow.source,initial=required_inventory(steps),steps=steps,
                        metadata=dict(self.workflow.metadata,state_labels='annotation_identity_v1'))

    def rename(self,predicted,initial,charitable=True):
        """A predicted workflow with every object string resolved to a gold chain where attested.

        charitable resolves unanchored names against the chains alive after the renamed prefix.
        """
        from .verify import verify
        self.resolved=Counter();steps=[]
        for step in predicted.steps:
            i=int(step.id[1:]);alive=None
            if charitable:
                prefix=Workflow(id=predicted.id,source=predicted.source,initial=initial,steps=steps)
                alive=set(verify(prefix,evidence_checks=False,invariants=False)['final_state']['exists'])
            steps.append(make_step(self.rename_operator(step.operator,i,alive=alive),predicted.source,i,
                                   step.evidence['trigger'].start,steps))
        metadata={k:v for k,v in predicted.metadata.items() if k not in ('identity','identity_lexicon')}
        return Workflow(id=predicted.id,source=predicted.source,initial=initial,steps=steps,metadata=metadata)
