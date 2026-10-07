"""Predicted object identity for the verifier, without gold annotation.

State keys are entities rather than strings. An entity is named by the mention that
introduced it; a second distinct entity with the same name gets a " #n" suffix, so
most keys stay source strings and the repair operators keep working on them.
A mention denotes, in order of preference: the most recent alive entity with the
same name; an alive entity with the same head word; an alive entity whose head can
turn into this head under a training-corpus coreference transition; otherwise a
new name, which is missing unless the step produces it.
"""
import json,re
from functools import lru_cache
from pathlib import Path
from .ir import norm

ROOT=Path(__file__).resolve().parents[1]


def head(text):
    words=[w.strip('.,;:()*') for w in norm(text).split()]
    words=[w for w in words if w]
    if not words:return ''
    w=words[-1]
    return w[:-1] if len(w)>3 and w.endswith('s') and not w.endswith('ss') else w


def surface(key):
    """The mention text behind an entity key."""
    return re.sub(r' #\d+$','',key)


def identity_resolver(workflow):
    """None for text identity; otherwise the resolver named by workflow.metadata['identity']."""
    mode=workflow.metadata.get('identity')
    if mode=='resolved':
        lexicon=workflow.metadata.get('identity_lexicon')
        return EntityResolver(workflow.initial,load_transitions(lexicon) if lexicon else None)
    if mode=='oracle':
        from .annotation import oracle_resolver
        return oracle_resolver(workflow)
    return None


@lru_cache(maxsize=4)
def load_transitions(path):
    payload=json.loads((ROOT/path).read_text(encoding='utf-8'))
    out={}
    for a,b,_ in payload['transitions']:out.setdefault(b,set()).add(a)
    return out


class EntityResolver:
    def __init__(self,initial,transitions=None):
        self.transitions=transitions or {}
        self.names={};self.order=[];self.last={}
        self.aliases={}
        for i,text in enumerate(initial):self._register(self._fresh(text),norm(text),-1-len(initial)+i)

    def _fresh(self,text):
        base=norm(text);key=base;k=2
        while key in self.names:key=f'{base} #{k}';k+=1
        self.names[key]=base;return key

    def _register(self,key,alias,step):
        self.aliases.setdefault(alias,[])
        if key in self.aliases[alias]:self.aliases[alias].remove(key)
        self.aliases[alias].append(key)
        if key not in self.order:self.order.append(key)
        self.last[key]=step

    def resolve(self,text,alive,index=None):
        """Entity key for an input mention; does not change the resolver."""
        n=norm(text)
        same=self.aliases.get(n,[])
        living=[k for k in same if k in alive]
        if living:return living[-1]
        h=head(n)
        recent=sorted((k for k in alive if k in self.last),key=lambda k:self.last[k])
        by_head=[k for k in recent if any(head(a)==h for a,ks in self.aliases.items() if k in ks)]
        if h and by_head:return by_head[-1]
        sources=self.transitions.get(h,set())
        by_transition=[k for k in recent if any(head(a) in sources for a,ks in self.aliases.items() if k in ks)]
        if by_transition:return by_transition[-1]
        return same[-1] if same else n

    def commit(self,step_index,inputs,products):
        """inputs: (text,key) pairs the step read; products: texts it adds. Returns product keys by name."""
        for text,key in inputs:
            if key in self.names:self._register(key,norm(text),step_index)
        keys={}
        for text in products:
            if norm(text) in keys:continue
            keys[norm(text)]=self._fresh(text);self._register(keys[norm(text)],norm(text),step_index)
        return keys
