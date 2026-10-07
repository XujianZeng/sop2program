"""Reference programs and annotated identity for ChEMU 2020 reaction snippets.

ChEMU 2020 event annotation (Elsevier limited data licence; research use, no
redistribution) marks REACTION_STEP and WORKUP triggers with ARG1 and ARGM arguments.
ChEMU-Ref marks, over the same snippets, Coreference (identity) and the bridging
relations Reaction-associated, Work-up and Transformed, whose anaphor is a new entity
made from its antecedents. Joining the two gives what X-WLP gives: anchors, arguments and
cross-step entity chains. The derivation is deterministic:

* Steps are the event triggers in text order; the supplied anchors are their spans.
* Mentions are ChEMU-Ref spans; an event argument takes the smallest mention containing
  it, or its own span. Coreference joins mentions into chains.
* A bridging anaphor is produced by the trigger that takes it as ARG1, else by the last
  trigger before it. Its antecedents are that step's inputs. An anaphor produced by the
  step that produced its antecedent is the same entity: no annotated action lies between.
* A mention without its own relation that precedes the next trigger of its sentence is
  that trigger's input ("The mixture was stirred"); title-line mentions are not.
* A step that produces an entity is CONVERT (CREATE without inputs); other actions come
  from a fixed trigger lexicon. Effects use the unchanged action templates.

Chain labels are written to a separate identity file so that no prompt can read them.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from .annotation import GoldIdentity
from .ir import Argument, Operator, Workflow, make_step, norm, physical, required_inventory, semantic_template

BRIDGING = ('Reaction-associated', 'Work-up', 'Transformed')
CHEMICAL = {'STARTING_MATERIAL', 'REAGENT_CATALYST', 'SOLVENT', 'OTHER_COMPOUND', 'REACTION_PRODUCT'}
SETTING = {'TIME': 'setting', 'TEMPERATURE': 'setting', 'YIELD_PERCENT': 'measurement', 'YIELD_OTHER': 'measurement'}
LEXICON = {
    'TRANSFER': 'add added adding addition charge charged charging pour poured pouring introduce introduced placed place put '
                'transferred transfer loaded taken take dropped added dropwise',
    'MIX': 'stir stirred stirring mix mixed mixing dissolve dissolved dissolving suspended suspend suspending dilute diluted '
           'diluting admixed shaken shake combined combining slurried triturated trituration',
    'TEMP_TREAT': 'heat heated heating cool cooled cooling warm warmed warming reflux refluxed refluxing raised lowering lowered '
                  'maintained maintain chilled chill frozen freeze',
    'WASH': 'wash washed washing rinsed rinse',
    'REMOVE': 'extract extracted extraction extracting filter filtered filtration filtering concentrate concentrated '
              'concentration evaporate evaporated evaporation remove removed removal separated separate separation '
              'partitioned partition isolated isolate dried drying dry dehydrated decanted collected collect degassed',
    'WAIT': 'stand standing stood left allowed kept',
    'MEASURE': 'tlc monitored analyzed analysed measured',
    'CONVERT': 'quench quenched quenching purify purified purification purifying acidified acidify neutralized neutralised '
               'basified treated treat treating react reacted reacting reaction recrystallized recrystallised '
               'recrystallization chromatography chromatographed eluted eluting elute subjected hydrogenated adjusted',
    'CREATE': 'give gave given giving obtain obtained obtaining afford afforded affording yield yielded yielding provide '
              'provided providing prepare prepared preparing synthesized synthesised produce produced deliver delivered '
              'get got generate generated furnish furnished',
}
ACTION_OF = {w: a for a, words in LEXICON.items() for w in words.split()}
DETERMINER = re.compile(r'^(the|this|these|that|those|a|an|said)\s+', re.I)


def read_ann(path):
    entities, relations = {}, []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('T'):
            tid, info, text = line.split('\t')
            parts = info.split(' ')
            entities[tid] = {'label': parts[0], 'start': int(parts[1]), 'end': int(parts[-1]), 'text': text}
        elif line.startswith('R'):
            kind, a1, a2 = line.split('\t')[1].split(' ')
            relations.append((kind, a1.split(':', 1)[1], a2.split(':', 1)[1]))
    return entities, relations


def token_spans(text, sentences):
    spans, pos = [], 0
    for token in (t for s in sentences for t in s):
        i = text.find(token, pos)
        assert i >= 0, f'unaligned token {token!r}'
        spans.append((i, i + len(token)))
        pos = i + len(token)
    return spans


def sentence_bounds(text, sentences):
    tok = token_spans(text, sentences)
    bounds, k = [], 0
    for s in sentences:
        bounds.append((tok[k][0], tok[k + len(s) - 1][1]))
        k += len(s)
    return bounds


class Mentions:
    """Character-span mentions with union-find identity."""
    def __init__(self):
        self.spans, self.parent = [], []

    def add(self, span):
        if span in self.spans:
            return self.spans.index(span)
        self.spans.append(span)
        self.parent.append(len(self.parent))
        return len(self.spans) - 1

    def find(self, m):
        while self.parent[m] != m:
            self.parent[m] = self.parent[self.parent[m]]
            m = self.parent[m]
        return m

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def containing(self, start, end):
        inside = [i for i, (a, b) in enumerate(self.spans) if a <= start and end <= b]
        return min(inside, key=lambda i: self.spans[i][1] - self.spans[i][0]) if inside else None


def derive(doc_id, text, entities, relations, ref):
    """(Workflow, per-step chain ids aligned with arguments, chain display names)."""
    tok = token_spans(text, ref['sentences'])
    bounds = sentence_bounds(text, ref['sentences'])
    sentence = lambda x: next(i for i, (a, b) in enumerate(bounds) if x < b or i == len(bounds) - 1)
    mentions = Mentions()
    char = lambda span: (tok[span[0]][0], tok[span[1]][1])
    links = []
    for kind in ('Coreference', *BRIDGING):
        for anaphors, antecedents in ref[kind]:
            if anaphors and antecedents:
                links.append((kind, [mentions.add(char(s)) for s in anaphors], [mentions.add(char(s)) for s in antecedents]))
    for kind, anaphors, antecedents in links:
        if kind == 'Coreference':
            for m in anaphors[1:] + antecedents:
                mentions.union(anaphors[0], m)
    triggers = sorted((t for t, e in entities.items() if e['label'] in ('REACTION_STEP', 'WORKUP')),
                      key=lambda t: entities[t]['start'])
    arg1, argm = defaultdict(list), defaultdict(list)
    for kind, t, e in relations:
        if t in entities and e in entities:
            (arg1 if kind == 'ARG1' else argm)[t].append(e)
    entity_mention = {}
    for e, ent in entities.items():
        if ent['label'] in CHEMICAL:
            m = mentions.containing(ent['start'], ent['end'])
            entity_mention[e] = m if m is not None else mentions.add((ent['start'], ent['end']))
    position = {t: i for i, t in enumerate(triggers)}
    taken_by = defaultdict(set)
    for t in triggers:
        for e in arg1[t]:
            if e in entity_mention:
                taken_by[entity_mention[e]].add(t)

    def producer(m):
        start = mentions.spans[m][0]
        own = [t for t in taken_by.get(m, ()) if entities[t]['start'] >= start or sentence(entities[t]['start']) == sentence(start)]
        if own:
            return min(own, key=position.get)
        before = [t for t in triggers if entities[t]['start'] < start]
        return before[-1] if before else None

    inputs, products, produced_by = defaultdict(list), defaultdict(list), {}
    for kind, anaphors, antecedents in sorted((l for l in links if l[0] in BRIDGING), key=lambda l: mentions.spans[l[1][0]][0]):
        x = anaphors[0]
        p = producer(x)
        if p is None:
            continue
        fresh = []
        for a in antecedents:
            if produced_by.get(mentions.find(a)) == p:
                mentions.union(x, a)
            else:
                fresh.append(a)
        chain = mentions.find(x)
        if chain in produced_by and produced_by[chain] != p:
            continue
        if fresh or chain not in produced_by:
            produced_by[chain] = p
            products[p].append(x)
            inputs[p] += fresh
    used = {m for ms in list(inputs.values()) + list(products.values()) for m in ms} | set(taken_by)
    # Example label and title name the product; they are not inputs of the procedure.
    header = text.rfind('\n', 0, entities[triggers[0]]['start']) if triggers else -1
    for m, (start, end) in enumerate(mentions.spans):
        if m in used or start < header:
            continue
        later = [t for t in triggers if entities[t]['start'] >= end and sentence(entities[t]['start']) == sentence(start)]
        if later:
            inputs[later[0]].append(m)

    def head(m):
        start, end = mentions.spans[m]
        inside = [e for e, mm in entity_mention.items() if mm == m]
        if inside:
            return entities[min(inside, key=lambda e: entities[e]['start'])]['text']
        return DETERMINER.sub('', text[start:end]).strip()

    steps, chains = [], []
    for i, t in enumerate(triggers):
        own = {mentions.find(m) for m in products[t]}
        ins = [m for m in inputs[t] + [entity_mention[e] for e in arg1[t] if e in entity_mention]
               if mentions.find(m) not in own]
        outs = products[t]
        args, local, seen = [], [], set()
        def push(role, m):
            key = (role, mentions.find(m))
            if key in seen or len(args) >= 12:
                return
            seen.add(key)
            args.append(Argument(role=role, text=head(m), type='material'))
            local.append(mentions.find(m))
        if outs and ins:
            action = 'CONVERT'
            for m in ins:
                push('a', m)
            for m in outs:
                push('b', m)
        elif outs:
            action = 'CREATE'
            for m in outs:
                push('a', m)
        else:
            action = ACTION_OF.get(entities[t]['text'].lower(), 'OTHER')
            if action == 'CREATE':
                action = 'OTHER'
            for m in ins:
                push('a', m)
        for e in argm[t]:
            if entities[e]['label'] in SETTING and len(args) < 12:
                args.append(Argument(role='setting', text=entities[e]['text'], type=SETTING[entities[e]['label']]))
                local.append(None)
        op = Operator(action=action, trigger=entities[t]['text'], arguments=args, **semantic_template(action, args))
        steps.append(make_step(op, text, i, entities[t]['start'], steps))
        chains.append(local)
    first = {}
    for i, local in enumerate(chains):
        for arg, c in zip(steps[i].operator.arguments, local):
            if c is not None:
                first.setdefault(c, (i, norm(arg.text)))
    names = {c: f'{first[c][1]} #{n}' for n, c in enumerate(sorted(first, key=lambda c: first[c][0]))}
    workflow = Workflow(id=doc_id, source=text, initial=required_inventory(steps), steps=steps,
                        metadata={'corpus': 'chemu2020+chemu_ref', 'state_labels': 'weak_templates_v1',
                                  'inventory': 'gold_argument_inventory', 'triggers': 'gold_mentions'})
    return workflow, chains, names


class ChemuIdentity(GoldIdentity):
    """GoldIdentity over ChEMU chains; resolution, renaming and gold() are inherited unchanged."""
    def __init__(self, workflow, identity):
        from collections import Counter
        self.workflow, self.resolved, self.nodes = workflow, Counter(), None
        self.mentions = identity['chains']
        assert len(self.mentions) == len(workflow.steps)
        self.name = {int(c): n for c, n in identity['names'].items()}
        first = {}
        for i, (step, local) in enumerate(zip(workflow.steps, self.mentions)):
            assert len(local) == len(step.operator.arguments)
            for arg, c in zip(step.operator.arguments, local):
                if c is not None:
                    assert physical(arg)
                    first.setdefault(c, i)
        self.first_step = first
        self.by_text = defaultdict(set)
        for step, local in zip(workflow.steps, self.mentions):
            for arg, c in zip(step.operator.arguments, local):
                if c is not None:
                    self.by_text[norm(arg.text)].add(c)
