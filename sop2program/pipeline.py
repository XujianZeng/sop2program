"""Public anchored SOP -> compile -> deterministic replay -> bounded repair API."""
from .ir import Workflow, make_step
from .verify import verify, minimal_repair


def compile_workflow(source, initial, anchors, compiler, *, rules=(), workflow_id='sop', repair=True, llm_repair=False):
    """Anchors are source offsets {start,end}; inventory is explicitly supplied.

    This API does not claim automatic action discovery or inventory extraction.
    Schema failures remain explicit REVIEW cases and cannot yield a false PASS.
    """
    workflow=Workflow(id=workflow_id,source=source,initial=initial,steps=[],
                      metadata={'anchor_source':'supplied','inventory_source':'supplied',
                                'state_semantics':'conservative weak_templates_v1'})
    outputs=[]
    invalid=[]
    seen=set()
    for index,anchor in enumerate(anchors):
        start,end=anchor['start'],anchor['end']
        if not (isinstance(start,int) and isinstance(end,int) and 0<=start<end<=len(source)):
            raise ValueError(f'Invalid action span at anchor {index}')
        if (start,end) in seen:raise ValueError(f'Duplicate action span at anchor {index}')
        seen.add((start,end))
        line_start=source.rfind('\n',0,start)+1
        line_end=source.find('\n',end)
        if line_end<0:line_end=len(source)
        row={'source':source,'offset':start,'trigger':source[start:end],
             'context':source[max(0,line_start-400):line_end],
             'state':verify(workflow,rules)['final_state']}
        operator,info=compiler.compile(row)
        outputs.append({'step':f's{index}','input_state':row['state'],**info})
        if operator is None:
            invalid.append({'step':f's{index}','code':'SCHEMA_VIOLATION','field':'operator',
                            'required':'valid Operator JSON','actual':info.get('error')})
            continue
        if operator.trigger!=source[start:end]:
            invalid.append({'step':f's{index}','code':'ANCHOR_VIOLATION','field':'trigger',
                            'required':source[start:end],'actual':operator.trigger})
        workflow.steps.append(make_step(operator,source,index,start,workflow.steps))
    checked=verify(workflow,rules)
    # No-action discovery has not been run: an empty anchor list is never certified.
    if not anchors:
        invalid.append({'step':None,'code':'SCHEMA_MISSING_STEP','field':'anchors',
                        'required':'at least one supplied action anchor','actual':0})
    if repair and not invalid:
        if llm_repair:
            from .repair import repair_with_compiler
            fixed=repair_with_compiler(workflow,compiler,rules)
        else:fixed=minimal_repair(workflow,rules)
    else:fixed=None
    final=fixed['workflow'] if fixed else workflow
    status='REVIEW' if invalid else fixed['status'] if fixed else 'PASS' if checked['pass'] else 'REVIEW'
    return {'status':status,'workflow':final.model_dump(),'candidate_workflow':workflow.model_dump(),
            'verification':verify(final,rules),'compilation_errors':invalid,'outputs':outputs,
            'repair':{k:v for k,v in fixed.items() if k!='workflow'} if fixed else None}
