from sop2program.pipeline import compile_workflow
from sop2program.ir import Argument,Operator,semantic_template


class FakeCompiler:
    def __init__(self):self.states=[]
    def compile(self,row):
        self.states.append(row['state'])
        args=[Argument(role='a',text='pellet',type='material')]
        action={'Create':'CREATE','Wash':'WASH'}[row['trigger']]
        return Operator(action=action,trigger=row['trigger'],arguments=args,**semantic_template(action,args)),{'raw':'test'}


def test_pipeline_uses_predicted_state():
    compiler=FakeCompiler()
    result=compile_workflow('Create pellet. Wash pellet.',[],[{'start':0,'end':6},{'start':15,'end':19}],compiler)
    assert result['status']=='PASS'
    assert compiler.states[0]['exists']==[]
    assert compiler.states[1]['exists']==['pellet']


def test_pipeline_empty_anchor_list_requires_review():
    assert compile_workflow('Wash pellet.',[],[],FakeCompiler())['status']=='REVIEW'


def test_pipeline_missing_state_requires_review():
    result=compile_workflow('Wash pellet.',[],[{'start':0,'end':4}],FakeCompiler())
    assert result['status']=='REVIEW'
    assert result['verification']['final_state']['exists']==[]
