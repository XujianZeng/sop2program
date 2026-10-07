"""Vector artwork at the manuscript's final column width, from audited outcomes."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
P=Path(__file__).resolve().parents[1]
O=P/'figures';O.mkdir(exist_ok=True)
A=json.loads((P/'evidence/audit.json').read_text())['pooled']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
colors=['#557f9a','#c2783e','#315c46']
def save(fig,i):
    # Main article retains the pipeline; full outcome plots live in Online Resource 1.
    folder = O if i == 1 else P/'supplement'
    folder.mkdir(exist_ok=True)
    stem = 'Fig1' if i == 1 else f'SupplementaryFig{i-1}'
    for ext in ('pdf','eps','png'):fig.savefig(folder/f'{stem}.{ext}',bbox_inches='tight',dpi=400)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(4.8,3.8))
ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
def box(x,y,w,h,t,fc='#f3f5f6'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.07,rounding_size=0.08',fc=fc,ec='#283c49',lw=.8))
    ax.text(x,y,t,ha='center',va='center',fontsize=8,linespacing=1.25)
def arrow(xy,xy2,style='-',rad=0):
    ax.annotate('',xy=xy2,xytext=xy,arrowprops={'arrowstyle':'-|>','lw':.9,'color':'#283c49','linestyle':style,'connectionstyle':f'arc3,rad={rad}'})
box(1.6,6.85,2.7,1.2,'Protocol text\nGiven anchors\nGiven inventory')
box(5,6.85,2.65,1.2,'LLM compiler\nAction and\narguments', '#e5eef4')
box(8.4,6.85,2.65,1.2,'State replay\nEvidence checks\nInduced rules','#f7e9da')
arrow((3.0,6.85),(3.6,6.85));arrow((6.4,6.85),(7.0,6.85))
box(1.6,4.05,2.7,1.35,'Training traces\nfor rules and\nmodel tuning','#ece7f1')
box(5,4.05,2.65,1.35,'Optional trained\nrepair proposer','#e5eef4')
box(8.4,4.05,2.65,1.35,'Bounded edits\nSource and\ninventory fixed','#e6eee5')
arrow((8.0,6.2),(8.0,4.8));arrow((8.8,4.8),(8.8,6.2))
arrow((7.5,6.18),(5.7,4.8))
ax.text(6.4,5.55,'feedback',ha='right',fontsize=7)
ax.text(9.2,5.5,'replay',ha='left',fontsize=7)
arrow((6.4,4.05),(7,4.05))
arrow((3.0,4.05),(3.6,4.05),'--')
arrow((2.7,4.85),(4.35,6.15),'--')
# Induced rules feed the verifier through a distinct, labelled route.
ax.plot([.18,.05,.05,8.4,8.4],[4.05,4.05,7.9,7.9,7.5],color='#6b5878',ls='--',lw=.8)
ax.text(5,7.88,'rules + development filtering',fontsize=7,ha='center',va='bottom',bbox={'fc':'white','ec':'none','pad':1})
box(8.4,1.15,2.65,1.25,'Accepted\nor REVIEW','#f3f5f6')
arrow((8.4,3.3),(8.4,1.85))
box(3.35,1.15,5.25,1.25,'Entity-chain replay (evaluation only)\nReference-derived inventory\nJoint acceptance criterion','#f4f4f4')
arrow((7.0,1.15),(6.05,1.15))
save(fig,1)

fig,axes=plt.subplots(2,1,figsize=(4.8,5.15))
cv=[('Few-shot\n14B',A['fewshot3']),('Fine-tuned\n3B',A['lora3b']),('Fine-tuned\n14B',A['qwen14'])]
orig=[('Zero-shot\n14B',{'deterministic':{'accepted':7,'confirmed':5},'trained':{'accepted':18,'confirmed':11}}),
      ('Few-shot\n14B',{'deterministic':{'accepted':12,'confirmed':11},'trained':{'accepted':21,'confirmed':14}}),
      ('Fine-tuned\n3B',{'deterministic':{'accepted':30,'confirmed':24},'trained':{'accepted':38,'confirmed':26}}),
      ('Fine-tuned\n14B',{'deterministic':{'accepted':29,'confirmed':23},'trained':{'accepted':38,'confirmed':29}})]
for ax,items,total,panel in [(axes[0],cv,A['qwen14']['documents'],'(a)'),(axes[1],orig,42,'(b)')]:
    for i,(name,r) in enumerate(items):
        for j,arm in enumerate(('deterministic','trained')):
            x=i+(j-.5)*.36;d=r[arm];c=d['confirmed'];n=d['accepted']
            ax.bar(x,c,.31,fc=colors[j],ec='#263238',lw=.6)
            ax.bar(x,n-c,.31,bottom=c,fc='white',ec='#263238',hatch='////',lw=.6)
            ax.text(x,n+total*.015,f'{100*(n-c)/n+1e-9:.1f}%',ha='center',va='bottom',fontsize=7)
    ax.set_xticks(range(len(items)),[r[0] for r in items]);ax.set_ylim(0,total*1.08)
    ax.set_ylabel(f'Programs (of {total})');ax.text(.01,.95,panel,transform=ax.transAxes,ha='left',va='top',fontweight='bold')
    ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
fig.legend(handles=[Patch(fc=colors[0],ec='#263238',label='Deterministic: confirmed'),Patch(fc=colors[1],ec='#263238',label='Trained: confirmed'),Patch(fc='white',hatch='////',ec='#263238',label='Accepted but unconfirmed')],loc='lower center',ncol=1,frameon=False,fontsize=7.5,bbox_to_anchor=(.53,-.02))
fig.tight_layout(rect=(0,.12,1,1),h_pad=1.5)
save(fig,2)

fig,ax=plt.subplots(figsize=(4.8,2.8))
names={'fewshot3':'Few-shot 14B','lora3b':'Fine-tuned 3B','qwen14':'Fine-tuned 14B'}
for j,(c,r) in enumerate(A.items()):
    ys=[r['first'],r['deterministic']['confirmed'],r['trained']['confirmed']]
    ax.plot(range(3),ys,color=colors[j],marker=['o','s','^'][j],label=names[c],lw=1.25,ms=4.5)
    for i,y in enumerate(ys):ax.annotate(str(y),(i,y),xytext=(6,3 if j!=1 else -12),textcoords='offset points',fontsize=8)
untuned=A['fewshot3']['untuned']['confirmed']
deterministic=A['fewshot3']['deterministic']['confirmed']
ax.scatter([1.4],[untuned],marker='x',color=colors[0],s=25)
ax.annotate(f'Untuned: {untuned}',(1.4,untuned),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=8)
ax.plot([1,1.4],[deterministic,untuned],ls=':',color=colors[0],lw=.8)
ax.set_xticks([0,1,2],['First pass','Deterministic','Trained cascade'])
ax.set(xlim=(-.12,2.35),ylim=(0,max(r['trained']['confirmed'] for r in A.values())*1.15),ylabel='Annotation-confirmed programs')
ax.legend(loc='upper left',frameon=False,fontsize=8)
ax.yaxis.grid(True,alpha=.18);ax.set_axisbelow(True);fig.tight_layout()
save(fig,3)
path=P/'manuscript.tex'
if path.exists():
    s=path.read_text(encoding='utf-8')
    for old,i in [('fig_pipeline',1),('fig_overestimation',2),('fig_cv_gains',3)]:s=s.replace(f'figures/{old}.pdf',f'figures/Fig{i}.pdf')
    s=s.replace('width=0.75\\textwidth','width=\\textwidth')
    path.write_text(s,encoding='utf-8')
print('Wrote three vector figures (PDF and EPS) and PNG previews.')
