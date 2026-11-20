"""Draw the primary paired evidence and end-to-end link results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/'pooled.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
colors=['#647489','#117A76','#B35B28']
labels=['Pilot only','Pilot + payload','Selected by codeword score']
for j,(arm,label) in enumerate(zip(('pilot_only','rebuild','code_gate'),labels)):
    e=data['full_reset']['metrics'][arm+'_recover']
    val=e['rate'];lo,hi=e['ci95_percent']
    ax[0].barh(2-j,val,color=colors[j],height=.54)
    ax[0].errorbar(val,2-j,xerr=np.array([[val-lo],[hi-val]]),fmt='none',color='#172B3A',capsize=3)
    ax[0].text(hi+1.1,2-j,f'{val:.1f}%',va='center',fontsize=9)
ax[0].set_yticks([2,1,0],labels)
ax[0].set_xlim(0,78);ax[0].set_xlabel('Share of joint failures recovered (%)')
ax[0].set_title('Full reset · 2,666 joint failures',loc='left',fontsize=13)
x=np.arange(3)
for j,(arm,label) in enumerate(zip(('default','gate','code_gate'),('Previous receiver','Covariance switch','Codeword switch'))):
    y=[data[r]['metrics'][arm+'_wrong']['rate'] for r in ('full_reset','partial_shift','continuous')]
    ax[1].bar(x+(j-1)*.25,y,width=.22,label=label,color=colors[j])
ax[1].set_xticks(x,['Full reset','One path shifts','Continuous']);ax[1].set_ylabel('Block error rate (%)')
ax[1].set_ylim(0,78);ax[1].legend(loc='upper right',frameon=False,fontsize=8)
ax[1].set_title('Receiver output · 4,096 blocks per regime',loc='left',fontsize=13)
for a in ax:
    a.spines[['top','right']].set_visible(False)
fig.savefig(root/'reconstruction_results.pdf',bbox_inches='tight')
fig.savefig(root/'reconstruction_results.png',dpi=185,bbox_inches='tight')
