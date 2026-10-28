"""Decision-first figures with the simulation and uncertainty limits visible."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT

OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
summary=json.loads((ROOT/'results/test_summary.json').read_text());selection=json.loads((ROOT/'results/selection.json').read_text());results=summary['results']
NAVY='#17324d';TEAL='#007d7b';GREY='#8393a3';ORANGE='#bc5d28'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold','figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff','pdf.fonttype':42})

def save(fig,name,title,subtitle,footnote):
    fig.suptitle(title,x=.07,y=.965,ha='left',fontsize=19,fontweight='bold')
    fig.text(.07,.902,subtitle,fontsize=10,color=TEAL,ha='left')
    fig.text(.07,.035,footnote,fontsize=8,color='#4a5a68',ha='left',va='bottom',linespacing=1.5)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'{name}.{suffix}',dpi=180)
    plt.close(fig)

def lookup(partition,model):return next(r for r in results if r['partition']==partition and r['model']==model)

fig,axes=plt.subplots(1,2,figsize=(12,6));fig.subplots_adjust(left=.07,right=.97,top=.80,bottom=.23,wspace=.35)
labels=['FD001: one condition','FD002: six conditions'];x=np.arange(2)
for j,(model,color,label) in enumerate([(selection['baseline'],GREY,'Age-only baseline'),(selection['winner'],TEAL,'Selected sensor model')]):
    values=[lookup(p,model) for p in ['test_FD001','test_FD002']];pos=x+(j-.5)*.32
    bars=axes[0].bar(pos,[r['engine_rmse'] for r in values],width=.30,color=color,label=label)
    axes[0].bar_label(bars,fmt='%.1f',padding=4,fontsize=10)
    bars=axes[1].bar(pos,[100*r['inspection']['horizons'][1]['precision'] for r in values],width=.30,color=color)
    for bar,row in zip(bars,values):
        h=row['inspection']['horizons'][1];axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f"{h['captured']}/{row['inspection']['budget']}",ha='center',fontsize=10)
for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set(ylabel='Engine-balanced RMSE (remaining cycles)',ylim=(0,58),title='Lower error in the matched condition')
axes[1].set(ylabel='Selected engines within30 cycles (%)',ylim=(0,118),yticks=[0,25,50,75,100],title='Inspection advantage reverses under shift')
axes[0].legend(frameon=False,loc='upper left')
save(fig,'decision-summary','Better life estimates do not guarantee robust inspection rankings','NASA SIMULATION / RETROSPECTIVE  |  Fixed10% inspection budget; last observed window per engine','FD001:100 engines,10 inspections. FD002:259 engines,25 inspections. Labels show captured/inspected engines.\nEach batch is a benchmark construction, not a synchronized operating fleet. No cost or safety benefit was measured.')

search=pd.read_csv(ROOT/'results/search.csv').sort_values('engine_mse',ascending=True)
fig,axes=plt.subplots(1,2,figsize=(12,6.8));fig.subplots_adjust(left=.27,right=.97,top=.80,bottom=.20,wspace=.45)
colors=[TEAL if name==selection['winner'] else GREY for name in search.name]
names=[]
for _,trial in search.iterrows():
    if trial.kind=='baseline':names.append('Age-only baseline')
    elif trial.kind=='forest':names.append(f"Forest depth {int(trial.max_depth) if pd.notna(trial.max_depth) else 'unlimited'} / leaf {int(trial.min_samples_leaf)}")
    else:names.append(f"Boost depth {int(trial.max_depth)} / LR {trial.learning_rate:g} / L2 {trial.l2_regularization:g}")
axes[0].barh(np.arange(len(search)),search.engine_mse,color=colors);axes[0].set_yticks(np.arange(len(search)),names,fontsize=8);axes[0].invert_yaxis();axes[0].set(xlabel='Tuning engine-balanced MSE (cycles²)',title='All15 configurations retained');axes[0].grid(axis='x',alpha=.15);axes[0].set_axisbelow(True)
for i,record in enumerate(summary['bootstrap']):
    low,high=record['percentile95'];axes[1].errorbar(record['mean'],i,xerr=[[record['mean']-low],[high-record['mean']]],fmt='o',color=TEAL,capsize=5)
axes[1].axvline(0,color=ORANGE,lw=1);axes[1].set(yticks=[0,1],yticklabels=['FD001','FD002'],xlabel='Selected minus baseline MSE (cycles²)',title='Paired-engine test differences',ylim=(-.5,1.5));axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.15)
save(fig,'optimization-evidence','Selection and test uncertainty are separate evidence','NASA SIMULATION / RETROSPECTIVE  |  70 fit,15 tune,15 uncertainty-calibration engines','Winner: histogram boosting, depth2, learning rate0.1, L2=0. The learning-rate choice lies at the searched boundary.\nIntervals:2,000 paired-engine bootstrap draws, conditional on these simulated test engines; not a global-optimum proof.')

fig,axes=plt.subplots(1,2,figsize=(12,6));fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.23,wspace=.4)
for j,(model,color,label) in enumerate([(selection['baseline'],GREY,'Age-only baseline'),(selection['winner'],TEAL,'Selected sensor model')]):
    rows=[lookup(p,model) for p in ['test_FD001','test_FD002']];pos=x+(j-.5)*.32
    bars=axes[0].bar(pos,[r['mean_interval_width'] for r in rows],width=.30,color=color,label=label);axes[0].bar_label(bars,fmt='%.0f',padding=4)
    bars=axes[1].bar(pos,[100*r['trajectory_coverage'] for r in rows],width=.30,color=color);axes[1].bar_label(bars,fmt='%.1f',padding=4)
for ax in axes:ax.set_xticks(x,['FD001','FD002']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set(ylabel='Mean prediction-interval width (cycles)',ylim=(0,335),title='High coverage comes with broad intervals')
fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(.07,.825))
axes[1].axhline(90,color=ORANGE,lw=1,ls='--');axes[1].set(ylabel='Engines with every window covered (%)',ylim=(0,115),yticks=[0,25,50,75,100],title='Observed trajectory coverage')
save(fig,'uncertainty-tradeoff','Broad uncertainty limits maintenance specificity','NASA SIMULATION / RETROSPECTIVE  |  Nominal90% simultaneous trajectory coverage','Each radius is the largest maximum-trajectory residual among15 held-out calibration engines; lower bounds clip at0.\nDashed line: nominal90%. Exchangeability is an assumption, and these wide intervals do not establish deployment readiness.')
print(json.dumps({'figures':3,'formats':['png','pdf'],'simulation_label_on_every_figure':True}))
