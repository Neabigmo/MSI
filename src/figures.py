"""Five manuscript figures from formal results. No expression matrices required."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from calibration_metrics import calibration_curve_frame, integrated_calibration_index

COHORTS = ['TCGA_COAD','GSE13294','GSE13067','GSE18088','GSE26682','GSE39084']
EXT = COHORTS[1:]
FIXED, ADAPTIVE, RANK = 'fixed_log_expression','target_cohort_adaptive_zscore','single_sample_rank'
REPS = [FIXED, ADAPTIVE, RANK]
BLUE, GREEN, ORANGE, INK = '#0072B2','#009E73','#D55E00','#243746'
COLORS = [INK, BLUE, ORANGE, GREEN, '#CC79A7','#E69F00']
MARKERS = ['o','s','^','D','v']
NAMES = ['figure1_framework','figure2_domain_task','figure3_interaction_calibration','figure4_same_platform','figure5_stability_biology']

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'axes.titleweight':'bold','xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'axes.edgecolor':'#82909C','text.color':INK,'axes.labelcolor':INK,
        'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})

def heading(ax, letter, title):
    ax.set_title(title,loc='left',pad=13)
    ax.text(-.14,1.055,letter,transform=ax.transAxes,fontsize=13,fontweight='bold',va='bottom')

def save(fig,out,name,note):
    fig.text(.06,.018,note,fontsize=8,color='#53616D',va='bottom')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=400)
    plt.close(fig)

def cohort_legend(fig,y=.075):
    handles=[Line2D([],[],ls='',marker=m,color='#637381',label=c,markersize=5) for c,m in zip(EXT,MARKERS)]
    handles.append(Line2D([],[],ls='',marker='s',color='black',label='Equal-weight mean',markersize=6))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,y),ncol=6,handletextpad=.4,columnspacing=1.3)

def paired(ax,frame,metric,column,levels,labels,colors=(BLUE,GREEN),ylim=None):
    z=frame.pivot(index='cohort',columns=column,values=metric).reindex(EXT)[levels]
    if z.isna().any().any(): raise ValueError('Incomplete paired data')
    offsets=np.linspace(-.10,.10,len(z))
    for i,(_,row) in enumerate(z.iterrows()):
        ax.plot([offsets[i],1+offsets[i]],row.values,lw=.9,color='#C2C9CE',zorder=1)
        for j in range(2): ax.scatter(j+offsets[i],row.iloc[j],marker=MARKERS[i],color=colors[j],s=32,edgecolor='white',linewidth=.4,zorder=3)
    means=z.mean(); ax.plot([0,1],means,'s--',color='black',ms=5.5,lw=1.1,zorder=4)
    ax.text(.02,.025,f'Mean: {means.iloc[0]:.3f} → {means.iloc[1]:.3f}',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    ax.set_xticks([0,1],labels); ax.set_xlim(-.25,1.25); ax.grid(axis='y',color='#EDF0F2',lw=.6)
    if ylim: ax.set_ylim(*ylim)

def slope_interval(y, probability, seed, n_boot=500):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(probability,dtype=float),1e-7,1-1e-7)
    x=np.log(p/(1-p)).reshape(-1,1)
    def fit(idx):
        yy=y[idx]
        if np.unique(yy).size<2: return np.nan
        return LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000).fit(x[idx],yy).coef_[0,0]
    point=fit(np.arange(len(y))); rng=np.random.default_rng(seed); draws=[]
    for _ in range(n_boot):
        value=fit(rng.integers(0,len(y),len(y)))
        if np.isfinite(value): draws.append(value)
    return point,*np.quantile(draws,[.025,.975]),len(draws),int(np.sum(np.asarray(draws)<=0))

def figure1(d,out):
    fig=plt.figure(figsize=(11.7,8.0)); gs=fig.add_gridspec(2,20,left=.075,right=.98,top=.94,bottom=.16,hspace=.55,wspace=.65,height_ratios=[1.35,1])
    ax=fig.add_subplot(gs[0,:5]); m=d['cohort_manifest'].set_index('cohort').loc[COHORTS]; y=np.arange(6)
    ax.barh(y,m.msi,color=ORANGE,height=.64,label='MSI'); ax.barh(y,m.mss,left=m.msi,color='#B9D9EC',height=.64,label='MSS')
    for i,row in enumerate(m.itertuples()):
        ax.text(row.msi/2,i,str(row.msi),color='white',va='center',ha='center',fontsize=8)
        ax.text(row.msi+row.mss/2,i,str(row.mss),va='center',ha='center',fontsize=8)
        ax.text(row.n+3,i,f'n={row.n}',va='center',fontsize=7.5)
    ax.set_yticks(y,['TCGA-COAD',*EXT]); ax.invert_yaxis(); ax.set_xlim(0,196); ax.set_xlabel('Number of samples')
    ax.legend(loc='lower right',ncol=2,bbox_to_anchor=(1,-.29)); heading(ax,'a','Study cohorts')
    ax=fig.add_subplot(gs[0,5:]); ax.axis('off'); heading(ax,'b','Development and target-information access')
    from matplotlib.patches import Rectangle
    def rect(x,y,w,h,text,face='white',size=7.2,weight='normal'):
        ax.add_patch(Rectangle((x,y),w,h,transform=ax.transAxes,fc=face,ec='#AAB4BC',lw=.75))
        ax.text(x+w/2,y+h/2,text,transform=ax.transAxes,ha='center',va='center',fontsize=size,weight=weight,linespacing=1.18)
    def arrow(x1,y1,x2,y2,color=INK,ls='-'):
        ax.annotate('',(x2,y2),(x1,y1),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=color,lw=.8,linestyle=ls))
    ax.text(.01,.93,'1  DISCOVERY DEVELOPMENT · TCGA-COAD, n=160',transform=ax.transAxes,fontsize=7.4,weight='bold')
    rect(.01,.68,.14,.18,'Expression\n11,316 genes','#F4F6F7',weight='bold'); rect(.19,.68,.17,.18,'Fixed  |  Adaptive Z\nWithin-sample rank','#EAF3F8',weight='bold')
    rect(.40,.68,.22,.18,'3-fold CV\nfold-wise screen 200\n9-setting tuning','#F4F6F7',weight='bold'); rect(.66,.68,.15,.18,'Pool OOF\nlock threshold','#F4F6F7'); rect(.85,.68,.14,.18,'Final fit\nfreeze model\n+ threshold','#EAF3F8',weight='bold')
    for a,b in [(.15,.19),(.36,.40),(.62,.66),(.81,.85)]: arrow(a,.77,b,.77)
    ax.plot([.01,.99],[.61,.61],transform=ax.transAxes,c='#7B8790',ls='--',lw=.8)
    ax.text(.01,.55,'2  EXTERNAL PREDICTION · five GEO cohorts, n=486',transform=ax.transAxes,fontsize=7.4,weight='bold')
    rect(.01,.25,.15,.22,'GPL570 input\nfixed probe map\nsample-wise mean','#F4F6F7'); rect(.24,.43,.23,.10,'Fixed expression\n(no target statistics)','#EAF3F8',weight='bold'); rect(.24,.315,.23,.10,'Adaptive Z\n(target mean + SD)','#FDEFE9',weight='bold'); rect(.24,.20,.23,.10,'Within-sample rank\n(no target statistics)','#EAF3F8',weight='bold')
    rect(.59,.25,.18,.22,'Apply matching\nfrozen classifier\n+ locked threshold','#F4F6F7',weight='bold'); rect(.84,.25,.15,.22,'Per-sample\nprobability\n+ decision','#F4F6F7')
    # A single trunk avoids crossing arrows while preserving the three branches.
    arrow(.16,.36,.20,.36); ax.plot([.20,.20],[.25,.48],transform=ax.transAxes,c=INK,lw=.8)
    for yy in [.48,.365,.25]: arrow(.20,yy,.24,yy)
    ax.plot([.47,.55],[.48,.48],transform=ax.transAxes,c=INK,lw=.8); ax.plot([.47,.55],[.365,.365],transform=ax.transAxes,c=INK,lw=.8); ax.plot([.47,.55],[.25,.25],transform=ax.transAxes,c=INK,lw=.8); ax.plot([.55,.55],[.25,.48],transform=ax.transAxes,c=INK,lw=.8); arrow(.55,.365,.59,.365); arrow(.77,.36,.84,.36)
    ax.text(.355,.135,'Unlabelled target batch statistics',transform=ax.transAxes,fontsize=6.8,color=ORANGE,ha='center'); arrow(.355,.16,.355,.315,ORANGE,'--')
    m=d['metrics']; ext=m[(m.design=='TCGA_to_GEO')&(m.model=='elastic_net')&m.representation.isin([FIXED,RANK])]
    for i,metric,title in [(0,'auroc','Discrimination · Elastic Net'),(1,'balanced_accuracy','Locked decisions · Elastic Net')]:
        ax=fig.add_subplot(gs[1,10*i:10*i+10]); heading(ax,'cd'[i],title)
        paired(ax,ext,metric,'representation',[FIXED,RANK],['Fixed expression','Within-sample rank'],ylim=(.45,1.035))
        ax.set_ylabel('AUROC' if i==0 else 'Balanced accuracy')
    cohort_legend(fig,.067); save(fig,out,NAMES[0],'Lines pair the same external cohort. Squares show the unweighted mean across five cohorts.')

def figure2(d,out):
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.3)); fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.27,wspace=.43)
    coords=d['source_pca_coordinates']
    for ax,rep,letter,title in zip(axes,[FIXED,RANK],'ab',['Fixed expression','Within-sample rank']):
        g=coords[coords.representation==rep]
        for cohort,color in zip(COHORTS,COLORS):
            z=g[g.cohort==cohort]; ax.scatter(z.pc1,z.pc2,s=8,color=color,alpha=.38,edgecolors='none',rasterized=True)
        ax.set_xlabel(f'PC1 ({g.pc1_variance_pct.iloc[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({g.pc2_variance_pct.iloc[0]:.1f}%)'); heading(ax,letter,title)
    ax=axes[2]; heading(ax,'c','Source identification within labels'); src=d['source_classifier_stratified']
    for j,rep in enumerate([FIXED,RANK]):
        g=src[src.representation==rep].set_index('label_stratum').loc[['MSS','MSI']]; x=np.arange(2)+(j-.5)*.19
        ax.scatter(x,g.source_balanced_accuracy,marker=['o','^'][j],color=[BLUE,GREEN][j],s=42,label=['Fixed','Rank'][j],zorder=3)
        for k,row in enumerate(g.itertuples()): ax.annotate(f'{row.source_balanced_accuracy:.3f}',(x[k],row.source_balanced_accuracy),xytext=(0,9 if j==0 else -18),textcoords='offset points',ha='center',fontsize=7)
    ax.axhline(1/6,color='#7B8790',ls='--',lw=.9); ax.text(.02,1/6+.035,'Six-source chance = 1/6',fontsize=7.5)
    sizes=src.drop_duplicates('label_stratum').set_index('label_stratum').n
    ax.set_xticks([0,1],[f'MSS (n={sizes.MSS})',f'MSI (n={sizes.MSI})']); ax.set(ylim=(0,1.12),xlim=(-.45,1.45),ylabel='Source balanced accuracy'); ax.legend(loc='center right')
    handles=[Line2D([],[],ls='',marker='o',color=color,label=c.replace('_','-'),markersize=5) for c,color in zip(COHORTS,COLORS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=6)
    save(fig,out,NAMES[1],'PCA is fitted separately in each representation; distances are not comparable between panels.\nSource classification uses training-fold feature selection; points are pooled five-fold out-of-fold estimates.')

def figure3(d,out):
    m=d['metrics'].query("design == 'TCGA_to_GEO'")
    fig=plt.figure(figsize=(11.4,9.3)); gs=fig.add_gridspec(2,6,left=.10,right=.96,top=.93,bottom=.22,hspace=.55,wspace=.95,height_ratios=[1.0,1.25])
    for i,metric,title in [(0,'auroc','AUROC'),(1,'balanced_accuracy','Locked balanced accuracy'),(2,'brier_skill','Brier skill')]:
        ax=fig.add_subplot(gs[0,2*i:2*i+2]); mat=m.groupby(['representation','model'])[metric].mean().unstack().reindex(REPS)[['elastic_net','svm_rbf','xgboost']]
        norm=TwoSlopeNorm(vmin=-2.5,vcenter=0,vmax=1) if i==2 else plt.Normalize(.5,1)
        im=ax.imshow(mat,cmap='PuOr' if i==2 else 'cividis',norm=norm,aspect='auto')
        for y in range(3):
            for x in range(3):
                val=mat.iloc[y,x]; rgba=im.cmap(norm(val)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(x,y,f'{val:.3f}',ha='center',va='center',fontsize=9,color='black' if lum>.56 else 'white')
        ax.set_xticks(range(3),['EN','SVM-RBF','XGB'],rotation=0,ha='center'); ax.set_yticks(range(3),['Fixed','Adaptive Z','Rank']); heading(ax,'abc'[i],title)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.27,aspect=28); cb.ax.tick_params(labelsize=7,length=2)
    ax=fig.add_subplot(gs[1,:3]); heading(ax,'d','GSE13294 · Elastic Net calibration'); ax.plot([0,1],[0,1],ls='--',color='#89949D',lw=1,label='Ideal')
    pred=d['predictions']
    for rep,color,ls in [(FIXED,BLUE,'-'),(RANK,GREEN,'--')]:
        g=pred[(pred.design=='TCGA_to_GEO')&(pred.cohort=='GSE13294')&(pred.model=='elastic_net')&(pred.representation==rep)]
        curve=calibration_curve_frame(g.label,g.probability); use=curve.predicted_probability.between(g.probability.min(),g.probability.max()); ici=integrated_calibration_index(g.label,g.probability)
        ax.plot(curve.loc[use,'predicted_probability'],curve.loc[use,'smoothed_observed_probability'],color=color,lw=1.9,ls=ls,label=f"{'Fixed' if rep==FIXED else 'Rank'} · ICI {ici:.3f}")
        ax.plot(g.probability,np.repeat(.015 if rep==FIXED else .04,len(g)),'|',color=color,alpha=.35,ms=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted MSI probability',ylabel='Smoothed observed proportion'); ax.legend(loc='upper left',fontsize=8)
    ax=fig.add_subplot(gs[1,3:]); heading(ax,'e','Calibration slopes across cohorts')
    interval_rows=[]
    for i,rep in enumerate([FIXED,RANK]):
        xs=np.arange(5)+(i-.5)*.22; points=[]; lows=[]; highs=[]
        for k,cohort in enumerate(EXT):
            g=pred[(pred.design=='TCGA_to_GEO')&(pred.cohort==cohort)&(pred.model=='elastic_net')&(pred.representation==rep)]
            point,low,high,n_finite,n_nonpositive=slope_interval(g.label,g.probability,20261008+i*20+k); points.append(point); lows.append(low); highs.append(high)
            interval_rows.append(dict(cohort=cohort,representation=rep,calibration_slope=point,ci_lower=low,ci_upper=high,n_bootstrap_finite=n_finite,n_bootstrap_nonpositive=n_nonpositive))
        points=np.array(points); lows=np.array(lows); highs=np.array(highs)
        visible_highs=np.minimum(highs,60)
        ax.errorbar(xs,points,yerr=[points-lows,visible_highs-points],fmt=['o','^'][i],color=[BLUE,GREEN][i],ms=5,capsize=2,lw=.9,label=['Fixed','Rank'][i])
        for x,upper in zip(xs,highs):
            if upper > 60:
                ax.annotate('',xy=(x,57),xytext=(x,43),arrowprops=dict(arrowstyle='-|>',color=[BLUE,GREEN][i],lw=.9),annotation_clip=False)
                ax.text(x,38,f'{upper:.0f}',ha='center',va='top',fontsize=6.2,color=[BLUE,GREEN][i],rotation=90)
    counts=pred[(pred.design=='TCGA_to_GEO')&(pred.model=='elastic_net')&(pred.representation==FIXED)].groupby('cohort').label.agg(['size','sum']).loc[EXT]
    ax.set_yscale('log'); ax.set_ylim(.05,60); ax.axhline(1,c='#89949D',ls='--',lw=1)
    ax.set_xticks(range(5),EXT,rotation=18,ha='right',fontsize=7.2)
    table=ax.table(cellText=[[int(counts.loc[c,'size']) for c in EXT],[int(counts.loc[c,'sum']) for c in EXT]],rowLabels=['n','MSI'],cellLoc='center',bbox=[0,-.31,1,.16])
    table.auto_set_font_size(False); table.set_fontsize(6.5)
    for cell in table.get_celld().values(): cell.set_linewidth(0); cell.set_facecolor('white')
    ax.set_ylabel('Calibration slope (log scale; 95% bootstrap CI)'); ax.legend(loc='lower left')
    pd.DataFrame(interval_rows).to_csv(out/'calibration_slope_bootstrap_intervals.csv',index=False)
    save(fig,out,NAMES[2],'Heatmaps: equal-weight external-cohort means. Curves: LOWESS (span 0.75), recomputed from formal predictions.\nRug marks show predicted probabilities; slope = 1 is ideal. Upward arrows mark CIs truncated at the display limit; labels give true upper bounds.')

def figure4(d,out):
    fig=plt.figure(figsize=(11.4,8.6)); gs=fig.add_gridspec(2,6,left=.095,right=.97,top=.92,bottom=.18,hspace=.57,wspace=1.05)
    agrid=gs[0,:3].subgridspec(1,2,width_ratios=[1.35,.80],wspace=.28)
    ax=fig.add_subplot(agrid[0]); heading(ax,'a','RBF kernel similarity across training designs')
    a,b=d['rank_svm_kernel_distribution'],d['rank_svm_loco_kernel_distribution']
    groups=[a[a.kernel_type=='discovery within'],a[a.kernel_type=='external cross'],b[b.kernel_type=='LOCO training within'],b[b.kernel_type!='LOCO training within']]
    vals=[z.log_kernel_similarity.to_numpy()/np.log(10) for z in groups]
    bp=ax.boxplot(vals,patch_artist=True,widths=.5,showfliers=False,medianprops=dict(color='black',lw=1))
    for patch,color in zip(bp['boxes'],['#B9D9EC',BLUE,'#BFE2D6',GREEN]): patch.set_facecolor(color); patch.set_edgecolor('#64727E')
    ax.set_yscale('symlog',linthresh=1); ax.set_yticks([-1000,-100,-10,-1,0],['−1000','−100','−10','−1','0']); ax.set_ylim(min(np.min(v) for v in vals)*1.1,.4)
    ax.set_xticks([1,2,3,4],['TCGA\nwithin','TCGA→GEO\ncross','GEO-LOCO\nwithin','GEO-LOCO\ncross'],fontsize=7.5); ax.set_ylabel(r'$\log_{10} K$ (symmetric-log axis)')
    ax.axhline(np.log10(np.nextafter(0.,1.)),lw=.7,c='#8B949E',ls=':'); ax.text(.98,.04,'Dotted: float64 underflow boundary',fontsize=7,ha='right',transform=ax.transAxes)
    # A separate axis keeps the training-within scale readable while the main
    # axis retains the full cross-cohort range. Nothing is drawn on top of the
    # main plot.
    axins=fig.add_subplot(agrid[1])
    within=[vals[0],vals[2]]
    ibp=axins.boxplot(within,patch_artist=True,widths=.48,showfliers=False,medianprops=dict(color='black',lw=.8))
    for patch,color in zip(ibp['boxes'],['#B9D9EC','#BFE2D6']): patch.set_facecolor(color); patch.set_edgecolor('#64727E')
    axins.set_ylim(-1.0,.05); axins.set_yticks([-1,-.5,0],['−1','−0.5','0']); axins.set_xticks([1,2],['TCGA','GEO-LOCO'],fontsize=6.5)
    axins.tick_params(axis='y',labelsize=6.5,length=2); axins.set_title('Training-within scale',fontsize=7.5,pad=5); axins.set_ylabel(r'$\log_{10} K$',fontsize=7.5)
    ax=fig.add_subplot(gs[0,3:]); heading(ax,'b','TCGA-trained rank-SVM probabilities · expanded scale'); p=d['predictions']; z=p[(p.design=='TCGA_to_GEO')&(p.representation==RANK)&(p.model=='svm_rbf')]; rng=np.random.default_rng(20261008)
    observed=z.probability.to_numpy(); lo=max(0.,float(observed.min())-.012); hi=min(1.,float(observed.max())+.012)
    for i,cohort in enumerate(EXT):
        g=z[z.cohort==cohort]
        for label in [0,1]:
            values=g[g.label==label].probability
            x=i+rng.uniform(-.24,.24,len(values))
            display_jitter=rng.normal(0,.0012,len(values))
            ax.scatter(x,values.to_numpy()+display_jitter,s=10,alpha=.42,marker='o' if label==0 else '^',color=BLUE if label==0 else ORANGE,edgecolors='none',label=('MSS' if label==0 else 'MSI') if i==0 else None)
        ax.hlines(g.probability.mean(),i-.28,i+.28,color=INK,lw=1.5,zorder=5)
        ax.text(i,lo+.001,f'n={len(g)}',ha='center',fontsize=7)
        ax.text(i,lo+.0032,f'true SD {g.probability.std():.1e}',ha='center',fontsize=5.8,color='#5E6B75')
    threshold=z.threshold.unique(); assert len(threshold)==1
    ax.set_xlim(-.45,4.45); ax.set_ylim(lo,hi); ax.set_xticks(range(5),EXT,rotation=28,ha='right',fontsize=7); ax.set_yticks(np.round(np.linspace(lo,hi,5),3)); ax.tick_params(axis='y',labelsize=7,length=2)
    ax.set_ylabel('Display position around predicted MSI probability\n(vertical spread is artificial jitter)'); ax.grid(axis='y',color='#EDF0F2',lw=.6)
    ax.text(.99,.96,f'Display-jittered observations\nLocked threshold = {threshold[0]:.3f} (off scale)',transform=ax.transAxes,ha='right',va='top',fontsize=7.2,color='#4C5963',bbox=dict(fc='white',ec='#CBD5DC',pad=3,alpha=.9))
    ax.legend(loc='upper left',fontsize=7)
    m=d['metrics']; m=m[(m.representation==RANK)&m.design.isin(['TCGA_to_GEO','GEO_LOCO'])]
    for i,model,metric,title in [(0,'svm_rbf','auroc','Rank-SVM · discrimination'),(1,'svm_rbf','balanced_accuracy','Rank-SVM · locked decisions'),(2,'elastic_net','auroc','Rank Elastic Net · control')]:
        ax=fig.add_subplot(gs[1,2*i:2*i+2]); heading(ax,'cde'[i],title)
        paired(ax,m[m.model==model],metric,'design',['TCGA_to_GEO','GEO_LOCO'],['TCGA→GEO','GEO-LOCO'],ylim=(.45,1.04)); ax.set_ylabel('Balanced accuracy' if metric=='balanced_accuracy' else 'AUROC')
    cohort_legend(fig,.085)
    save(fig,out,NAMES[3],'A: full kernel scale; right axis = training-within zoom. B: expanded probability scale; point jitter is for display only. Threshold 0.525 is off-scale; exact values are retained in the supplied CSV.')

def figure5(d,out):
    stab=d['compact_gene_stability'].head(20); genes=stab.gene.tolist(); coef=d['compact_rank_frozen_coefficients'].set_index('gene').loc[genes]
    matrix=d['stable_gene_cohort_directions'].pivot(index='gene',columns='cohort',values='median_rank_msi_minus_mss').loc[genes,COHORTS]
    assert not matrix.isna().any().any(); scaled=matrix.div(matrix.abs().max(axis=1).replace(0,1),axis=0)
    fig=plt.figure(figsize=(11.4,10.5)); gs=fig.add_gridspec(2,3,left=.10,right=.96,top=.92,bottom=.17,height_ratios=[1.9,1],hspace=.46,wspace=.58); y=np.arange(20)
    ax=fig.add_subplot(gs[0,0]); heading(ax,'a','Full-pipeline selection stability'); ax.barh(y,stab.selection_frequency,color=GREEN,height=.66)
    for i,val in enumerate(stab.selection_frequency): ax.text(val+.015,i,f'{val:.3f}',va='center',fontsize=7)
    ax.set_xlim(0,1.19); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xlabel('Selection frequency · 500 bootstraps'); ax.tick_params(axis='y',labelsize=8.5); ax.set_yticks(y,genes); ax.set_ylim(19.6,-.6)
    ax=fig.add_subplot(gs[0,1]); heading(ax,'b','Frozen compact-model coefficients'); ax.axvline(0,color='#9AA6AF',ls='--',lw=.8)
    ax.hlines(y,0,coef.coefficient_scaled,color='#BCC5CD',lw=1); ax.scatter(coef.coefficient_scaled,y,s=26,c=np.where(coef.coefficient_scaled<0,BLUE,ORANGE))
    ax.set_yticks(y,genes); ax.tick_params(axis='y',labelsize=8.5); ax.set_ylim(19.6,-.6); ax.set_xlabel('Standardized coefficient')
    ax=fig.add_subplot(gs[0,2]); heading(ax,'c','MSI–MSS rank directions'); im=ax.imshow(scaled,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_yticks(y,genes); ax.tick_params(axis='y',labelsize=8.5); ax.set_ylim(19.6,-.6); ax.set_xticks(range(6),['TCGA',*EXT],rotation=35,ha='right',fontsize=7)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04); cb.set_ticks([-1,0,1]); cb.set_label('Within-gene scaled median difference',fontsize=8)
    m=d['metrics']; ext=m[(m.design=='TCGA_to_GEO')&(m.representation==RANK)&m.model.isin(['elastic_net','compact_rank_20'])]
    for i,metric,title,ylim in [(0,'auroc','Discrimination',(.87,1.0)),(1,'balanced_accuracy','Locked decisions',(.65,1.035)),(2,'brier_skill','Probability accuracy',(-1.75,1.05))]:
        ax=fig.add_subplot(gs[1,i]); heading(ax,'def'[i],title); paired(ax,ext,metric,'model',['elastic_net','compact_rank_20'],['Rank EN\n(200 selected)','Compact rank EN\n(20 selected)'],(BLUE,ORANGE),ylim)
        ax.set_ylabel({'auroc':'AUROC','balanced_accuracy':'Balanced accuracy','brier_skill':'Brier skill'}[metric])
        if metric=='brier_skill': ax.axhline(0,color='#9AA6AF',ls='--',lw=.8)
    cohort_legend(fig,.084)
    save(fig,out,NAMES[4],'a–c share the same gene order. Coefficients are from the final fitted model; they are not bootstrap medians.\nc: compare signs and within-gene patterns, not effect sizes between genes. Compact-20 still requires 11,316-gene ranks.')

def supplementary_fixed_feature_control(d,out):
    """Plot the cohort-level fixed-panel rank-SVM control beside the formal fit."""
    formal=d['metrics'][(d['metrics'].design=='TCGA_to_GEO') &
                        (d['metrics'].representation==RANK) &
                        (d['metrics'].model=='svm_rbf')][['cohort','auroc','balanced_accuracy','brier_skill']].copy()
    formal['feature_panel']='Formal representation-specific panel'
    control=d['fixed_feature_control_metrics'][(d['fixed_feature_control_metrics'].design=='TCGA_to_GEO') &
                                               (d['fixed_feature_control_metrics'].representation==RANK) &
                                               (d['fixed_feature_control_metrics'].model=='svm_rbf')][['cohort','auroc','balanced_accuracy','brier_skill']].copy()
    control['feature_panel']='Fixed discovery 200-gene panel'
    frame=pd.concat([formal,control],ignore_index=True)
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.5)); fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.24,wspace=.45)
    levels=['Formal representation-specific panel','Fixed discovery 200-gene panel']
    for ax,metric,title,letter,ylim in zip(axes,['auroc','balanced_accuracy','brier_skill'],['AUROC','Locked balanced accuracy','Brier skill'],['a','b','c'],[(.45,1.04),(.45,1.04),(-1.75,1.04)]):
        z=frame.pivot(index='cohort',columns='feature_panel',values=metric).reindex(EXT)[levels]
        offsets=np.array([-.12,.12])
        for i,(_,row) in enumerate(z.iterrows()):
            ax.plot(offsets+0,row.values,color='#C2C9CE',lw=.9,zorder=1)
            for j in range(2):
                ax.scatter(offsets[j],row.iloc[j],marker=MARKERS[i],s=38,color=[BLUE,ORANGE][j],edgecolor='white',linewidth=.4,zorder=3)
        means=z.mean(); ax.plot(offsets,means.values,'s--',color='black',ms=5.5,lw=1.1,zorder=4)
        ax.set_xticks(offsets,['Formal panel','Fixed 200 genes']); ax.set_xlim(-.35,.35); ax.set_ylim(*ylim); ax.grid(axis='y',color='#EDF0F2',lw=.6); ax.set_ylabel(title); heading(ax,letter,title)
        if metric=='brier_skill': ax.axhline(0,color='#9AA6AF',ls='--',lw=.8)
        ax.text(.04,.04,f'Mean: {means.iloc[0]:.3f} → {means.iloc[1]:.3f}',transform=ax.transAxes,fontsize=7.5,bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    handles=[Line2D([],[],ls='',marker=m,color='#637381',label=c,markersize=5) for c,m in zip(EXT,MARKERS)]
    handles += [Line2D([],[],ls='',marker='o',color=BLUE,label='Formal panel',markersize=5),Line2D([],[],ls='',marker='o',color=ORANGE,label='Fixed 200 genes',markersize=5)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=7,handletextpad=.4,columnspacing=1.0,fontsize=7.5)
    save(fig,out,'supplementary_fixed_feature_control','Rank-SVM external metrics by cohort. The fixed 200-gene panel is selected within each discovery outer fold for nested OOF evaluation and on all discovery samples for the frozen external fit. Thresholds are locked from nested OOF predictions.')

def validate(d,out):
    m,p=d['metrics'],d['predictions']; keys=['design','cohort','representation','model']
    assert not m.duplicated(keys).any() and not p.duplicated(keys+['sample_id']).any()
    indexed=m.set_index(keys); rows=[]
    for key,g in p.groupby(keys):
        if key[0] not in ['TCGA_to_GEO','GEO_LOCO']: continue
        row=indexed.loc[key]; prob,y=g.probability.to_numpy(),g.label.to_numpy(); brier=brier_score_loss(y,prob)
        values=dict(auroc=roc_auc_score(y,prob),balanced_accuracy=balanced_accuracy_score(y,prob>=g.threshold),brier=brier,brier_skill=1-brier/(y.mean()*(1-y.mean())),ici=integrated_calibration_index(y,prob))
        for metric,value in values.items():
            delta=abs(value-row[metric])
            if not np.isfinite(value) or not np.isfinite(row[metric]) or delta>2e-6:
                raise ValueError(f'Prediction/metric mismatch {key}: {metric}, difference={delta}')
            rows.append(dict(zip(keys,key),metric=metric,recomputed=value,reported=row[metric],absolute_difference=delta))
    pd.DataFrame(rows).to_csv(out/'prediction_metric_checks.csv',index=False)
    stab=d['compact_gene_stability']; assert len(stab)==11316 and stab.bootstrap_count.eq(500).all()
    assert stab.selection_frequency.between(0,1).all() and stab.sign_consistency.between(0,1).all()
    assert d['cohort_manifest'].n.sum()==646
    summaries=[]
    for name in ['rank_svm_kernel_distribution','rank_svm_loco_kernel_distribution']:
        for key,g in d[name].groupby(['design','cohort','kernel_type']):
            logs=g.log_kernel_similarity/np.log(10)
            summaries.append(dict(design=key[0],cohort=key[1],kernel_type=key[2],n_saved=len(g),log10_min=logs.min(),log10_q25=logs.quantile(.25),log10_median=logs.median(),log10_q75=logs.quantile(.75),log10_max=logs.max(),mean_saved_kernel=g.kernel_similarity.mean(),saved_zero_fraction=g.kernel_similarity.eq(0).mean()))
    pd.DataFrame(summaries).to_csv(out/'kernel_distribution_summary.csv',index=False)
    svm=p[(p.design=='TCGA_to_GEO')&(p.model=='svm_rbf')&(p.representation==RANK)]
    svm.groupby('cohort').probability.agg(['count','min','median','max','std','nunique']).to_csv(out/'rank_svm_probability_summary.csv')
    print(f'Checked {len(rows)} metric values against formal predictions; 500 bootstraps / 11,316 genes.',flush=True)

def main():
    root=Path(__file__).resolve().parents[1]; parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=root/'results'/'generated'); parser.add_argument('--output-dir',type=Path,default=root/'figures'/'generated'); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    inputs=['cohort_manifest','metrics','predictions','source_pca_coordinates','source_classifier_stratified','compact_gene_stability','compact_rank_frozen_coefficients','stable_gene_cohort_directions','rank_svm_kernel_distribution','rank_svm_loco_kernel_distribution','fixed_feature_control_metrics']
    d={name:pd.read_csv(args.data_dir/f'{name}.csv') for name in inputs}; validate(d,args.output_dir); style()
    for fn in [figure1,figure2,figure3,figure4,figure5,supplementary_fixed_feature_control]: fn(d,args.output_dir); print(fn.__name__,'saved',flush=True)
    print(args.output_dir)

if __name__=='__main__': main()
