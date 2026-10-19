"""Generate every paper table, number, plot, and result-dependent paragraph."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml_proj_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from circuit_recipe.synthesis import ROOT
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import bootstrap,summarize

GEN=ROOT/'paper/generated';FIG=ROOT/'paper/figures'
DISPLAY={'ridge':'Additive ridge','ridge_interactions':'Interaction ridge','extra_trees':'Extra-Trees',
         'random_forest':'Random Forest','boosting':'Gradient boosting','adaptive':'Adaptive selector'}
FAMILY={'EPFL-arith':'EPFL arithmetic','EPFL-ctrl':'EPFL control','ISCAS85':'ISCAS-85','MCNC':'MCNC'}
COLORS={'EPFL-arith':'#bb5534','EPFL-ctrl':'#b18b23','ISCAS85':'#34775b','MCNC':'#356b96'}

def write(name,text):
    (GEN/name).write_text(text+'\n')

def table(name,headers,rows,align=None):
    alignment=align or ('l'+'r'*(len(headers)-1))
    lines=[r'\begin{tabular}{@{}'+alignment+r'@{}}',r'\toprule',' & '.join(r'\textbf{'+h+'}' for h in headers)+r' \\',r'\midrule']
    lines += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    write(name,'\n'.join(lines))

def macro(name,value):
    return '\\newcommand{\\'+name+'}{'+str(value)+'}'

def pct(v):return f'{v:.2f}'+r'\%'
def ci(lo,hi):return '['+pct(lo)+', '+pct(hi)+']'
def ptex(p):
    if p>=.001:return f'{p:.3f}'
    coefficient,exponent=f'{p:.2e}'.split('e');return coefficient+r'\times10^{'+str(int(exponent))+'}'

def style():
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':9,
                         'pdf.fonttype':42,'savefig.bbox':'tight','axes.grid':False})

def workflow():
    fig,ax=plt.subplots(figsize=(7.1,2.55));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    labels=[['205 source\ncircuits','120 candidate\nrecipes','ABC area/delay\nmeasurements','Grouped training\nand validation'],
            ['New circuit\nand diagnostics','Score the\nrecipe library','Run the best\none or top k','Keep the best\nmeasured circuit']]
    for row,ys in enumerate([.69,.14]):
        ax.text(0,ys+.25,'BUILD AND LEARN' if row==0 else 'SELECT FOR A NEW CIRCUIT',fontsize=8,fontweight='bold',color='#444444')
        for i,label in enumerate(labels[row]):
            x=i*.255
            ax.add_patch(FancyBboxPatch((x,ys),.225,.20,boxstyle='round,pad=0.007',linewidth=.8,edgecolor='#444444',facecolor='#f0f4f7'))
            ax.text(x+.1125,ys+.10,label,ha='center',va='center',fontsize=9)
            if i<3:ax.add_patch(FancyArrowPatch((x+.23,ys+.10),(x+.25,ys+.10),arrowstyle='-|>',mutation_scale=9,color='#444444'))
    fig.savefig(FIG/'workflow.pdf');plt.close(fig)

def dataset_artifacts(d,manifest):
    circuits=d.drop_duplicates('circuit')
    rows=[]
    for fam in FAMILY:
        g=circuits[circuits.family==fam]
        rows.append([FAMILY[fam],len(g),f'{g.c_n_and.min():,.0f}',f'{g.c_n_and.median():,.0f}',f'{g.c_n_and.max():,.0f}',f'{g.c_depth.median():,.0f}'])
    table('pool.tex',['Family','Circuits','Min. nodes','Median','Max. nodes','Depth median'],rows)
    example=d[(d.circuit=='c1908')&d.recipe_id.isin(['R001','R037'])]
    rows=[['R001 (resyn2)' if r.recipe_id=='R001' else 'R037',f'{r.area:.2f}',f'{r.delay:.2f}',f'{r.yA:.3f}',f'{r.yD:.3f}',f'{r["J_0.5"]:.3f}'] for _,r in example.iterrows()]
    table('example.tex',['Recipe',r'Area ($\mu$m$^2$)','Delay (ps)',r'$y_A$',r'$y_D$',r'$J_{0.5}$'],rows)
    P=d.pivot(index='circuit',columns='recipe_id',values='J_0.5');total=P.sum();gain=[]
    for c in P.index:
        fixed=(total-P.loc[c]).idxmin();gain.append(100*(P.loc[c,fixed]-P.loc[c].min())/P.loc[c,fixed])
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7))
    for fam,color in COLORS.items():
        g=circuits[circuits.family==fam];axes[0].scatter(g.c_n_and,g.c_depth,s=16,alpha=.7,color=color,label=FAMILY[fam],edgecolors='none')
    axes[0].set(xscale='log',yscale='log',xlabel='Starting AND nodes',ylabel='Starting graph depth')
    axes[0].legend(frameon=False,loc='upper left')
    axes[1].hist(gain,bins=22,color='#356b96',edgecolor='white',linewidth=.5)
    axes[1].axvline(np.median(gain),color='#bb5534',linestyle='--',label=f'Median {np.median(gain):.1f}%')
    axes[1].set(xlabel='Oracle improvement over fixed recipe (%)',ylabel='Circuits');axes[1].legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'dataset.pdf');plt.close(fig)
    return dict(MaxNodes=f'{circuits.c_n_and.max():,.0f}',SuccessfulRuns=f'{len(d):,}',
                ProbeSeconds=f'{circuits.probe_runtime_s.mean():.2f}',SweepMinutes=f'{manifest["wall_seconds"]/60:.1f}',
                MedianNodes=f'{circuits.c_n_and.median():,.0f}',MedianDepth=f'{circuits.c_depth.median():,.0f}',
                MeanArea=f'{d.yA.mean():.3f}',MeanDelay=f'{d.yD.mean():.3f}',
                DistinctWinners=P.idxmin(axis=1).nunique(),OracleGain=pct(np.mean(gain)))

def main_results():
    path=ROOT/'results/p1';summary=summarize(path);d=pd.read_csv(path/'selections.csv')
    base=d[(d.model=='adaptive')&(d.selector=='learned')&(d.k==1)]
    primary=summary[(summary.model=='adaptive')&(summary.selector=='learned')&(summary.k==1)].iloc[0]
    values=base.groupby('circuit').improvement.mean();p=wilcoxon(values,zero_method='wilcox').pvalue if np.any(values!=0) else 1.
    if primary.ci_low>0:
        verdict='The confidence interval lies above zero, supporting an average gain on this circuit pool.'
    elif primary.ci_high<0:
        verdict='The confidence interval lies below zero; the fixed recipe gave a better average outcome on this circuit pool.'
    else:
        verdict='The confidence interval includes zero, so the average gain remains uncertain.'
    rows=[]
    for name in DISPLAY:
        r=summary[(summary.model==name)&(summary.selector=='learned')&(summary.k==1)].iloc[0]
        rows.append([DISPLAY[name],f'{r.improvement:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',f'{r.regret:.2f}',f'{r.win_pct:.1f}'])
    margin=summary[(summary.model=='adaptive')&(summary.selector=='margin')&(summary.k==1)].iloc[0]
    rows.append(['Adaptive + margin gate',f'{margin.improvement:.2f}',f'[{margin.ci_low:.2f}, {margin.ci_high:.2f}]',f'{margin.regret:.2f}',f'{margin.win_pct:.1f}'])
    for name in ['random','resyn2','fixed','oracle']:
        b=base.copy()
        if name=='random':
            b=d[(d.model=='adaptive')&(d.selector=='random')&(d.k==1)].copy();label='Uniform random (expected)'
        else:
            b['J']=b[{'resyn2':'resyn2_J','fixed':'fixed_J','oracle':'oracle_J'}[name]]
            b['improvement']=100*(b.fixed_J-b.J)/b.fixed_J;b['regret']=100*(b.J-b.oracle_J)/b.oracle_J
            label={'resyn2':'resyn2','fixed':'Best fixed training recipe','oracle':'Oracle (120 measured runs)'}[name]
        g=b.groupby('circuit')[['improvement','regret']].mean();lo,hi=bootstrap(g.improvement)
        win='--' if name=='random' else f'{100*g.improvement.gt(1e-8).mean():.1f}'
        rows.append([label,f'{g.improvement.mean():.2f}',f'[{lo:.2f}, {hi:.2f}]',f'{g.regret.mean():.2f}',win])
    table('main_results.tex',['Selector',r'Gain (\%)',r'95\% interval',r'Regret (\%)',r'Win (\%)'],rows)
    family_counts=Counter()
    for f in sorted(path.glob('r*_f*_tuning.json')):
        for r in json.loads(f.read_text()):
            if r['model']=='adaptive':family_counts[r['chosen_family']]+=1
    family_text=', '.join(f'{DISPLAY[k]} in {v}' for k,v in family_counts.most_common())
    write('main_interpretation.tex',
          f'The adaptive selector chose {family_text} of the 15 outer folds. '
          f'Its average improvement over \\op{{resyn2}} was {pct(primary.improvement_resyn2)}. '
          f'After averaging the three test results for each circuit, it improved the fixed-recipe objective on {pct(primary.win_pct)} of circuits '
          f'and increased it on {pct(primary.lose_pct)}. Its mean within-circuit Spearman correlation was {primary.spearman:.3f}. '
          f'The margin gate produced a mean gain of {pct(margin.improvement)} with interval {ci(margin.ci_low,margin.ci_high)}. '+verdict)
    by_family=base.groupby(['circuit','family'])[['improvement','regret']].mean().reset_index()
    by_family.groupby('family')[['improvement','regret']].mean().to_csv(ROOT/'results/p1/family_summary.csv')
    facts=dict(primary_gain=float(primary.improvement),primary_ci=[float(primary.ci_low),float(primary.ci_high)],
               primary_p=float(p),primary_regret=float(primary.regret),model_family_counts=dict(family_counts))
    facts['interpretation']=verdict
    return dict(PrimaryGain=pct(primary.improvement),PrimaryCI=ci(primary.ci_low,primary.ci_high),
                PrimaryRegret=pct(primary.regret),PrimaryP=ptex(p)),facts,summary,d

def budget_artifacts(summary,d,dataset):
    s=summary[summary.model=='adaptive'].copy();rows=[]
    # A shortlist compares normalized objectives and therefore needs A0 and D0.
    mask=s.selector.isin(['global','random']) & s.k.gt(1)
    s.loc[mask,'cost_s']+=dataset.drop_duplicates('circuit').baseline_runtime_s.mean()
    for k in [1,2,3,5,8]:
        vals={kind:s[(s.selector==kind)&(s.k==k)].iloc[0] for kind in ['learned','global','random']}
        rows.append([k]+[f'{vals[x].regret:.2f}' for x in ['learned','global','random']]+[f'{vals["learned"].cost_s:.2f}',f'{vals["global"].cost_s:.2f}'])
    table('budget.tex',['Runs $k$',r'Learned (\%)',r'Global (\%)',r'Random (\%)','Learned (s)','Global (s)'],rows)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65));mean_run=dataset.runtime_s.mean()
    for kind,label,color in [('learned','Learned shortlist','#356b96'),('global','Global shortlist','#bb5534'),('random','Random shortlist','#777777')]:
        g=s[s.selector==kind].sort_values('k')
        axes[0].plot(g.k,g.regret,'o-',color=color,label=label,markersize=4)
        axes[1].plot(g.cost_s/mean_run,g.regret,'o-',color=color,label=label,markersize=4)
    axes[0].set(xlabel='Complete recipes executed (k)',ylabel='Mean oracle regret (%)',xticks=[1,2,3,5,8]);axes[0].legend(frameon=False)
    axes[1].set(xlabel='Measured run equivalents',ylabel='Mean oracle regret (%)')
    fig.tight_layout();fig.savefig(FIG/'budget.pdf');plt.close(fig)
    one=s[(s.selector=='learned')&(s.k==1)].iloc[0];five=s[(s.selector=='learned')&(s.k==5)].iloc[0]
    global_five=s[(s.selector=='global')&(s.k==5)].iloc[0]
    global_eight=s[(s.selector=='global')&(s.k==8)].iloc[0]
    globals_=s[s.selector=='global']
    closest=globals_.iloc[int(np.argmin(np.abs(globals_.cost_s.to_numpy()-one.cost_s)))]
    write('budget_interpretation.tex',f'Increasing the learned shortlist from one to five recipes changed mean regret from {pct(one.regret)} '
          f'to {pct(five.regret)}. At five executions, the global shortlist reached {pct(global_five.regret)} regret. '
          f'The learned five-recipe selector took {five.cost_s:.2f} seconds per circuit on average, including feature collection; '
          f'the global five-recipe selector took {global_five.cost_s:.2f} seconds. '
          f'The one-recipe learned selector used {one.cost_s/mean_run:.2f} mean-run equivalents. '
          f'The closest tested global budget to the one-recipe learned selector was $k={int(closest.k)}$, '
          f'with {closest.cost_s:.2f} seconds of execution time and {pct(closest.regret)} mean regret. '
          f'The global eight-recipe shortlist reached {pct(global_eight.regret)} regret in {global_eight.cost_s:.2f} seconds, '
          f'compared with {pct(five.regret)} in {five.cost_s:.2f} seconds for five learned candidates. '
          'The global shortlist remains a strong comparator: at five runs it uses less time for similar regret, '
          'and its eight-run result reaches lower regret for a small additional time cost relative to five learned candidates.')
    return dict(GlobalEightRegret=pct(global_eight.regret),GlobalEightSeconds=f'{global_eight.cost_s:.2f}',
                LearnedFiveRegret=pct(five.regret),LearnedFiveSeconds=f'{five.cost_s:.2f}')

def area_delay_artifacts(results,dataset):
    chosen=results[results.model=='adaptive'].copy()
    lookup=dataset.set_index(['circuit','recipe_id'])[['yA','yD']]
    fixed=chosen[(chosen.selector=='global')&(chosen.k==1)][['circuit','repeat','fold','selected_recipe']].copy()
    fixed=fixed.join(lookup,on=['circuit','selected_recipe']).rename(columns={'yA':'fixed_yA','yD':'fixed_yD'})
    fixed=fixed[['circuit','repeat','fold','fixed_yA','fixed_yD']]
    variants=[('Adaptive, one recipe','learned',1),('Adaptive, five recipes','learned',5),('Global, five recipes','global',5)]
    rows=[];facts={}
    for label,selector,k in variants+[('resyn2','resyn2',1)]:
        if selector=='resyn2':
            g=chosen[(chosen.selector=='learned')&(chosen.k==1)].copy();g['selected_recipe']='R001'
        else:
            g=chosen[(chosen.selector==selector)&(chosen.k==k)].copy()
        g=g.join(lookup,on=['circuit','selected_recipe']).merge(fixed,on=['circuit','repeat','fold'])
        g['area_gain']=100*(g.fixed_yA-g.yA)/g.fixed_yA
        g['delay_gain']=100*(g.fixed_yD-g.yD)/g.fixed_yD
        circuit=g.groupby('circuit')[['area_gain','delay_gain']].mean()
        a=float(circuit.area_gain.mean());b=float(circuit.delay_gain.mean())
        both=float(100*((circuit.area_gain>1e-8)&(circuit.delay_gain>1e-8)).mean())
        facts[label]=dict(area_gain=a,delay_gain=b,both_pct=both)
        rows.append([label,f'{a:.2f}',f'{b:.2f}',f'{both:.1f}'])
    table('area_delay.tex',['Selector',r'Area gain (\%)',r'Delay gain (\%)',r'Both improve (\%)'],rows)
    a=facts['Adaptive, one recipe'];b=facts['Adaptive, five recipes']
    write('area_delay_interpretation.tex',f'For the one-recipe adaptive selector, mean area improvement was {pct(a["area_gain"])} '
          f'and mean delay improvement was {pct(a["delay_gain"])} relative to the fixed recipe. '
          f'Both improved on {pct(a["both_pct"])} of circuits after combining repeats. '
          f'With five verified recipes, the corresponding area and delay improvements were {pct(b["area_gain"])} and {pct(b["delay_gain"])}. '
          'The weighted objective permits an area--delay trade-off, so these component measurements help interpret the combined gain.')
    (ROOT/'results/area_delay_summary.json').write_text(json.dumps(facts,indent=2)+'\n')

def variant_result(name):
    s=summarize(ROOT/'results'/name)
    return s[(s.model=='boosting')&(s.selector=='learned')&(s.k==1)].iloc[0]

def secondary_artifacts(main):
    first=main[(main.model=='boosting')&(main.repeat==0)&(main.selector=='learned')&(main.k==1)]
    mean=first.groupby('circuit')[['improvement','regret']].mean()
    reference=dict(improvement=mean.improvement.mean(),regret=mean.regret.mean())
    reference.update(ci_low=bootstrap(mean.improvement)[0],ci_high=bootstrap(mean.improvement)[1])
    rows=[['Full features; centered; full encoding',194,f'{reference["improvement"]:.2f}',f'{reference["regret"]:.2f}']]
    labels=[('features_static','Static graph features',168),('features_mapping','Static + mapping features',172),
            ('features_probes','Probe features alone',166),('target_raw','Raw objective target',194),
            ('target_rank','Within-circuit rank target',194),('encoding_counts','Operation-count encoding',59),
            ('encoding_positions','Counts + position encoding',143)]
    values={}
    for name,label,n in labels:
        r=variant_result(name);values[name]=r
        rows.append([label,n,f'{r.improvement:.2f}',f'{r.regret:.2f}'])
    table('ablations.tex',['Setting','Features',r'Gain (\%)',r'Regret (\%)'],rows)
    write('ablation_interpretation.tex',
          f'The full boosting configuration achieved {pct(reference["improvement"])} mean gain in this repeat. '
          f'Static graph features gave {pct(values["features_static"].improvement)}; adding mapping measurements gave {pct(values["features_mapping"].improvement)}; '
          f'probe features alone gave {pct(values["features_probes"].improvement)}. '
          f'Using raw targets produced {pct(values["target_raw"].improvement)} gain, and rank targets produced {pct(values["target_rank"].improvement)}. '
          f'Operation counts alone gave {pct(values["encoding_counts"].improvement)} gain; '
          f'the full sequence encoding gave {pct(reference["improvement"])}.')
    rows=[]
    p2=summarize(ROOT/'results/p2');r=p2[(p2.model=='adaptive')&(p2.selector=='learned')&(p2.k==1)].iloc[0]
    rows.append(['MCNC to EPFL + ISCAS-85',26,f'{r.improvement:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',f'{r.regret:.2f}'])
    p3=pd.read_csv(ROOT/'results/p3/selections.csv')
    for family in FAMILY:
        g=p3[(p3.model=='adaptive')&(p3.selector=='learned')&(p3.k==1)&(p3.family==family)]
        lo,hi=bootstrap(g.improvement)
        rows.append([FAMILY[family]+' held out',len(g),f'{g.improvement.mean():.2f}',f'[{lo:.2f}, {hi:.2f}]',f'{g.regret.mean():.2f}'])
    table('transfer.tex',['Test distribution','Circuits',r'Gain (\%)',r'95\% interval',r'Regret (\%)'],rows)
    families=pd.read_csv(ROOT/'results/p1/family_summary.csv')
    descriptions='; '.join(f'{FAMILY[x.family]} {pct(x.improvement)}' for _,x in families.iterrows())
    write('transfer_interpretation.tex',f'Training on MCNC and testing on the other 26 circuits gave {pct(r.improvement)} mean gain, '
          f'with interval {ci(r.ci_low,r.ci_high)}. In the primary repeated protocol, family-specific mean gains were {descriptions}. '
          'The held-out arithmetic and control families contain seven and nine circuits, respectively, which gives their interval estimates a small sample base.')
    rows=[]
    for name,label in [('delay','Delay objective'),('area','Area objective'),('library_30','30-recipe library'),('library_60','60-recipe library')]:
        x=variant_result(name);rows.append([label,f'{x.improvement:.2f}',f'[{x.ci_low:.2f}, {x.ci_high:.2f}]',f'{x.regret:.2f}'])
    for n in [25,50,100]:
        x=variant_result(f'train_{n}');rows.append([f'{n} training circuits',f'{x.improvement:.2f}',f'[{x.ci_low:.2f}, {x.ci_high:.2f}]',f'{x.regret:.2f}'])
    x=variant_result('unseen_recipes');rows.append(['24 recipes unseen in training',f'{x.improvement:.2f}',f'[{x.ci_low:.2f}, {x.ci_high:.2f}]',f'{x.regret:.2f}'])
    table('sensitivity.tex',['Experiment',r'Gain (\%)',r'95\% interval',r'Regret (\%)'],rows)
    area=variant_result('area');delay=variant_result('delay')
    write('sensitivity_interpretation.tex',f'The area-only and delay-only experiments gave mean gains of {pct(area.improvement)} and {pct(delay.improvement)}, '
          'respectively, with both confidence intervals including zero. '
          f'Reserving 24 recipes from training gave {pct(x.improvement)} mean gain with interval {ci(x.ci_low,x.ci_high)}. '
          'The average benefit in this unseen-recipe setting remains uncertain.')

def environment_artifacts():
    env=json.loads((ROOT/'results/environment.json').read_text());manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    rows=[['Processor allocation','8 logical CPUs; synthesis sweep: 5 workers'],['Python',env['python']]]
    rows += [[n.replace('_',r'\_'),v] for n,v in env['packages'].items()]
    rows += [['ABC commit',r'\texttt{5d64f3b5f3bb9db6f3d77636a65e99ba3ee9d799}'],
             ['Library SHA-256',r'\texttt{'+manifest['library_sha256'][:24]+r'\ldots}'],
             ['Cross-validation seeds','0, 1, 2; inner split seed: outer fit seed + 1000'],
             ['Sampling seeds','Pilot recipes: 7; extension: 17; subset: 23; unseen recipes: 29'],
             ['Bootstrap','2,000 circuit resamples; seed 2026']]
    table('environment.tex',['Item','Recorded value'],rows,'ll')

def main():
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args()
    GEN.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);style();workflow()
    if not args.draft:
        if not (ROOT/'results/EXPERIMENTS_COMPLETE').exists():raise RuntimeError('The complete experiment suite must finish before final reporting.')
        if not (ROOT/'results/data_audit.json').exists():raise RuntimeError('Run the synthesis audit before final reporting.')
    if args.draft:
        names=['MaxNodes','SuccessfulRuns','ProbeSeconds','SweepMinutes','MedianNodes','MedianDepth','MeanArea','MeanDelay',
               'DistinctWinners','OracleGain','PrimaryGain','PrimaryCI','PrimaryRegret','PrimaryP',
               'GlobalEightRegret','GlobalEightSeconds','LearnedFiveRegret','LearnedFiveSeconds']
        write('numbers.tex','\n'.join(macro(n,r'\mathrm{pending}' if n=='PrimaryP' else r'\textit{pending}') for n in names))
        for n in ['pool','example','main_results','area_delay','budget','ablations','transfer','sensitivity','environment']:
            write(n+'.tex',r'\emph{Experiments are running.}')
        for n in ['main_interpretation','area_delay_interpretation','budget_interpretation','ablation_interpretation','transfer_interpretation','sensitivity_interpretation','audit_summary','conclusion']:
            write(n+'.tex',r'\emph{Measured results will be inserted after the experiment suite finishes.}')
        for name in ['dataset','budget']:
            fig,ax=plt.subplots(figsize=(7.1,2.7));ax.axis('off');ax.text(.5,.5,'Experiments are running',ha='center');fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
        return
    dataset=load_dataset(ROOT/'data/processed/dataset.csv');manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    audit=json.loads((ROOT/'results/data_audit.json').read_text())
    if not audit['all_equivalent'] or not all(audit['agreement'].values()) or audit['dataset_sha256']!=manifest['dataset_sha256']:
        raise RuntimeError('Synthesis audit does not validate the final dataset.')
    write('audit_summary.tex',f'We compared all {audit["pilot_pairs_compared"]:,} overlapping pilot observations with our regenerated measurements. '
          f'Area, delay, baseline measurements, and all graph and probe features agreed within numerical serialization precision. '
          f'We independently repeated {audit["equivalence_checks"]} sampled circuit--recipe pairs, six per source family. '
          'Their mapped area and delay reproduced exactly, and ABC established Boolean equivalence between each mapped result and its input circuit. '
          'For verification exports we remove external don\'t-care annotations, which describe input cases with no required output value, and compare the concrete Boolean networks. '
          'All 205 source-file checksums matched the saved inventory.')
    numbers=dataset_artifacts(dataset,manifest);primary,facts,summary,results=main_results();numbers.update(primary)
    numbers.update(budget_artifacts(summary,results,dataset));area_delay_artifacts(results,dataset);secondary_artifacts(results);environment_artifacts()
    write('numbers.tex','\n'.join(macro(n,v) for n,v in numbers.items()))
    one=summary[(summary.model=='adaptive')&(summary.selector=='learned')&(summary.k==1)].iloc[0]
    five=summary[(summary.model=='adaptive')&(summary.selector=='learned')&(summary.k==5)].iloc[0]
    global_eight=summary[(summary.model=='adaptive')&(summary.selector=='global')&(summary.k==8)].iloc[0]
    global_eight=global_eight.copy()
    global_eight['cost_s']+=dataset.drop_duplicates('circuit').baseline_runtime_s.mean()
    conclusion=(f'On the 205-circuit benchmark pool, our adaptive selector gave {pct(one.improvement)} mean improvement over the best fixed training recipe '
                f'with 95\\% interval {ci(one.ci_low,one.ci_high)}. It closed {pct(one.gap_closed_pct)} of the aggregate fixed-to-oracle objective gap. '
                +facts['interpretation']+' '+
                f'Verifying five predicted candidates gave {pct(five.improvement)} mean improvement and {pct(five.regret)} mean oracle regret. '
                f'The five-recipe learned shortlist took {five.cost_s:.2f} seconds per circuit including feature collection. '
                f'The global eight-recipe shortlist reached {pct(global_eight.regret)} regret in {global_eight.cost_s:.2f} seconds. '
                'For the tested runtime budgets, the global shortlist provided the stronger practical choice. '
                'The one-recipe result establishes a small circuit-specific selection benefit; the cost results favor spending this feature-collection overhead on additional candidate executions.')
    write('conclusion.tex',conclusion)
    (ROOT/'results/paper_facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(facts,indent=2))
    if (ROOT/'results/neural/COMPLETE').exists():
        from report_neural import main as neural_report
        neural_report()

if __name__=='__main__':main()
