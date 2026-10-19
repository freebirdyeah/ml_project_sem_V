"""Generate the neural study's paper text, tables, and runtime comparisons."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from circuit_recipe.synthesis import ROOT
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import bootstrap,summarize
from report import table,write,macro,pct,ci,style,FIG
import matplotlib.pyplot as plt

def main():
    base=ROOT/'results/neural';runtime=ROOT/'results/neural_runtime'
    if not (base/'COMPLETE').exists() or not (runtime/'COMPLETE').exists():
        raise RuntimeError('Finish run_neural.py and benchmark_neural_runtime.py before reporting.')
    d=load_dataset(ROOT/'data/processed/dataset.csv');s=summarize(base)
    predictions=pd.read_csv(base/'selections.csv')
    def get(name,k=1,selector='learned'):
        return s[(s.model==name)&(s.selector==selector)&(s.k==k)].iloc[0]
    full=get('nn_full');static=get('nn_static');five=get('nn_full',5);static5=get('nn_static',5)
    global5=get('nn_full',5,'global');fixed=get('nn_full',1,'global')
    rows=[]
    for label,r in [('NN, 50 circuit features',full),('NN, 24 graph features',static),
                    ('NN, 50 features; top five',five),('NN, 24 features; top five',static5),
                    ('Global top five',global5),('Best fixed recipe',fixed)]:
        rows.append([label,int(r.k),f'{r.improvement:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',f'{r.regret:.2f}'])
    primary=predictions[(predictions.model=='nn_full')&(predictions.selector=='learned')&(predictions.k==1)]
    oracle=primary.groupby('circuit').apply(lambda g:float((100*(g.fixed_J-g.oracle_J)/g.fixed_J).mean()),include_groups=False)
    lo,hi=bootstrap(oracle)
    rows.append(['Exhaustive search',120,f'{oracle.mean():.2f}',f'[{lo:.2f}, {hi:.2f}]','0.00'])
    table('neural_results.tex',['Selector','Runs',r'Gain (\%)',r'95\% interval',r'Regret (\%)'],rows)
    metrics={}
    for name in ['nn_full','nn_static']:
        for k in [1,2,3,5,8]:
            g=predictions[(predictions.model==name)&(predictions.selector=='learned')&(predictions.k==k)]
            per=g.groupby('circuit')[['improvement','regret','cost_s','oracle_hit']].mean()
            metrics[f'{name}_{k}']=dict(gain=float(per.improvement.mean()),ci=bootstrap(per.improvement),
                 regret=float(per.regret.mean()),oracle_hit_pct=float(100*per.oracle_hit.mean()),
                 within_one_pct=float(100*g.regret.le(1).mean()),within_five_pct=float(100*g.regret.le(5).mean()),
                 estimated_sweep_time_s=float(per.cost_s.mean()))
    ng=predictions[(predictions.model=='nn_full')&(predictions.selector=='learned')&(predictions.k==5)]
    gg=predictions[(predictions.model=='nn_full')&(predictions.selector=='global')&(predictions.k==5)]
    paired=ng.merge(gg,on=['circuit','repeat','fold'],suffixes=('_nn','_global'))
    delta=(100*(paired.J_global-paired.J_nn)/paired.J_global).groupby(paired.circuit).mean()
    delta_ci=bootstrap(delta)
    lookup=d.set_index(['circuit','recipe_id'])[['yA','yD']]
    baseline=predictions[(predictions.model=='nn_full')&(predictions.selector=='global')&(predictions.k==1)]
    baseline=baseline.join(lookup,on=['circuit','selected_recipe'])[['circuit','repeat','fold','yA','yD']]
    components={}
    for k in [1,5]:
        selected=predictions[(predictions.model=='nn_full')&(predictions.selector=='learned')&(predictions.k==k)]
        selected=selected.join(lookup,on=['circuit','selected_recipe']).merge(baseline,on=['circuit','repeat','fold'],suffixes=('_nn','_fixed'))
        a=(100*(selected.yA_fixed-selected.yA_nn)/selected.yA_fixed).groupby(selected.circuit).mean()
        delay=(100*(selected.yD_fixed-selected.yD_nn)/selected.yD_fixed).groupby(selected.circuit).mean()
        components[str(k)]=dict(area_gain=float(a.mean()),delay_gain=float(delay.mean()))
    write('neural_interpretation.tex',
          f'With one complete recipe execution, the 50-feature network gave {pct(full.improvement)} mean improvement over the best fixed training recipe, '
          f'with a 95\\% interval of {ci(full.ci_low,full.ci_high)}. The 24-feature network gave {pct(static.improvement)} '
          f'with interval {ci(static.ci_low,static.ci_high)}. Both point estimates are close to the fixed-recipe result; '
          'the intervals include zero, leaving the direction of the average difference uncertain. '
          f'For the 50-feature network, mean area fell by {pct(components["1"]["area_gain"])} '
          f'and mean delay rose by {pct(-components["1"]["delay_gain"])} relative to the fixed recipe.\n\n'
          f'Running the full network\'s five highest-ranked recipes gave {pct(five.improvement)} improvement over the fixed recipe '
          f'and {pct(five.regret)} mean oracle regret. It reached the exact library optimum on '
          f'{pct(metrics["nn_full_5"]["oracle_hit_pct"])} of held-out predictions and came within 5\\% of it on '
          f'{pct(metrics["nn_full_5"]["within_five_pct"])}. These rates average the three held-out predictions per circuit. '
          f'The corresponding mean area and delay reductions were {pct(components["5"]["area_gain"])} and {pct(components["5"]["delay_gain"])}.\n\n'
          f'The global top-five baseline improved the fixed-recipe objective by {pct(global5.improvement)} with {pct(global5.regret)} regret. '
          f'The NN top-five result improved this global shortlist by {pct(delta.mean())} on average '
          f'(95\\% interval {ci(*delta_ci)}). This interval also includes zero. '
          'The experiments show how a small candidate budget recovers much of the available quality; the global shortlist remains a strong inexpensive alternative.')
    rt=pd.read_csv(runtime/'summary.csv');rtrows=[]
    def rtget(name,k):return rt[(rt.model==name)&(rt.k==k)].iloc[0]
    for name,k,label in [('global',1,'Best fixed recipe'),('nn_full',1,'NN, 50 features'),('nn_static',1,'NN, 24 features'),
                         ('global',5,'Global top five'),('nn_full',5,'NN, 50 features; top five'),
                         ('nn_static',5,'NN, 24 features; top five'),('global',8,'Global top eight'),('exhaustive',120,'Exhaustive search')]:
        r=rtget(name,k)
        rtrows.append([label,k,f'{r.cost_s:.3f}',f'{r.ratio_of_mean_times:.1f}',f'{r.time_saved_pct:.1f}'])
    table('neural_runtime.tex',['Selector','Runs','Time (s)',r'Speedup ($\times$)',r'Saved (\%)'],rtrows)
    t1=rtget('nn_full',1);t5=rtget('nn_full',5);ts=rtget('nn_static',1);tg=rtget('global',5)
    measurements=pd.read_csv(runtime/'measurements.csv')
    full_timing=[];static_timing=[];epochs={}
    for group in ['full','static']:
        tuning=[json.loads(p.read_text()) for p in sorted(base.glob(f'r*_f*_nn_{group}_tuning.json'))]
        epochs[group]=dict(Counter(str(t['chosen']['params']['epochs']) for t in tuning))
        values=[t['refit_seconds'] for t in tuning]
        if group=='full':full_timing=values
        else:static_timing=values
    offline=json.loads((ROOT/'data/processed/manifest.json').read_text())
    neural_seconds=json.loads((base/'COMPLETE').read_text())['wall_seconds']
    write('neural_runtime_interpretation.tex',
          f'Across the 12 timing circuits, exhaustive search took {t1.exhaustive_s:.2f} seconds per circuit on average. '
          f'One full-network recommendation, including diagnostics and its synthesis run, took {t1.cost_s:.2f} seconds: '
          f'{t1.ratio_of_mean_times:.1f} times faster and {pct(t1.time_saved_pct)} less elapsed time. '
          f'Executing five recommendations took {t5.cost_s:.2f} seconds, giving a {t5.ratio_of_mean_times:.1f}-fold speedup '
          f'and {pct(t5.time_saved_pct)} time saving. The graph-only network with one executed recipe took '
          f'{ts.cost_s:.2f} seconds ({ts.ratio_of_mean_times:.1f}-fold speedup).\n\n'
          f'The global top-five baseline took {tg.cost_s:.2f} seconds. It avoids the eleven diagnostic probes '
          'and also reduces exhaustive-search cost substantially. Table~\\ref{tab:nn-runtime} therefore separates '
          'the savings from executing fewer candidates from the additional cost of neural selection. '
          'The speedups divide mean exhaustive time by mean selector time on the same 12 circuits. '
          'They describe serial execution with a loaded model; parallel exhaustive search would have a different elapsed-time trade-off.\n\n'
          f'The training table was built once in {offline["wall_seconds"]/60:.1f} minutes with five ABC workers. '
          f'The complete NN evaluation, including both input variants, three repeats, and inner tuning, took {neural_seconds:.1f} seconds. '
          f'Refitting the full network after settings were chosen took {np.mean(full_timing):.3f} seconds per outer fold on average. '
          'These are development costs. A new circuit uses the saved model and pays only feature collection, prediction, '
          'and its chosen recipe runs. The measured search table is shared by the fixed-recipe baseline and the NN.')
    style();fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for name,label,color in [('nn_full','NN: 50 features','#356b96'),('nn_static','NN: 24 features','#34775b'),('global','Global shortlist','#bb5534')]:
        q=s[(s.model=='nn_full')&(s.selector=='global')] if name=='global' else s[(s.model==name)&(s.selector=='learned')]
        axes[0].plot(q.k,q.regret,'o-',label=label,color=color,markersize=3)
        q=rt[rt.model==name].sort_values('k')
        axes[1].plot(q.cost_s,q.regret,'o-',label=label,color=color,markersize=3)
        if name=='nn_full':
            for _,r in q[q.k.isin([1,5,8])].iterrows():
                offset=(3,-10) if r.k==8 else (3,5)
                axes[1].annotate(str(int(r.k)),(r.cost_s,r.regret),xytext=offset,textcoords='offset points',fontsize=8,color=color)
    exhaustive=rtget('exhaustive',120)
    axes[1].scatter([exhaustive.cost_s],[0],marker='*',s=65,color='#444444',label='Exhaustive: 120')
    axes[0].set(xlabel='Complete recipes executed',ylabel='Mean oracle regret (%)',xticks=[1,2,3,5,8],title='All 205 circuits: held-out quality')
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(xlabel='Serial elapsed time (s; log scale)',ylabel='Mean oracle regret (%)',xscale='log',title='12 timing circuits: quality and time')
    axes[1].legend(frameon=False,fontsize=7,loc='upper right')
    fig.tight_layout();fig.savefig(FIG/'neural_tradeoff.pdf');plt.close(fig)
    numbers=dict(NNGain=pct(full.improvement),NNCI=ci(full.ci_low,full.ci_high),NNRegret=pct(full.regret),
                 NNStaticGain=pct(static.improvement),NNFiveGain=pct(five.improvement),NNFiveRegret=pct(five.regret),
                 NNOneSeconds=f'{t1.cost_s:.2f}',NNFiveSeconds=f'{t5.cost_s:.2f}',ExhaustiveSeconds=f'{t1.exhaustive_s:.2f}',
                 NNOneSpeedup=f'{t1.ratio_of_mean_times:.1f}',NNFiveSpeedup=f'{t5.ratio_of_mean_times:.1f}',
                 NNOneSaved=pct(t1.time_saved_pct),NNFiveSaved=pct(t5.time_saved_pct))
    write('neural_numbers.tex','\n'.join(macro(k,v) for k,v in numbers.items()))
    write('neural_conclusion.tex',
          f'A small neural network can select from a fixed library of 120 synthesis recipes using measurements from earlier circuits. '
          f'On 205 held-out benchmark circuits, executing one recommendation gave {pct(full.improvement)} mean improvement over '
          f'the best fixed training recipe, with interval {ci(full.ci_low,full.ci_high)}. '
          f'This places its average objective near the fixed-recipe baseline. Executing five recommendations improved that baseline '
          f'by {pct(five.improvement)} and reduced mean regret to {pct(five.regret)}.\n\n'
          f'On the separate 12-circuit serial timing audit, one recommendation was {t1.ratio_of_mean_times:.1f} times faster than '
          f'exhaustive search, and five recommendations were {t5.ratio_of_mean_times:.1f} times faster, including feature collection. '
          'The gain comes from replacing most full synthesis runs with a prediction. '
          'A global shortlist offers a similar quality--compute trade-off with less selection overhead, and Extra-Trees achieved '
          'a stronger one-recipe average improvement in the model comparison. These results make the budget explicit: '
          'a fixed recipe gives the cheapest decision, a short list gives better measured quality, and exhaustive evaluation gives the library optimum.')
    facts=dict(dataset_circuits=205,recipes=120,primary_model='nn_full',metrics=metrics,components=components,
               nn_vs_global_top5_gain=float(delta.mean()),nn_vs_global_top5_ci=delta_ci,
               timing_circuits=12,timing=rt.to_dict('records'),epochs_chosen=epochs,
               evaluation_seconds=neural_seconds,outer_refit_mean_seconds=float(np.mean(full_timing)))
    (ROOT/'results/neural_facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(dict(nn_gain=full.improvement,nn_ci=[full.ci_low,full.ci_high],
                          nn_top5_gain=five.improvement,nn_top5_regret=five.regret,
                          serial_one_speedup=t1.ratio_of_mean_times,serial_five_speedup=t5.ratio_of_mean_times),indent=2))

if __name__=='__main__':main()
