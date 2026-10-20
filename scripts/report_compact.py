"""Concise paper tables and prose, derived from the existing saved experiments."""
import json
import pandas as pd
from circuit_recipe.synthesis import ROOT
from circuit_recipe.evaluate import bootstrap
from report import table,write,pct,ci


def main():
    classical=pd.read_csv(ROOT/'results/p1/summary.csv')
    neural=pd.read_csv(ROOT/'results/neural/summary.csv')
    facts=json.loads((ROOT/'results/neural_facts.json').read_text())
    saved=pd.read_csv(ROOT/'results/p1/selections.csv')
    reference=saved[(saved.model=='adaptive')&(saved.selector=='learned')&(saved.k==1)]

    def get(frame,name,selector='learned',k=1):
        return frame[(frame.model==name)&(frame.selector==selector)&(frame.k==k)].iloc[0]

    rows=[]
    for frame,name,label,selector in [
        (neural,'nn_full','NN: 50 circuit features','learned'),
        (neural,'nn_static','NN: 24 graph features','learned'),
        (classical,'ridge','Additive ridge','learned'),
        (classical,'ridge_interactions','Interaction ridge','learned'),
        (classical,'extra_trees','Extra-Trees','learned'),
        (classical,'random_forest','Random Forest','learned'),
        (classical,'boosting','Gradient boosting','learned'),
        (classical,'adaptive','Adaptive classical selector','learned'),
        (classical,'adaptive','Adaptive + margin gate','margin'),
        (classical,'adaptive','Uniform random (expected)','random'),
    ]:
        r=get(frame,name,selector)
        rows.append([label,f'{r.improvement:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',f'{r.regret:.2f}'])
    for label,col in [('resyn2','resyn2_J'),('Best fixed training recipe','fixed_J'),('Exhaustive-search oracle','oracle_J')]:
        gain=(100*(reference.fixed_J-reference[col])/reference.fixed_J).groupby(reference.circuit).mean()
        regret=(100*(reference[col]-reference.oracle_J)/reference.oracle_J).groupby(reference.circuit).mean()
        lo,hi=bootstrap(gain)
        rows.append([label,f'{gain.mean():.2f}',f'[{lo:.2f}, {hi:.2f}]',f'{regret.mean():.2f}'])
    table('compact_models.tex',['One-recipe selector',r'Gain (\%)',r'95\% interval',r'Gap (\%)'],rows)

    runtime=pd.read_csv(ROOT/'results/neural_runtime/summary.csv')
    rows=[]
    for name,k,label in [('global',1,'Best fixed recipe'),('nn_full',1,'NN: 50 features'),
                         ('nn_static',1,'NN: 24 features'),('global',5,'Global top five'),
                         ('nn_full',5,'NN: top five'),('nn_static',5,'Graph-only NN: top five'),
                         ('global',8,'Global top eight'),('exhaustive',120,'Exhaustive search')]:
        t=runtime[(runtime.model==name)&(runtime.k==k)].iloc[0]
        gap=0. if name=='exhaustive' else get(neural,'nn_full' if name=='global' else name,
                                             'global' if name=='global' else 'learned',k).regret
        rows.append([label,k,f'{gap:.2f}',f'{t.cost_s:.2f}',f'{t.ratio_of_mean_times:.1f}'])
    table('compact_budget.tex',['Selector','Runs',r'Gap (\%)','Time (s)',r'Speedup ($\times$)'],rows)

    full=get(neural,'nn_full');static=get(neural,'nn_static');five=get(neural,'nn_full',k=5)
    global5=get(neural,'nn_full','global',5);et=get(classical,'extra_trees')
    write('compact_quality.tex',
          f'One full-NN recommendation gave {pct(full.improvement)} mean gain over the best fixed recipe '
          f'(95\\% interval {ci(full.ci_low,full.ci_high)}); the graph-only NN gave {pct(static.improvement)} '
          f'({ci(static.ci_low,static.ci_high)}). Their average quality is close to the fixed recipe, '
          'with uncertainty about whether either improves it. '
          f'Extra-Trees gave the strongest one-recipe mean gain, {pct(et.improvement)} '
          f'({ci(et.ci_low,et.ci_high)}), while the inner-selected classical family gave \\PrimaryGain.\n\n'
          f'Executing the full NN\'s top five recommendations improved the fixed recipe by {pct(five.improvement)} '
          f'(95\\% interval {ci(five.ci_low,five.ci_high)}) and reduced mean oracle gap to {pct(five.regret)}. '
          f'It found an optimum among the 120 recipes on {pct(facts["metrics"]["nn_full_5"]["oracle_hit_pct"])} '
          f'of held-out predictions and came within 5\\% on {pct(facts["metrics"]["nn_full_5"]["within_five_pct"])}. '
          f'The global top-five baseline reached {pct(global5.regret)} gap. The NN\'s mean gain over that shortlist '
          f'was {pct(facts["nn_vs_global_top5_gain"])} ({ci(*facts["nn_vs_global_top5_ci"])}).\n\n'
          f'The full NN\'s one-recipe choices reduced mean area by {pct(facts["components"]["1"]["area_gain"])} '
          f'and increased mean delay by {pct(-facts["components"]["1"]["delay_gain"])} relative to the fixed recipe. '
          f'With five recommendations, mean area and delay fell by {pct(facts["components"]["5"]["area_gain"])} '
          f'and {pct(facts["components"]["5"]["delay_gain"])}. The combined objective permits these trade-offs.')

    t1=runtime[(runtime.model=='nn_full')&(runtime.k==1)].iloc[0]
    t5=runtime[(runtime.model=='nn_full')&(runtime.k==5)].iloc[0]
    tg=runtime[(runtime.model=='global')&(runtime.k==5)].iloc[0]
    ts=runtime[(runtime.model=='nn_static')&(runtime.k==1)].iloc[0]
    write('compact_compute.tex',
          f'Exhaustive search averaged {t1.exhaustive_s:.2f} seconds per timing circuit. '
          f'One full-NN recommendation took {t1.cost_s:.2f} seconds, including features and synthesis: '
          f'{t1.ratio_of_mean_times:.1f} times faster, saving {pct(t1.time_saved_pct)} of elapsed time. '
          f'Its top five took {t5.cost_s:.2f} seconds, a {t5.ratio_of_mean_times:.1f}-fold speedup '
          f'and {pct(t5.time_saved_pct)} saving. The graph-only NN used {ts.cost_s:.2f} seconds for one recipe.\n\n'
          f'Global top-five search took {tg.cost_s:.2f} seconds and achieved similar quality. '
          'Most savings therefore come from executing fewer recipes. Neural selection adds circuit-specific ranking '
          'and feature-collection cost. The cheapest option remains the best fixed recipe. '
          'These speedups describe serial search with a loaded predictor; parallel search and a fresh model launch have different elapsed costs.\n\n'
          f'Building the training table took \\SweepMinutes\\ minutes with five ABC workers. '
          f'NN evaluation and inner tuning took {facts["evaluation_seconds"]:.1f} seconds; an outer-fold NN refit '
          f'averaged {facts["outer_refit_mean_seconds"]:.2f} seconds after selecting its settings. '
          'The table is a shared offline cost for the fixed baseline and learned models. A new circuit needs only '
          'feature extraction, prediction, and the selected recipe runs.')

    # Retain every earlier ablation, transfer and sensitivity setting in one appendix table.
    groups=[]
    repeat=saved[(saved.model=='boosting')&(saved.repeat==0)&(saved.selector=='learned')&(saved.k==1)]
    per=repeat.groupby('circuit')[['improvement','regret']].mean();lo,hi=bootstrap(per.improvement)
    ablations=[['All features; centered target',per.improvement.mean(),lo,hi,per.regret.mean()]]
    for directory,label in [('features_static','Static graph features'),('features_mapping','Static + mapping'),
                            ('features_probes','Probe features alone'),('target_raw','Raw objective target'),
                            ('target_rank','Within-circuit rank target'),('encoding_counts','Operation counts'),
                            ('encoding_positions','Counts + positions')]:
        r=get(pd.read_csv(ROOT/'results'/directory/'summary.csv'),'boosting')
        ablations.append([label,r.improvement,r.ci_low,r.ci_high,r.regret])
    groups.append(('Ablations: boosting, first five-fold repeat',ablations))
    r=get(pd.read_csv(ROOT/'results/p2/summary.csv'),'adaptive')
    transfer=[['MCNC to other families (26)',r.improvement,r.ci_low,r.ci_high,r.regret]]
    p3=pd.read_csv(ROOT/'results/p3/selections.csv')
    for family,label in [('EPFL-arith','EPFL arithmetic held out (7)'),('EPFL-ctrl','EPFL control held out (9)'),
                         ('ISCAS85','ISCAS-85 held out (10)'),('MCNC','MCNC held out (179)')]:
        g=p3[(p3.model=='adaptive')&(p3.selector=='learned')&(p3.k==1)&(p3.family==family)]
        lo,hi=bootstrap(g.improvement)
        transfer.append([label,g.improvement.mean(),lo,hi,g.regret.mean()])
    groups.append(('Family transfer: adaptive classical selector',transfer))
    sensitivity=[]
    for directory,label in [('delay','Delay-only objective'),('area','Area-only objective'),
                            ('library_30','30-recipe library'),('library_60','60-recipe library'),
                            ('train_25','25 training circuits'),('train_50','50 training circuits'),
                            ('train_100','100 training circuits'),('unseen_recipes','24 recipes withheld from training')]:
        r=get(pd.read_csv(ROOT/'results'/directory/'summary.csv'),'boosting')
        sensitivity.append([label,r.improvement,r.ci_low,r.ci_high,r.regret])
    groups.append(('Sensitivity: boosting, first five-fold repeat',sensitivity))
    lines=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
           r'\textbf{Setting} & \textbf{Gain (\%)} & \textbf{95\% interval} & \textbf{Gap (\%)} \\']
    for title,entries in groups:
        lines.extend([r'\midrule',r'\multicolumn{4}{@{}l}{\textit{'+title+r'}} \\'])
        lines.extend(f'{label} & {gain:.2f} & [{lo:.2f}, {hi:.2f}] & {gap:.2f} '+r'\\'
                     for label,gain,lo,hi,gap in entries)
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    write('compact_secondary.tex','\n'.join(lines))


if __name__=='__main__':main()
