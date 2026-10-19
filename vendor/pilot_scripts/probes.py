import subprocess, re, sys, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
ABC='/home/claude/pilot/abc_src/abc'; ansi=re.compile(r'\x1b\[[0-9;]*m')
OPS={'rw':'rewrite','rwz':'rewrite -z','rf':'refactor','rfz':'refactor -z','rs':'resub','rsz':'resub -z','b':'balance'}
sel=pd.read_csv('out/selected.csv')
def stats(path,cmds):
    rd='read_bench' if path.endswith('.bench') else 'read_blif'
    s=f'{rd} {path}; strash; '+'; '.join(cmds)+('; ' if cmds else '')+'print_stats'
    o=ansi.sub('',subprocess.run([ABC,'-c',s],capture_output=True,text=True,timeout=300).stdout)
    m=re.search(r'and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)',o); return int(m.group(1)),int(m.group(2))
def probe(c):
    n0,l0=stats(c['path'],[]); r={'circuit':c['circuit']}
    for k,v in OPS.items():
        n,l=stats(c['path'],[v]); r[f'p_{k}_nodes']=n/n0; r[f'p_{k}_depth']=l/max(l0,1)
    # two-step probes capturing order effects
    for a,b in [('b','rw'),('rw','b'),('rf','b'),('b','rf')]:
        n,l=stats(c['path'],[OPS[a],OPS[b]]); r[f'p_{a}{b}_nodes']=n/n0; r[f'p_{a}{b}_depth']=l/max(l0,1)
    return r
with ThreadPoolExecutor(2) as ex: rows=list(ex.map(probe,sel.to_dict('records')))
pd.DataFrame(rows).to_csv('out/probes.csv',index=False); print(len(rows),'circuits probed')
