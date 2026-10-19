import subprocess, csv, os, re, sys, time, json, random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0,'/home/claude/pilot/scripts')
from features import features
ABC='/home/claude/pilot/abc_src/abc'; LIB='/home/claude/pilot/nangate.lib'
ansi=re.compile(r'\x1b\[[0-9;]*m')
def rd(path): return 'read_bench' if path.endswith('.bench') else 'read_blif'
def run_abc(path, recipe_cmd, timeout):
    body = f'read_lib -w {LIB}; {rd(path)} {path}; strash; '
    if recipe_cmd: body += recipe_cmd + '; '
    body += 'print_stats; map; topo; stime'
    t=time.time()
    try:
        o=subprocess.run([ABC,'-c',body],capture_output=True,text=True,timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return dict(status='timeout',runtime_s=time.time()-t)
    rt=time.time()-t; o=ansi.sub('',o)
    m=re.search(r'and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)',o)
    s=re.search(r'Area\s*=\s*([\d.]+).*?Delay\s*=\s*([\d.]+)\s*ps',o)
    if not (m and s): return dict(status='parsefail',runtime_s=rt)
    return dict(status='ok',aig_nodes=int(m.group(1)),aig_depth=int(m.group(2)),
                area=float(s.group(1)),delay=float(s.group(2)),runtime_s=rt)
def do_circuit(c, recipes, timeout, workdir):
    # features from the strashed AIG
    aig=f'{workdir}/{c["circuit"]}.aig'
    subprocess.run([ABC,'-c',f'{rd(c["path"])} {c["path"]}; strash; write_aiger {aig}'],capture_output=True,timeout=120)
    feats=features(aig); os.remove(aig)
    base=run_abc(c['path'],None,timeout)
    rows=[]
    for r in recipes:
        res=run_abc(c['path'],r['abc'],timeout)
        row=dict(family=c['family'],circuit=c['circuit'],**{f'c_{k}':v for k,v in feats.items()},
                 recipe_id=r['recipe_id'],recipe_kind=r['kind'],recipe_ops=r['ops'],
                 base_area=base.get('area'),base_delay=base.get('delay'),
                 base_aig_nodes=base.get('aig_nodes'),base_aig_depth=base.get('aig_depth'),**res)
        rows.append(row)
    return rows
if __name__=='__main__':
    sel=pd.read_csv(sys.argv[1]); out=sys.argv[2]; timeout=int(sys.argv[3]) if len(sys.argv)>3 else 120
    nrec=int(sys.argv[4]) if len(sys.argv)>4 else 10**9
    recipes=pd.read_csv('out/recipes.csv').to_dict('records')[:nrec]
    wd='/tmp/claude-0/aigs'; os.makedirs(wd,exist_ok=True)
    allrows=[]; t0=time.time()
    def work(c):
        rows=do_circuit(c,recipes,timeout,wd)
        print(f'{c["circuit"]:14s} {len(rows)} runs  elapsed {time.time()-t0:.0f}s',flush=True); return rows
    with ThreadPoolExecutor(2) as ex:
        for rows in ex.map(work, sel.to_dict('records')): allrows+=rows
    pd.DataFrame(allrows).to_csv(out,index=False)
