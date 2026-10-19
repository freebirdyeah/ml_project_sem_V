import subprocess, glob, os, re, csv, sys
from concurrent.futures import ThreadPoolExecutor
ABC='/home/claude/pilot/abc_src/abc'
D='/home/claude/pilot/data'
cands=[]
for f in sorted(glob.glob(f'{D}/epfl/arithmetic/*.blif')): cands.append(('EPFL-arith',f))
for f in sorted(glob.glob(f'{D}/epfl/random_control/*.blif')): cands.append(('EPFL-ctrl',f))
for f in sorted(glob.glob(f'{D}/ispras/iscas85/bench/*.bench')): cands.append(('ISCAS85',f))
for f in sorted(glob.glob(f'{D}/ispras/mcnc/Combinational/blif/*.blif')):
    b=os.path.basename(f)
    if re.match(r'C\d+\.blif',b): continue   # ISCAS85 duplicates
    cands.append(('MCNC',f))
ansi=re.compile(r'\x1b\[[0-9;]*m')
def run(c):
    fam,f=c
    name=os.path.splitext(os.path.basename(f))[0]
    rd='read_bench' if f.endswith('.bench') else 'read_blif'
    cmd=f'{rd} {f}; strash; print_stats'
    try:
        o=subprocess.run([ABC,'-c',cmd],capture_output=True,text=True,timeout=60).stdout
    except Exception as e:
        return dict(family=fam,circuit=name,path=f,status='timeout')
    o=ansi.sub('',o)
    m=re.search(r'i/o\s*=\s*(\d+)/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)',o)
    if not m: return dict(family=fam,circuit=name,path=f,status='readfail')
    pi,po,lat,an,lev=map(int,m.groups())
    return dict(family=fam,circuit=name,path=f,status='ok',pi=pi,po=po,lat=lat,ands=an,lev=lev)
with ThreadPoolExecutor(2) as ex: rows=list(ex.map(run,cands))
keys=['family','circuit','path','status','pi','po','lat','ands','lev']
with open('out/inventory.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
import collections
print(collections.Counter((r['family'],r['status']) for r in rows))
