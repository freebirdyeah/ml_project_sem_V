"""ABC measurements. Every recipe and probe starts from the source circuit."""
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .recipes import COMMANDS, extend
from .aig import features

ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r'\x1b\[[0-9;]*m')
STATS = re.compile(r'i/o\s*=\s*(\d+)/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)')
TIMING = re.compile(r'Area\s*=\s*([\d.]+).*?Delay\s*=\s*([\d.]+)\s*ps')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strip_exdc_export(path):
    """Export the concrete Boolean function without auxiliary don't-care metadata.

    ABC's BLIF reader crashes on some mapped .gate networks with an attached
    unmapped .exdc network. The main network is retained byte-for-byte.
    """
    path=Path(path);text=path.read_text()
    marker=re.search(r'(?m)^\.exdc\s*$',text)
    if marker:
        path.write_text(text[:marker.start()].rstrip()+'\n.end\n')
        return True
    return False

class ABC:
    def __init__(self, binary=None, library=None, timeout=300):
        self.binary = str(Path(binary or ROOT/'tools/abc/abc').resolve())
        self.library = str(Path(library or ROOT/'data/raw/nangate45.lib').resolve())
        self.timeout = timeout

    @staticmethod
    def read(path):
        path = Path(path).resolve()
        # ABC's command parser does not consistently accept shell quoting.
        if any(c.isspace() or c in ';"\'' for c in str(path)):
            raise ValueError('ABC input paths must contain no whitespace, quotes, or semicolons.')
        command = {'.bench':'read_bench','.blif':'read_blif','.aig':'read_aiger','.aiger':'read_aiger'}.get(path.suffix)
        if command is None:
            raise ValueError('Provide a combinational .blif, .bench, or .aig circuit.')
        return f'{command} {path}; strash'

    def execute(self, command):
        t = time.perf_counter()
        p = subprocess.run([self.binary,'-q',command],capture_output=True,text=True,timeout=self.timeout)
        text = ANSI.sub('', p.stdout+'\n'+p.stderr)
        if p.returncode or re.search(r'(?im)^(Error:|Cannot open|ABC command line:.*failed)',text):
            raise RuntimeError(text[-3000:])
        return text, time.perf_counter()-t

    def measure(self, path, recipe='', output=None):
        command = f'read_lib -w {self.library}; {self.read(path)}; '
        if recipe:
            command += '; '.join(COMMANDS[o] for o in recipe.split())+'; '
        command += 'print_stats; map; topo; stime'
        if output:
            command += f'; write_blif {Path(output).resolve()}'
        text, elapsed = self.execute(command)
        if output:
            strip_exdc_export(output)
        m, t = STATS.search(text), TIMING.search(text)
        if not (m and t):
            raise RuntimeError('Unparseable ABC output: '+text[-3000:])
        if int(m[3]):
            raise ValueError('Only combinational circuits are supported.')
        return dict(status='ok',aig_nodes=int(m[4]),aig_depth=int(m[5]),
                    area=float(t[1]),delay=float(t[2]),runtime_s=elapsed)

    def describe(self, path):
        base = self.measure(path)
        structural_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='recipe-aig-') as tmp:
            aig = Path(tmp)/'circuit.aig'
            self.execute(f'{self.read(path)}; write_aiger {aig}')
            f = {f'c_{k}':v for k,v in features(aig).items()}
        structural_seconds = time.perf_counter()-structural_start
        if f['c_n_and'] < 1 or f['c_depth'] < 1:
            raise ValueError('A nonempty combinational AIG is required.')
        n0,l0 = f['c_n_and'],f['c_depth']
        probes = [(k,[v]) for k,v in COMMANDS.items()]
        probes += [(a+b,[COMMANDS[a],COMMANDS[b]]) for a,b in [('b','rw'),('rw','b'),('rf','b'),('b','rf')]]
        start = time.perf_counter()
        for name, commands in probes:
            text,_ = self.execute(self.read(path)+'; '+'; '.join(commands)+'; print_stats')
            m = STATS.search(text)
            if not m:
                raise RuntimeError(text)
            f[f'p_{name}_nodes'] = int(m[4])/n0
            f[f'p_{name}_depth'] = int(m[5])/l0
        f.update(base_area=base['area'],base_delay=base['delay'],
                 base_aig_nodes=n0,base_aig_depth=l0,baseline_runtime_s=base['runtime_s'],
                 structural_runtime_s=structural_seconds,
                 probe_runtime_s=time.perf_counter()-start)
        return f

def source_inventory():
    """Recover exactly the 205 supplied pilot circuits and verify unique paths."""
    pilot = pd.read_csv(ROOT/'data/raw/dataset_pilot_v1.csv')
    unique = pilot.drop_duplicates('circuit')
    roots = {'EPFL-arith':ROOT/'data/raw/epfl/arithmetic',
             'EPFL-ctrl':ROOT/'data/raw/epfl/random_control',
             'ISCAS85':ROOT/'data/raw/hdl-benchmarks/iscas85/bench',
             'MCNC':ROOT/'data/raw/hdl-benchmarks/mcnc/Combinational/blif'}
    rows = []
    for _, r in unique.iterrows():
        ext = '.bench' if r.family=='ISCAS85' else '.blif'
        matches = list(roots[r.family].rglob(r.circuit+ext))
        if len(matches)!=1:
            raise ValueError(f'Expected one source for {r.circuit}, found {matches}')
        p=matches[0]
        rows.append(dict(circuit=r.circuit,family=r.family,path=str(p.relative_to(ROOT)),sha256=digest(p)))
    return pd.DataFrame(rows)

def sweep(workers=4):
    inv=source_inventory(); inv.to_csv(ROOT/'data/processed/circuits.csv',index=False)
    recipes=extend(ROOT/'data/raw/recipes.csv'); recipes.to_csv(ROOT/'data/processed/recipes.csv',index=False)
    checkpoint=ROOT/'data/processed/checkpoints'; checkpoint.mkdir(exist_ok=True)
    abc=ABC()
    def work(c):
        dest=checkpoint/(c['circuit']+'.csv')
        if dest.exists():
            old=pd.read_csv(dest)
            if len(old)==len(recipes) and old.status.eq('ok').all():
                return c['circuit'],'cached'
        path=ROOT/c['path']; f=abc.describe(path); rows=[]
        for r in recipes.to_dict('records'):
            measured=abc.measure(path,r['ops'])
            row=dict(family=c['family'],circuit=c['circuit'],**f,
                     recipe_id=r['recipe_id'],recipe_kind=r['kind'],recipe_ops=r['ops'],**measured)
            row.update(yA=row['area']/f['base_area'],yD=row['delay']/f['base_delay'])
            row['J_0.5']=(row['yA']+row['yD'])/2
            rows.append(row)
        pd.DataFrame(rows).to_csv(dest.with_suffix('.tmp'),index=False)
        dest.with_suffix('.tmp').replace(dest)
        return c['circuit'],f'{len(rows)} runs'
    start=time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        for i,fut in enumerate(as_completed([ex.submit(work,c) for c in inv.to_dict('records')]),1):
            name,status=fut.result(); print(f'{i}/205 {name}: {status}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    data=pd.concat([pd.read_csv(checkpoint/(c+'.csv')) for c in inv.circuit],ignore_index=True)
    data.to_csv(ROOT/'data/processed/dataset.csv.tmp',index=False)
    (ROOT/'data/processed/dataset.csv.tmp').replace(ROOT/'data/processed/dataset.csv')
    manifest=dict(abc_commit='5d64f3b5f3bb9db6f3d77636a65e99ba3ee9d799',
                  epfl_commit='82d8cc6910419298e713a46644ed59fd3df53038',
                  hdl_commit='10c3fb51d0810ce862659211862660dcc270a6af',
                  library_source_commit='7bcaf475c884e15f674e428aa04dbf92bd0426ed',
                  library_sha256=digest(abc.library),dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),
                  recipe_extension_seed=17,workers=workers,wall_seconds=time.perf_counter()-start,
                  circuits=len(inv),recipes=len(recipes),rows=len(data))
    (ROOT/'data/processed/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
