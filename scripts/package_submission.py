"""Bundle the paper, implementation, data, results, and selected source circuits."""
from pathlib import Path
import zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'results/EXPERIMENTS_COMPLETE').exists():
    raise RuntimeError('Finish the experiment suite before packaging the submission.')
files=[]
for name in ['README.md','pyproject.toml','requirements.lock.txt','Makefile']:
    files.append(ROOT/name)
for directory in ['src','scripts','tests','examples','paper','results','vendor/pilot_scripts']:
    for p in (ROOT/directory).rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['__pycache__','.pytest_cache']) or any(x.endswith('.egg-info') for x in p.parts):continue
        if p.suffix in ['.aux','.log','.out','.tmp','.pyc'] or p.name=='model_benchmark.log':continue
        files.append(p)
for p in (ROOT/'data/processed').glob('*'):
    if p.is_file():files.append(p)
for name in ['dataset_pilot_v1.csv','dataset_sample.csv','recipes.csv','nangate45.lib']:
    files.append(ROOT/'data/raw'/name)
for name in ['final_paper.pdf','paper_source.zip','detailed_paper.pdf','detailed_paper_source.zip','classical_paper.pdf','classical_paper_source.zip','literature_survey.pdf','methodology_bundle.zip']:
    files.append(ROOT/'docs'/name)
inventory=pd.read_csv(ROOT/'data/processed/circuits.csv')
files += [ROOT/p for p in inventory.path]
files += [ROOT/'data/raw/epfl/LICENSE',ROOT/'data/raw/hdl-benchmarks/LICENSE']
dest=ROOT/'docs/project_submission.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(set(files)):
        if p.exists():archive.write(p,p.relative_to(ROOT))
print(dest,round(dest.stat().st_size/1024**2,2),'MiB')
