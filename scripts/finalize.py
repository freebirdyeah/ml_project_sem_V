"""Generate the final paper and submission when all experiment stages finish."""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
required=['results/EXPERIMENTS_COMPLETE','results/data_audit.json','results/model.joblib','results/deployment_smoke.json']
required+=['results/neural/COMPLETE','results/neural_runtime/COMPLETE','results/neural_model.joblib',
           'results/neural_static_model.joblib','results/neural_deployment_smoke.json',
           'results/neural_static_deployment_smoke.json','results/neural_validation.json']
while args.wait and not all((ROOT/x).exists() for x in required):time.sleep(5)
if not all((ROOT/x).exists() for x in required):raise RuntimeError('Required experiment or validation outputs are missing.')
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['MPLCONFIGDIR']='/tmp/ml_proj_mpl'
subprocess.run([sys.executable,'scripts/report.py'],cwd=ROOT,env=env,check=True)
if any(re.search(r'\bpending\b',x.read_text(),re.IGNORECASE) or 'experiments are running' in x.read_text().lower()
       for x in (ROOT/'paper/generated').glob('*.tex')):
    raise RuntimeError('A draft placeholder remains in the generated paper.')
for script in ['scripts/build_paper.py','scripts/package_submission.py']:
    subprocess.run([sys.executable,script],cwd=ROOT,env=env,check=True)
(ROOT/'results/SUBMISSION_COMPLETE').write_text('Dataset, evaluation, model, audit, paper, and submission archives generated.\n')
print('Submission generated.',flush=True)
