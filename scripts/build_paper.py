"""Compile the paper twice and copy the PDF into docs/."""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env['TEXMFVAR']='/tmp/ml_proj_texmfvar';env['TEXMFCONFIG']='/tmp/ml_proj_texmfconfig'
for _ in range(2):
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT/'paper',env=env,check=True)
shutil.copyfile(ROOT/'paper/main.pdf',ROOT/'docs/final_paper.pdf')
with zipfile.ZipFile(ROOT/'docs/paper_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    files=[ROOT/'paper/main.tex',ROOT/'paper/etoolbox.sty']
    files+=list((ROOT/'paper/generated').glob('*.tex'))+list((ROOT/'paper/figures').glob('*.pdf'))
    for p in files:z.write(p,p.relative_to(ROOT/'paper'))
    z.writestr('README.txt','Compile main.tex twice with pdfLaTeX. The archive contains every generated table, result paragraph, and vector figure needed by the paper. Latin Modern fonts and standard LaTeX packages are required. The bundled etoolbox.sty retains its upstream LPPL notice.\n')
print(ROOT/'docs/final_paper.pdf')
