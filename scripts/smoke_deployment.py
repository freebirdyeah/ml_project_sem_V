"""Check the saved artifact, live feature extraction, verification, and mapped output."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from circuit_recipe.cli import select
from circuit_recipe.synthesis import ROOT,ABC

def check(artifact='model.joblib',prefix=''):
    capture=io.StringIO();netlist=ROOT/f'results/{prefix}example_optimized.blif'
    with contextlib.redirect_stdout(capture):
        result=select(ROOT/'examples/ripple8.blif',ROOT/'results'/artifact,3,netlist)
    (ROOT/f'results/{prefix}deployment_example.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(result['measurements'])==3 and netlist.exists()
    assert result['best']['J']==min(m['J'] for m in result['measurements'])
    abc=ABC()
    with tempfile.TemporaryDirectory(prefix='recipe-smoke-') as tmp:
        reference=Path(tmp)/'reference.blif'
        abc.execute(abc.read(ROOT/'examples/ripple8.blif')+f'; write_blif {reference}')
        text,elapsed=abc.execute(f'read_lib -w {abc.library}; cec {reference} {netlist}')
        assert 'Networks are equivalent' in text,text
    dest=ROOT/f'results/{prefix}deployment_smoke.json'
    dest.write_text(json.dumps(dict(
        candidates_verified=3,chosen_recipe=result['best']['recipe_id'],mapped_output_equivalent=True,
        area=result['best']['area'],delay=result['best']['delay'],J=result['best']['J']),indent=2)+'\n')
    print(dest.read_text())

def main():
    check()
    check('neural_model.joblib','neural_')
    check('neural_static_model.joblib','neural_static_')

if __name__=='__main__':main()
