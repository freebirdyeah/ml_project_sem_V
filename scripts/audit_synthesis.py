"""Audit source hashes, pilot agreement, and functional equivalence of mapped results."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from circuit_recipe.synthesis import ROOT,ABC,digest,strip_exdc_export
from circuit_recipe.data import load_dataset

def main():
    d=load_dataset(ROOT/'data/processed/dataset.csv')
    pilot=load_dataset(ROOT/'data/raw/dataset_pilot_v1.csv')
    joined=d.merge(pilot,on=['circuit','recipe_id'],suffixes=('_new','_pilot'))
    fields=['area','delay','base_area','base_delay','yA','yD']+[c for c in pilot if c.startswith(('c_','p_'))]
    agreement={c:bool(np.allclose(joined[c+'_new'],joined[c+'_pilot'],rtol=1e-10,atol=1e-10)) for c in fields}
    inv=pd.read_csv(ROOT/'data/processed/circuits.csv').set_index('circuit')
    assert all(digest(ROOT/r.path)==r.sha256 for _,r in inv.iterrows())
    assert all(agreement.values()),agreement
    abc=ABC(timeout=300);records=[];rng=np.random.default_rng(43)
    # Six circuits per family, one independently sampled recipe on each.
    circuits=[]
    for family,g in inv.groupby('family'):
        circuits+=list(rng.choice(sorted(g.index),6,replace=False))
    for name in circuits:
        g=d[d.circuit==name];row=g.iloc[int(rng.integers(len(g)))];path=ROOT/inv.loc[name,'path']
        with tempfile.TemporaryDirectory(prefix='recipe-cec-') as tmp:
            reference=Path(tmp)/'reference.blif';mapped=Path(tmp)/'mapped.blif'
            abc.execute(abc.read(path)+f'; write_blif {reference}')
            reference_had_exdc=strip_exdc_export(reference)
            rerun=abc.measure(path,row.recipe_ops,mapped)
            assert rerun['area']==row.area and rerun['delay']==row.delay
            output,elapsed=abc.execute(f'read_lib -w {abc.library}; cec {reference} {mapped}')
            equivalent='Networks are equivalent' in output
            record=dict(circuit=name,recipe_id=row.recipe_id,family=row.family,equivalent=equivalent,
                        area_reproduced=rerun['area']==row.area,delay_reproduced=rerun['delay']==row.delay,seconds=elapsed)
            record['reference_had_exdc']=reference_had_exdc
            records.append(record);print(record,flush=True)
            if not equivalent:
                raise RuntimeError(output)
    pd.DataFrame(records).to_csv(ROOT/'results/equivalence_checks.csv',index=False)
    audit=dict(pilot_pairs_compared=len(joined),agreement=agreement,source_hashes_match=True,
               dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),
               equivalence_checks=len(records),all_equivalent=all(r['equivalent'] for r in records))
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest_path=ROOT/'data/processed/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['library_source_commit']='7bcaf475c884e15f674e428aa04dbf92bd0426ed'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
