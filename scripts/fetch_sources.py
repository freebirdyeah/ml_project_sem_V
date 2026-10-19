"""Download pinned source archives and the exact mapping library used here."""
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
    ('berkeley-abc/abc','5d64f3b5f3bb9db6f3d77636a65e99ba3ee9d799','tools/abc'),
    ('ispras/hdl-benchmarks','10c3fb51d0810ce862659211862660dcc270a6af','data/raw/hdl-benchmarks'),
    ('lsils/benchmarks','82d8cc6910419298e713a46644ed59fd3df53038','data/raw/epfl')]
LIB_COMMIT='7bcaf475c884e15f674e428aa04dbf92bd0426ed'
LIB_SHA='8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1'

def main():
    with tempfile.TemporaryDirectory(prefix='recipe-sources-') as tmp:
        for repo,commit,target in SOURCES:
            dest=ROOT/target
            if dest.exists() and any(dest.iterdir()):
                print('Already present:',target);continue
            archive=Path(tmp)/'source.tar.gz'
            subprocess.run(['curl','--fail','-L','--retry','2',f'https://api.github.com/repos/{repo}/tarball/{commit}','-o',str(archive)],check=True)
            dest.mkdir(parents=True,exist_ok=True)
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    name=m.name.split('/',1)
                    if len(name)<2 or not name[1] or m.issym() or m.islnk():continue
                    m.name=name[1];t.extract(m,dest,filter='data')
        library=ROOT/'data/raw/nangate45.lib'
        if not library.exists():
            url=f'https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/{LIB_COMMIT}/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib'
            subprocess.run(['curl','--fail','-L','--retry','2',url,'-o',str(library)],check=True)
        if hashlib.sha256(library.read_bytes()).hexdigest()!=LIB_SHA:
            raise ValueError('Library checksum mismatch.')
    print('Sources ready. Build with: make -C tools/abc -j4 ABC_USE_NO_READLINE=1')

if __name__=='__main__':main()
