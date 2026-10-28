"""Refit the frozen protocol in a separate directory and compare required outputs."""
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def compare(a,b,key='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),key
        for k in a:
            if k not in ['frozen_utc','evaluated_utc','elapsed_seconds']:compare(a[k],b[k],key+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),key
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,key+f'[{i}]')
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(key,a,b)
    else:assert a==b,(key,a,b)

with tempfile.TemporaryDirectory(prefix='maintenance-replay-') as folder:
    replay=Path(folder)
    for directory in ['src','data']:
        shutil.copytree(ROOT/directory,replay/directory,ignore=shutil.ignore_patterns('*.joblib','__pycache__','official-mirror.zip'))
    for script,args in [('analyze.py',['validate']),('verify.py',['validation']),('check_features.py',['validation']),('analyze.py',['test']),('verify.py',['test']),('check_features.py',['test']),('figures.py',[])]:
        subprocess.run([sys.executable,str(replay/'src'/script),*args],cwd=replay,check=True,timeout=600)
    for name in ['selection.json','test_summary.json','training_quality.json']:
        compare(json.loads((ROOT/'results'/name).read_text()),json.loads((replay/'results'/name).read_text()),name)
    print('PASS: fifteen-model refit, engine-level calibration, protected test metrics and inspection ranks reproduce.')
