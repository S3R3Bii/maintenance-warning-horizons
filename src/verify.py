"""Independent arithmetic from exported predictions; no estimator or scorer imports."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics as st
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def rows(name):
    opener=gzip.open if name.endswith('.gz') else open
    with opener(OUT/name,'rt',encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7),(a,b)

def groups(records,key):
    result=defaultdict(list)
    for row in records:result[row[key]].append(row)
    return result

def unit_losses(records):
    result={}
    for engine,values in groups(records,'engine').items():
        e=[float(r['prediction'])-float(r['rul']) for r in values]
        result[int(engine)]=dict(mse=st.mean(x*x for x in e),mae=st.mean(abs(x) for x in e))
    return result

def check_source_targets(records):
    """Read the original numeric labels directly, independently of the feature builder."""
    for partition,values in groups(records,'partition').items():
        original=defaultdict(list)
        with open(ROOT/f'data/raw/{partition}.txt') as handle:
            for line in handle:
                row=[float(x) for x in line.split()];assert len(row)==26;original[int(row[0])].append(row)
        offsets={engine:0 for engine in original}
        if partition.startswith('test'):
            with open(ROOT/f"data/raw/RUL_{partition.split('_')[1]}.txt") as handle:offsets=dict(enumerate([float(x) for x in handle.read().split()],1))
        for row in values:
            engine,cycle=int(row['engine']),int(row['cycle']);assert 20<=cycle<=len(original[engine])
            close(float(row['rul']),len(original[engine])+offsets[engine]-cycle)

def check_error(records,expected):
    unit=unit_losses(records);mse=st.mean(r['mse'] for r in unit.values());mae=st.mean(r['mae'] for r in unit.values())
    close(mse,expected['engine_mse']);close(math.sqrt(mse),expected['engine_rmse']);close(mae,expected['engine_mae'])
    close(st.mean((float(r['prediction'])-float(r['rul']))**2 for r in records),expected['pooled_mse'])
    assert len(unit)==int(expected['engines']) and len(records)==int(expected['windows'])
    last=[max(v,key=lambda r:int(r['cycle'])) for v in groups(records,'engine').values()]
    close(st.mean((float(r['prediction'])-float(r['rul']))**2 for r in last),expected['last_mse'])
    close(st.mean(abs(float(r['prediction'])-float(r['rul'])) for r in last),expected['last_mae'])
    return unit,last

def verify_validation():
    selection=json.loads((OUT/'selection.json').read_text());search=rows('search.csv');prediction=groups(rows('validation_predictions.csv.gz'),'model')
    assert len(search)==len(prediction)==15
    losses={}
    for trial in search:
        values=prediction[trial['name']];assert {int(r['engine']) for r in values}==set(range(71,86))
        check_error(values,trial);losses[trial['name']]=st.mean(v['mse'] for v in unit_losses(values).values())
    check_source_targets(prediction[search[0]['name']])
    assert min(losses,key=lambda name:(losses[name],name))==selection['winner']
    calibration=groups(rows('calibration_predictions.csv.gz'),'model')
    for name,values in calibration.items():
        engines=groups(values,'engine');assert {int(x) for x in engines}==set(range(86,101))
        scores=sorted(max(abs(float(r['prediction'])-float(r['rul'])) for r in group) for group in engines.values())
        order=math.ceil((len(scores)+1)*.9);assert order==selection['calibration_order']==15;close(scores[order-1],selection['radii'][name])
        check_source_targets(values)
    assert set(selection['fit_engines'])==set(range(1,71)) and not set(selection['features'])&{'engine','rul','failure_age','final_cycle','partition'}
    for record in json.loads((ROOT/'data/manifest.json').read_text())['files']:
        assert hashlib.sha256((ROOT/record['file']).read_bytes()).hexdigest()==record['sha256']
    return dict(status='PASS',search_candidates=15,tuning_engines=15,calibration_engines=15,metric_recomputation='Independent scalar arithmetic for every candidate and calibration maximum/order statistic')

def verify_test():
    selection=json.loads((OUT/'selection.json').read_text());summary=json.loads((OUT/'test_summary.json').read_text());all_rows=rows('test_predictions.csv.gz');parts=groups(all_rows,'partition');unit_errors={}
    for expected in summary['results']:
        values=groups(parts[expected['partition']],'model')[expected['model']];unit,last=check_error(values,expected);unit_errors[(expected['partition'],expected['model'])]=unit
        check_source_targets(values)
        assert len(unit)==(100 if expected['partition']=='test_FD001' else 259)
        covered=[];widths=[];trajectory=defaultdict(list)
        for row in values:
            p,y=float(row['prediction']),float(row['rul']);q=selection['radii'][expected['model']]
            assert p>=0;close(max(0,p-q),row['lower']);close(p+q,row['upper'])
            inside=float(row['lower'])<=y<=float(row['upper']);assert inside==(row['covered']=='True');covered.append(inside);widths.append(float(row['upper'])-float(row['lower']));trajectory[row['engine']].append(inside)
        close(st.mean(covered),expected['point_coverage']);close(st.mean(all(v) for v in trajectory.values()),expected['trajectory_coverage']);close(st.mean(widths),expected['mean_interval_width'])
        batch=sorted(last,key=lambda r:(round(float(r['prediction']),8),int(r['engine'])));budget=math.floor(.1*len(batch));chosen=batch[:budget];decision=expected['inspection']
        assert budget==decision['budget'] and [int(r['engine']) for r in chosen]==decision['selected_engines']
        assert [float(r['rul']) for r in chosen]==decision['lead_times']
        for horizon in decision['horizons']:
            h=horizon['horizon'];eligible=sum(float(r['rul'])<=h for r in batch);capture=sum(float(r['rul'])<=h for r in chosen)
            assert eligible==horizon['eligible'] and capture==horizon['captured'];close(capture/eligible,horizon['recall']);close(capture/budget,horizon['precision'])
        for band in expected['bands']:
            def include(row):
                y=float(row['rul']);return y<=30 if band['band']=='0-30' else (31<=y<=90 if band['band']=='31-90' else y>90)
            subset=[r for r in values if include(r)];assert len(subset)==band['windows'];close(st.mean((float(r['prediction'])-float(r['rul']))**2 for r in subset),band['mse']);close(st.mean(r['covered']=='True' for r in subset),band['coverage'])
    rng=np.random.default_rng(20261028)
    for expected in summary['bootstrap']:
        a=unit_errors[(expected['partition'],selection['winner'])];b=unit_errors[(expected['partition'],selection['baseline'])];assert a.keys()==b.keys()
        differences=[a[k]['mse']-b[k]['mse'] for k in sorted(a)];close(st.mean(differences),expected['mean'])
        draws=rng.integers(0,len(differences),size=(2000,len(differences)));replicates=sorted(st.mean(differences[i] for i in draw) for draw in draws)
        for p,value in zip([.025,.975],expected['percentile95']):
            position=(len(replicates)-1)*p;lo=math.floor(position);hi=math.ceil(position);close(replicates[lo]+(position-lo)*(replicates[hi]-replicates[lo]),value)
    return dict(status='PASS',prediction_rows=len(all_rows),engine_partitions=[100,259],metric_recomputation='Independent MSE/MAE, calibration, intervals, strata, trajectory coverage, inspection ranks and paired-engine bootstrap')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validation','test']);phase=parser.parse_args().phase
    result=verify_validation() if phase=='validation' else verify_test();(OUT/f'verification_{phase}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
