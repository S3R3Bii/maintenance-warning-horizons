"""Fixed finite search, engine-level calibration, then protected test evaluation."""
from datetime import datetime,timezone
import argparse
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from prepare import ROOT,FEATURES,features,read_source,weights,feature_fingerprint

RESULTS=ROOT/'results';CACHE=ROOT/'data/cache';SEED=20261028

def dump(name,value):
    RESULTS.mkdir(exist_ok=True);(RESULTS/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def candidates():
    values=[dict(name='failure_age_baseline',kind='baseline')]
    for depth in [4,8,None]:
        for leaf in [5,20]:values.append(dict(name=f'rf_depth{depth}_leaf{leaf}',kind='forest',max_depth=depth,min_samples_leaf=leaf,n_estimators=160))
    for depth in [2,3]:
        for rate in [.05,.1]:
            for l2 in [0,1]:values.append(dict(name=f'hgb_depth{depth}_lr{rate}_l2{l2}',kind='boosting',max_depth=depth,learning_rate=rate,l2_regularization=l2,max_iter=100,min_samples_leaf=20))
    return values

def fit(config,frame):
    if config['kind']=='baseline':return float(frame.groupby('engine').apply(lambda g:float((g.rul+g.cycle).iloc[0]),include_groups=False).mean())
    kwargs={k:v for k,v in config.items() if k not in ['name','kind']}
    if config['kind']=='forest':model=RandomForestRegressor(**kwargs,max_features='sqrt',bootstrap=True,n_jobs=2,random_state=SEED)
    else:model=HistGradientBoostingRegressor(**kwargs,early_stopping=False,random_state=SEED)
    with threadpool_limits(limits=2):model.fit(frame[FEATURES],frame.rul,sample_weight=weights(frame))
    return model

def predict(model,frame):
    if isinstance(model,float):return np.maximum(0,model-frame.cycle.to_numpy())
    with threadpool_limits(limits=2):return np.maximum(0,model.predict(frame[FEATURES]))

def errors(frame):
    error=frame.prediction-frame.rul
    units=frame.assign(squared=error**2,absolute=abs(error)).groupby('engine')[['squared','absolute']].mean()
    last=frame.sort_values(['engine','cycle']).groupby('engine').tail(1)
    return dict(engines=int(len(units)),windows=int(len(frame)),engine_mse=float(units.squared.mean()),engine_rmse=float(np.sqrt(units.squared.mean())),engine_mae=float(units.absolute.mean()),pooled_mse=float(np.mean(error**2)),last_mse=float(np.mean((last.prediction-last.rul)**2)),last_mae=float(np.mean(abs(last.prediction-last.rul))))

def prediction_table(model,frame,name):
    result=frame[['partition','engine','cycle','rul']].copy();result['model']=name;result['prediction']=predict(model,frame);return result

def inspection(frame,fraction=.1):
    last=frame.sort_values(['engine','cycle']).groupby('engine').tail(1).copy();last['rank_value']=last.prediction.round(8)
    last=last.sort_values(['rank_value','engine']);budget=math.floor(fraction*len(last));chosen=last.iloc[:budget]
    return dict(engines=len(last),budget=budget,selected_engines=chosen.engine.tolist(),lead_times=chosen.rul.tolist(),horizons=[dict(horizon=h,eligible=int((last.rul<=h).sum()),captured=int((chosen.rul<=h).sum()),recall=float((chosen.rul<=h).sum()/(last.rul<=h).sum()) if (last.rul<=h).any() else None,precision=float((chosen.rul<=h).mean()) if budget else None) for h in [15,30,60]])

def validate():
    RESULTS.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    raw=read_source('train_FD001');frame=features(raw)
    fitting=frame[frame.engine<=70];tuning=frame[frame.engine.between(71,85)];calibration=frame[frame.engine>=86]
    trials=[];predictions=[]
    for config in candidates():
        start=time.perf_counter();model=fit(config,fitting);table=prediction_table(model,tuning,config['name'])
        score=errors(table);trials.append(dict(**config,**score,status='success',elapsed_seconds=time.perf_counter()-start));predictions.append(table)
        joblib.dump(model,CACHE/(config['name']+'.joblib'));print(json.dumps({'candidate':config['name'],'engine_mse':score['engine_mse']}),flush=True)
    pd.DataFrame(trials).to_csv(RESULTS/'search.csv',index=False);pd.concat(predictions).to_csv(RESULTS/'validation_predictions.csv.gz',index=False,compression='gzip')
    winner=min(trials,key=lambda row:(row['engine_mse'],row['name']))['name'];calibration_records=[];radii={}
    for name in [winner,'failure_age_baseline']:
        model=joblib.load(CACHE/(name+'.joblib'));table=prediction_table(model,calibration,name);calibration_records.append(table)
        scores=table.assign(error=abs(table.prediction-table.rul)).groupby('engine').error.max().sort_values().to_numpy();order=math.ceil((len(scores)+1)*.9);assert order<=len(scores)
        radii[name]=float(scores[order-1])
    pd.concat(calibration_records).to_csv(RESULTS/'calibration_predictions.csv.gz',index=False,compression='gzip')
    dump('selection.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),winner=winner,baseline='failure_age_baseline',criterion='mean per-engine tuning MSE; name tie break',fit_engines=list(range(1,71)),tuning_engines=list(range(71,86)),calibration_engines=list(range(86,101)),features=FEATURES,radii=radii,nominal_trajectory_coverage=.9,calibration_order=15,candidates=len(trials),seed=SEED,protected_test_read=False))
    dump('training_quality.json',dict(source_rows=len(raw),eligible_windows=len(frame),engines=int(raw.engine.nunique()),feature_count=len(FEATURES),feature_sha256=feature_fingerprint(frame),fit_windows=len(fitting),tuning_windows=len(tuning),calibration_windows=len(calibration),data_kind='NASA simulation benchmark',evaluation_scope='Retrospective correction after original test inspection; integer-moment feature arithmetic'))

def test():
    selection=json.loads((RESULTS/'selection.json').read_text());tables=[];summary=[];bootstrap=[];fingerprints={};rng=np.random.default_rng(SEED)
    for partition in ['test_FD001','test_FD002']:
        frame=features(read_source(partition));fingerprints[partition]=feature_fingerprint(frame);partition_tables={}
        for name in [selection['winner'],selection['baseline']]:
            model=joblib.load(CACHE/(name+'.joblib'));table=prediction_table(model,frame,name);q=selection['radii'][name]
            table['lower']=np.maximum(0,table.prediction-q);table['upper']=table.prediction+q;table['covered']=(table.rul>=table.lower)&(table.rul<=table.upper);tables.append(table);partition_tables[name]=table
            by_engine=table.groupby('engine').covered.agg(['mean','all']);bands=[]
            for label,condition in [('0-30',table.rul<=30),('31-90',table.rul.between(31,90)),('above90',table.rul>90)]:
                part=table[condition];bands.append(dict(band=label,windows=len(part),mse=float(np.mean((part.prediction-part.rul)**2)),coverage=float(part.covered.mean())))
            summary.append(dict(partition=partition,model=name,**errors(table),point_coverage=float(table.covered.mean()),trajectory_coverage=float(by_engine['all'].mean()),mean_interval_width=float((table.upper-table.lower).mean()),radius=q,bands=bands,inspection=inspection(table)))
        unit_errors=[]
        for name in [selection['winner'],selection['baseline']]:
            table=partition_tables[name];unit_errors.append(table.assign(squared=(table.prediction-table.rul)**2).groupby('engine').squared.mean().to_numpy())
        difference=unit_errors[0]-unit_errors[1];samples=rng.integers(0,len(difference),size=(2000,len(difference)));means=difference[samples].mean(axis=1)
        bootstrap.append(dict(partition=partition,quantity='selected minus baseline engine MSE',engines=len(difference),draws=2000,seed=SEED,mean=float(difference.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),scope='Conditional on these simulated benchmark engines; paired engine resampling'))
    pd.concat(tables).to_csv(RESULTS/'test_predictions.csv.gz',index=False,compression='gzip')
    dump('test_summary.json',dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),evaluation_scope='Retrospective correction after original test inspection',feature_sha256=fingerprints,results=summary,bootstrap=bootstrap))
    print(json.dumps({'evaluated_partitions':2,'models_per_partition':2,'output':'results/test_summary.json'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validate','test']);args=parser.parse_args();globals()[args.phase]()
