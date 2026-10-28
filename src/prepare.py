"""Past-only sensor summaries and explicit source-local engine identities."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW_COLUMNS=['engine','cycle']+[f'setting_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
FEATURES=['cycle']+RAW_COLUMNS[2:]+[f'sensor_{i}_{stat}' for i in range(1,22) for stat in ['mean20','sd20','slope20']]

def read_source(partition):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256']
    frame=pd.read_csv(ROOT/f'data/raw/{partition}.txt',sep=r'\s+',header=None)
    assert frame.shape[1]==26 and np.isfinite(frame.to_numpy()).all()
    frame.columns=RAW_COLUMNS
    assert not frame.duplicated(['engine','cycle']).any()
    for col in ['engine','cycle']:
        assert (frame[col]%1==0).all();frame[col]=frame[col].astype(int)
    for _,group in frame.groupby('engine',sort=True):
        assert group.cycle.tolist()==list(range(1,len(group)+1))
    engines=sorted(frame.engine.unique().tolist())
    assert engines==list(range(1,len(engines)+1))
    maximum=frame.groupby('engine').cycle.max()
    if partition.startswith('train'):
        failure_age=maximum
    else:
        suffix=partition.split('_')[1]
        remaining=pd.read_csv(ROOT/f'data/raw/RUL_{suffix}.txt',sep=r'\s+',header=None)
        assert remaining.shape==(len(engines),1) and (remaining[0]>=0).all()
        assert (remaining[0]%1==0).all()
        failure_age=maximum+pd.Series(remaining[0].to_numpy(),index=engines)
    frame['rul']=frame.engine.map(failure_age)-frame.cycle
    frame['partition']=partition
    return frame

def features(frame):
    output=[];time=np.arange(20,dtype=float);centered=time-time.mean();denom=sum(centered**2)
    for engine,group in frame.groupby('engine',sort=True):
        group=group.sort_values('cycle').copy()
        for sensor in RAW_COLUMNS[5:]:
            window=group[sensor].rolling(20,min_periods=20)
            group[sensor+'_mean20']=window.mean()
            group[sensor+'_sd20']=window.std(ddof=0)
            group[sensor+'_slope20']=window.apply(lambda x:float(np.dot(x,centered)/denom),raw=True)
        output.append(group.loc[group.cycle>=20])
    result=pd.concat(output,ignore_index=True)
    assert np.isfinite(result[FEATURES].to_numpy()).all()
    assert not {'rul','engine','partition','failure_age','final_cycle'}&set(FEATURES)
    return result

def weights(frame):
    counts=frame.groupby('engine').cycle.transform('count')
    return (len(frame)/frame.engine.nunique()/counts).to_numpy()

if __name__=='__main__':
    training=read_source('train_FD001');eligible=features(training)
    print(json.dumps({'source_rows':len(training),'eligible_rows':len(eligible),'engines':training.engine.nunique(),'features':len(FEATURES),'protected_test_not_read':True}))
