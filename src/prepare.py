"""Past-only sensor summaries and explicit source-local engine identities."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW_COLUMNS=['engine','cycle']+[f'setting_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
FEATURES=['cycle']+RAW_COLUMNS[2:]+[f'sensor_{i}_{stat}' for i in range(1,22) for stat in ['mean20','sd20','slope20']]

def read_source(partition):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256']
    frame=pd.read_csv(ROOT/f'data/raw/{partition}.txt',sep=r'\s+',header=None,float_precision='round_trip')
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
    output=[];twice_centered=np.arange(20,dtype=np.int64)*2-19
    for engine,group in frame.groupby('engine',sort=True):
        group=group.sort_values('cycle').copy()
        for sensor in RAW_COLUMNS[5:]:
            # Source values have at most four decimals. Integer moments avoid
            # cancellation noise in nearly constant windows and preserve that precision.
            values=np.rint(group[sensor].to_numpy()*10000).astype(np.int64)
            assert np.max(abs(values/10000-group[sensor].to_numpy()))<1e-9
            assert np.max(abs(values))<=150000000,'Integer-moment overflow bound'
            windows=np.lib.stride_tricks.sliding_window_view(values,20)
            total=windows.sum(axis=1);squares=(windows*windows).sum(axis=1)
            numerator=20*squares-total*total;assert (numerator>=0).all()
            means=total/200000
            deviations=np.array([math.sqrt(int(x))/200000 for x in numerator])
            slopes=(windows@twice_centered)/13300000
            group[sensor]=values/10000
            group[sensor+'_mean20']=np.r_[np.full(19,np.nan),means]
            group[sensor+'_sd20']=np.r_[np.full(19,np.nan),deviations]
            group[sensor+'_slope20']=np.r_[np.full(19,np.nan),slopes]
        output.append(group.loc[group.cycle>=20])
    result=pd.concat(output,ignore_index=True)
    assert np.isfinite(result[FEATURES].to_numpy()).all()
    assert not {'rul','engine','partition','failure_age','final_cycle'}&set(FEATURES)
    return result

def feature_fingerprint(frame):
    return hashlib.sha256(frame[FEATURES].to_numpy(dtype='<f8').tobytes()).hexdigest()

def weights(frame):
    counts=frame.groupby('engine').cycle.transform('count')
    return (len(frame)/frame.engine.nunique()/counts).to_numpy()

if __name__=='__main__':
    training=read_source('train_FD001');eligible=features(training)
    print(json.dumps({'source_rows':len(training),'eligible_rows':len(eligible),'engines':training.engine.nunique(),'features':len(FEATURES),'protected_test_not_read':True}))
