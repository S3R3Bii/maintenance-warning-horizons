"""Independent scalar audit of a declared small sample of actual source windows."""
import json
import math
import statistics
from collections import defaultdict
import argparse
from prepare import ROOT,features,read_source

def audit(partition):
    raw=defaultdict(list)
    with open(ROOT/f'data/raw/{partition}.txt') as handle:
        for line in handle:
            row=[float(x) for x in line.split()];raw[int(row[0])].append(row)
    exported=features(read_source(partition));count=0
    for engine in [min(raw),sorted(raw)[len(raw)//2],max(raw)]:
        for cycle in [20,len(raw[engine])]:
            actual=exported[(exported.engine==engine)&(exported.cycle==cycle)].iloc[0]
            for sensor in range(1,22):
                window=[r[4+sensor] for r in raw[engine][cycle-20:cycle]];mean=statistics.mean(window);sd=statistics.pstdev(window)
                slope=sum((i-9.5)*(value-mean) for i,value in enumerate(window))/sum((i-9.5)**2 for i in range(20))
                for name,expected in [('mean20',mean),('sd20',sd),('slope20',slope)]:assert math.isclose(actual[f'sensor_{sensor}_{name}'],expected,abs_tol=1e-7,rel_tol=1e-9)
            assert actual.cycle==cycle;count+=1
    return count

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validation','test']);phase=parser.parse_args().phase
    partitions=['train_FD001'] if phase=='validation' else ['test_FD001','test_FD002'];result={'status':'PASS','windows_by_partition':{p:audit(p) for p in partitions},'scope':'Six real source windows per partition, all21 sensors and three summaries; spot audit, not an exhaustive independent feature implementation'}
    (ROOT/f'results/feature_audit_{phase}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
