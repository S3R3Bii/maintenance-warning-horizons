"""Information boundaries and operational accounting, not model score assertions."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare import RAW_COLUMNS,FEATURES,features,weights
from analyze import inspection

def fixture():
    data=[]
    for engine in [1,2]:
        for cycle in range(1,31):data.append([engine,cycle,0,0,0]+[engine*10+cycle*i for i in range(1,22)])
    frame=pd.DataFrame(data,columns=RAW_COLUMNS);frame['rul']=50-frame.cycle;frame['partition']='fixture';return frame

class Contracts(unittest.TestCase):
    def test_future_values_cannot_change_past_features(self):
        a=fixture();b=a.copy();b.loc[b.cycle>25,'sensor_1']=1e9
        left=features(a).query('cycle<=25')[FEATURES];right=features(b).query('cycle<=25')[FEATURES]
        np.testing.assert_array_equal(left.to_numpy(),right.to_numpy())

    def test_engine_boundary_and_exact_window_math(self):
        result=features(fixture());row=result.query('engine==2 and cycle==20').iloc[0]
        self.assertAlmostEqual(row.sensor_1_mean20,30.5);self.assertAlmostEqual(row.sensor_1_sd20,np.std(np.arange(21,41)));self.assertAlmostEqual(row.sensor_1_slope20,1)
        self.assertTrue((result.cycle>=20).all());self.assertEqual(len(result),22)

    def test_labels_do_not_affect_features(self):
        a=fixture();b=a.copy();b['rul']=9999
        np.testing.assert_array_equal(features(a)[FEATURES],features(b)[FEATURES]);self.assertEqual(len(FEATURES),88)

    def test_each_engine_has_equal_total_weight(self):
        frame=fixture();frame=frame[~((frame.engine==2)&(frame.cycle>25))].copy();frame['weight']=weights(frame)
        totals=frame.groupby('engine').weight.sum();self.assertAlmostEqual(totals.loc[1],totals.loc[2])

    def test_budget_and_tie_rule(self):
        frame=pd.DataFrame({'engine':range(1,22),'cycle':[20]*21,'rul':range(21),'prediction':[5.0]*21})
        decision=inspection(frame);self.assertEqual(decision['budget'],2);self.assertEqual(decision['selected_engines'],[1,2]);self.assertEqual(decision['horizons'][1]['captured'],2)

if __name__=='__main__':unittest.main()
