import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_dev_quantization import ap_values, resample_ap, ci, localization, localization_summary, summarize_draws
from verify_cctsdb_capture import coco_size


def fixture():
    records = [
        {"image":"a.jpg","orig_shape":[100,100],"xyxy":[[0,0,10,10],[0,0,10,10]],"class_id":[0,0],"confidence":[.8,.8]},
        {"image":"b.jpg","orig_shape":[100,100],"xyxy":[[0,0,30,30]],"class_id":[1],"confidence":[.8]},
        {"image":"c.jpg","orig_shape":[100,100],"xyxy":[],"class_id":[],"confidence":[]}]
    xml = {"a.jpg":{"shape":[100,100],"rows":[(0,[0,0,10,10])]},
           "b.jpg":{"shape":[100,100],"rows":[(1,[0,0,30,30])]},
           "c.jpg":{"shape":[100,100],"rows":[(0,[0,0,10,10])]}}
    return records,xml


class DevAnalysisTests(unittest.TestCase):
    def test_cached_bootstrap_equals_fresh_duplicate_image_evaluation(self):
        records,xml=fixture()
        _,e=coco_size(records,xml,return_evaluator=True)
        original=ap_values(e).copy()
        for indices in ([0,0,2],[1,1,1],[0,1,2],[2,2,2]):
            rs=[];xs={}
            for j,i in enumerate(indices):
                record=copy.deepcopy(records[i]);record["image"]=f"{j:04d}.jpg"
                rs.append(record);xs[record["image"]]=copy.deepcopy(xml[records[i]["image"]])
            _,fresh=coco_size(rs,xs,return_evaluator=True)
            expected=ap_values(fresh)
            before=np.any(e.eval['precision']>=0,axis=(0,1,4))
            after=np.any(fresh.eval['precision']>=0,axis=(0,1,4))
            expected[np.any(before & ~after,axis=0)]=np.nan
            np.testing.assert_allclose(resample_ap(e,indices),expected,rtol=0,atol=1e-14,equal_nan=True)
        np.testing.assert_array_equal(ap_values(e),original)

    def test_invalid_indices(self):
        r,x=fixture();_,e=coco_size(r,x,return_evaluator=True)
        for bad in ([],[-1],[3]):
            with self.assertRaises(ValueError):
                resample_ap(e,bad)

    def test_percentiles_and_missing_endpoints(self):
        result=ci([0,1,2,3,float('nan')],1)
        np.testing.assert_allclose(result['ci95_percentile_pp'],[.075,2.925])
        self.assertEqual(result['undefined_resamples'],1)
        self.assertEqual(ci([float('nan')],float('nan'))['ci95_percentile_pp'],[None,None])

    def test_identical_engines_have_zero_paired_delta(self):
        draws=np.tile(np.arange(5)[:,None,None,None]/10,(1,4,6,2))
        points=draws[0]
        summary=summarize_draws(draws,points)
        for contrast in summary.values():
            for size in contrast.values():
                for metric in size.values():
                    self.assertEqual(metric['ci95_percentile_pp'],[0,0])

    def test_localization_transitions_and_pair_geometry(self):
        r,x=fixture();_,e=coco_size(r,x,return_evaluator=True)
        ref=localization(e)
        self.assertEqual(len(ref),3)
        self.assertEqual(ref['1']['matched50_pair_iou'],1)
        self.assertEqual(ref['1']['matched50_center_error_over_sqrt_gt_area'],0)
        r[0]['xyxy']=[];r[0]['class_id']=[];r[0]['confidence']=[]
        _,changed=coco_size(r,x,return_evaluator=True)
        summary=localization_summary({'fp16':ref,'uniform':localization(changed)})
        self.assertEqual(summary['uniform']['xs']['lost_vs_fp16_iou50'],1)
        self.assertEqual(summary['uniform']['xs']['gained_vs_fp16_iou50'],0)
        self.assertIsNone(summary['uniform']['xs']['matched50_pair_iou_median'])


if __name__ == '__main__':
    unittest.main()
