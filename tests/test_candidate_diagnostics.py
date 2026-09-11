import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_dev_pairs import candidates,compare,select_examples
from inspect_frozen_trt_engines import unwrap_engine,format_evidence


def record(pred,classes,scores):
    return {'image':'a.jpg','class_id':classes,'confidence':scores,
            'validator_input':{'target_xyxy':[[0,0,10,10]],'target_class_id':[0],'prediction_xyxy':pred}}


class CandidateTests(unittest.TestCase):
    def test_same_class_and_low_score_candidates(self):
        r=record([[0,0,10,10],[0,0,10,10],[0,0,9,9]],[1,0,0],[.99,.1,.8])
        c=candidates(r)[0]
        self.assertEqual(c['best_index'],1)
        self.assertEqual(c['candidate_count_iou50'],2)
        self.assertEqual(c['max_score_iou50'],.8)
        self.assertAlmostEqual(c['best_iou_conf025'],.81,places=6)

    def test_no_predictions_and_wrong_class(self):
        for r in (record([],[],[]),record([[0,0,10,10]],[1],[.9])):
            c=candidates(r)[0]
            self.assertEqual(c['best_iou'],0)
            self.assertIsNone(c['best_score'])

    def test_deterministic_selection_includes_gain_control(self):
        good=record([[0,0,10,10]],[0],[.9]);bad=record([[0,0,6,10]],[0],[.5])
        rows=compare(good,bad)+compare(bad,good)
        chosen=select_examples(rows)
        self.assertEqual([c['category'] for c in chosen],['iou75_loss','iou75_gain_control'])
        self.assertEqual(chosen,select_examples(rows))

    def test_metadata_and_plain_plan(self):
        meta=json.dumps({'names':{'0':'sign'}}).encode()
        plan=b'plan bytes'
        payload=len(meta).to_bytes(4,'little')+meta+plan
        self.assertEqual(unwrap_engine(payload),(plan,json.loads(meta)))
        self.assertEqual(unwrap_engine(plan),(plan,None))
        invalid=(99).to_bytes(4,'little')+b'too short'
        self.assertEqual(unwrap_engine(invalid),(invalid,None))

    def test_inspection_does_not_guess_dtype_from_names(self):
        self.assertEqual(format_evidence({'Layers':[{'Name':'INT8_convolution'}]}),[])
        fields=format_evidence({'Layers':[{'Name':'x','Inputs':[{'Format/Datatype':'Half'}]}]})
        self.assertEqual(fields,[{'field':'/Layers/0/Inputs/0/Format/Datatype','value':'Half'}])


if __name__=='__main__':
    unittest.main()
