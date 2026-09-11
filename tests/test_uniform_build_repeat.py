import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uniform_build_repeat import SETTINGS,STUDY,load_contract,ensure_idle,inspector_signature,write_bytes,validate_selection
from capture_cctsdb_validator import FROZEN_WEIGHTS_SHA256
from audit_cctsdb_measurement import sha256


class BuildRepeatTests(unittest.TestCase):
    def test_contract_guards_files_and_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write_bytes(p/'source.onnx',b'onnx');write_bytes(p/'calibration_uint8.npy',b'tensor')
            c={'study':STUDY,'settings':SETTINGS,'source_weights_sha256':FROZEN_WEIGHTS_SHA256,
               'onnx_sha256':sha256(p/'source.onnx'),'calibration_tensor_file_sha256':sha256(p/'calibration_uint8.npy')}
            (p/'study_manifest.json').write_text(json.dumps(c))
            self.assertEqual(load_contract(p),c)
            (p/'source.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                load_contract(p)
            with self.assertRaises(FileExistsError):
                write_bytes(p/'source.onnx',b'overwrite')

    def test_no_bbox_or_fp16_or_timing_reuse(self):
        self.assertFalse(SETTINGS['fp16'])
        self.assertEqual(SETTINGS['timing_cache'],'fresh_empty_for_each_repeat')
        self.assertIn('no bbox/classification override',SETTINGS['sigmoid'])

    def test_gpu_guard_no_killing(self):
        ensure_idle({'processes':''})
        ensure_idle({'processes':f'{os.getpid()}, python, 64 MiB'})
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':f'{os.getpid()+1}, another training, 12000 MiB'})

    def test_layer_signature_detects_tactics_and_formats(self):
        a={'Layers':[{'Name':'conv','LayerType':'CaskConvolution','ParameterType':'Convolution','Weights':{'Type':'Int8'},'TacticName':'A'}]}
        b=copy.deepcopy(a);b['Layers'][0]['TacticName']='B'
        self.assertNotEqual(inspector_signature(a)['plan_signature_sha256'],inspector_signature(b)['plan_signature_sha256'])
        self.assertEqual(inspector_signature(a)['conv_weight_types'],{'Int8':1})
        with self.assertRaises(ValueError):
            inspector_signature({'Layers':['name only']})

    def test_selection_rejects_nontrain_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);path=repo/'data/processed/cctsdb2021_clean/train/images';path.mkdir(parents=True)
            (path/'a.jpg').write_bytes(b'image')
            m={'strategy':'uniform','seed':42,'selected_size':1024,'candidate_size':14720,'files':[{'source_image':'train/images/a.jpg'}]*1024}
            with self.assertRaisesRegex(ValueError,'duplicated'):
                validate_selection(repo,m)
            m['files']=[{'source_image':'dev/images/a.jpg'}]
            with self.assertRaisesRegex(ValueError,'strictly train'):
                validate_selection(repo,m)


if __name__=='__main__':
    unittest.main()
