import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from importlib.metadata import PackageNotFoundError
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uniform_build_repeat import (SETTINGS,STUDY,load_contract,ensure_idle,inspector_signature,write_bytes,
                                  validate_selection,onnx_dependencies,classify_cuda_processes,
                                  is_allowed_snap_desktop_executable,SNAP_DESKTOP_ALLOWLIST,snapshot)
from capture_cctsdb_validator import FROZEN_WEIGHTS_SHA256
from audit_cctsdb_measurement import sha256


class BuildRepeatTests(unittest.TestCase):
    def test_onnxruntime_distribution_variants(self):
        for variant in ('onnxruntime','onnxruntime-gpu','onnxruntime-qnn'):
            def version(name):
                if name in ('onnx','onnxslim',variant):
                    return '1.0'
                raise PackageNotFoundError(name)
            module=SimpleNamespace(__version__='1.0',__file__='/env/onnxruntime/__init__.py',get_available_providers=lambda:['CPUExecutionProvider'])
            with patch('uniform_build_repeat.importlib.metadata.version',side_effect=version),patch('uniform_build_repeat.importlib.import_module',return_value=module):
                result=onnx_dependencies()
                self.assertIn(variant,result['distributions'])
                self.assertFalse(result['multiple_runtime_distributions'])

    def test_missing_or_broken_runtime_fails_before_export(self):
        def version(name):
            if name in ('onnx','onnxslim'):
                return '1.0'
            raise PackageNotFoundError(name)
        with patch('uniform_build_repeat.importlib.metadata.version',side_effect=version):
            with self.assertRaisesRegex(RuntimeError,'No ONNX Runtime'):
                onnx_dependencies()
        with patch('uniform_build_repeat.importlib.metadata.version',return_value='1.0'),patch('uniform_build_repeat.importlib.import_module',side_effect=ImportError('ABI')):
            with self.assertRaisesRegex(RuntimeError,'cannot import'):
                onnx_dependencies()

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
            ensure_idle({'processes':f'{os.getpid()+1}, another training, 12000 MiB',
                         'process_details':classify_cuda_processes(
                             f'{os.getpid()+1}, another training, 12000 MiB',
                             current_pid=os.getpid(), executable_resolver=lambda _: '/opt/train/python')})

    def test_verified_snap_desktop_is_logged_and_allowed(self):
        details=classify_cuda_processes(
            '445, snapd-desktop-integration, 5 MiB', current_pid=999,
            executable_resolver=lambda _: '/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration')
        state={'processes':'445, snapd-desktop-integration, 5 MiB', 'process_details':details,
               'process_guard':{'allowlist':SNAP_DESKTOP_ALLOWLIST}}
        ensure_idle(state)
        self.assertTrue(is_allowed_snap_desktop_executable(details[0]['resolved_executable']))
        self.assertEqual(details[0]['classification'],'allowed_snap_desktop')
        self.assertEqual(details[0]['allowlist_exception'],'snapd-desktop-integration')
        self.assertIn('verified executable matches allowlist',details[0]['reason'])
        self.assertFalse(is_allowed_snap_desktop_executable('/snap/other-package/391/usr/bin/snapd-desktop-integration'))
        self.assertFalse(is_allowed_snap_desktop_executable('/snap/snapd-desktop-integration/latest/usr/bin/snapd-desktop-integration'))

    def test_snapshot_records_desktop_exception_and_disclaimer(self):
        with patch('uniform_build_repeat.subprocess.run', side_effect=[
            SimpleNamespace(stdout='GPU snapshot'),
            SimpleNamespace(stdout='445, snapd-desktop-integration, 5 MiB'),
        ]), patch('uniform_build_repeat.resolve_process_executable',
                   return_value='/snap/snapd-desktop-integration/392/usr/bin/snapd-desktop-integration'):
            state=snapshot()
        ensure_idle(state)
        self.assertEqual(state['process_guard']['allowlist'],SNAP_DESKTOP_ALLOWLIST)
        self.assertIn('not proof of zero GPU interference',state['process_guard']['disclaimer'])
        self.assertEqual(state['process_details'][0]['resolved_executable'],
                         '/snap/snapd-desktop-integration/392/usr/bin/snapd-desktop-integration')
        self.assertEqual(state['process_details'][0]['classification'],'allowed_snap_desktop')

    def test_low_memory_training_is_still_blocked(self):
        details=classify_cuda_processes(
            '445, python, 64 MiB', current_pid=999,
            executable_resolver=lambda _: '/home/ubuntu/project/.venv/bin/python')
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':'445, python, 64 MiB', 'process_details':details})
        self.assertEqual(details[0]['classification'],'blocked_non_allowlisted_process')

    def test_desktop_name_with_different_executable_is_blocked(self):
        details=classify_cuda_processes(
            '445, snapd-desktop-integration, 5 MiB', current_pid=999,
            executable_resolver=lambda _: '/usr/local/bin/snapd-desktop-integration')
        self.assertFalse(details[0]['allowed'])
        self.assertEqual(details[0]['classification'],'blocked_non_allowlisted_process')
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':'445, snapd-desktop-integration, 5 MiB', 'process_details':details})

    def test_unreadable_executable_fails_closed(self):
        def unreadable(_):
            raise PermissionError('permission denied')
        details=classify_cuda_processes('445, desktop, 5 MiB', current_pid=999, executable_resolver=unreadable)
        self.assertEqual(details[0]['classification'],'blocked_unverifiable')
        self.assertIn('PermissionError',details[0]['reason'])
        with self.assertRaisesRegex(RuntimeError,'unverifiable'):
            ensure_idle({'processes':'445, desktop, 5 MiB', 'process_details':details})

    def test_process_race_fails_closed(self):
        def disappeared(_):
            raise FileNotFoundError('no such process')
        details=classify_cuda_processes('445, desktop, 5 MiB', current_pid=999, executable_resolver=disappeared)
        self.assertEqual(details[0]['classification'],'blocked_unverifiable')
        self.assertIn('FileNotFoundError',details[0]['reason'])
        with self.assertRaisesRegex(RuntimeError,'unverifiable'):
            ensure_idle({'processes':'445, desktop, 5 MiB', 'process_details':details})

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
