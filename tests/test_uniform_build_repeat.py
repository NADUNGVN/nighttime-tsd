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
                                  is_allowed_snap_desktop_executable,is_allowed_desktop_executable,
                                  desktop_allowlist_id,DESKTOP_ALLOWLIST,
                                  parse_desktop_confirmations,snapshot)
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
                             current_pid=os.getpid())})

    def test_verified_snap_desktop_is_logged_and_allowed(self):
        path='/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration'
        details=classify_cuda_processes(
            f'445, {path}, 5 MiB', current_pid=999, confirmed_desktop={445:path})
        state={'processes':f'445, {path}, 5 MiB', 'process_details':details,
               'process_guard':{'allowlist':DESKTOP_ALLOWLIST}}
        ensure_idle(state)
        self.assertTrue(is_allowed_snap_desktop_executable(details[0]['resolved_executable']))
        self.assertEqual(details[0]['classification'],'allowed_desktop_operator_confirmed')
        self.assertEqual(details[0]['allowlist_exception'],'snapd-desktop-integration')
        self.assertIn('explicitly confirmed by operator',details[0]['reason'])
        self.assertFalse(is_allowed_snap_desktop_executable('/snap/other-package/391/usr/bin/snapd-desktop-integration'))
        self.assertFalse(is_allowed_snap_desktop_executable('/snap/snapd-desktop-integration/latest/usr/bin/snapd-desktop-integration'))

    def test_snapshot_records_desktop_exception_and_disclaimer(self):
        path='/snap/snapd-desktop-integration/392/usr/bin/snapd-desktop-integration'
        with patch('uniform_build_repeat.subprocess.run', side_effect=[
            SimpleNamespace(stdout='GPU snapshot'),
            SimpleNamespace(stdout=f'445, {path}, 5 MiB'),
        ]):
            state=snapshot({445:path})
        ensure_idle(state)
        self.assertEqual(state['process_guard']['allowlist'],DESKTOP_ALLOWLIST)
        self.assertIn('not proof of zero GPU interference',state['process_guard']['disclaimer'])
        self.assertIn('/proc/<pid>/exe was not read',state['process_guard']['verification_method'])
        self.assertEqual(state['process_details'][0]['resolved_executable'],path)
        self.assertEqual(state['process_details'][0]['classification'],'allowed_desktop_operator_confirmed')

    def test_confirmed_xorg_and_gnome_desktop_paths_are_narrow(self):
        for allowlist_id,path in (
            ('xorg','/usr/lib/xorg/Xorg'),
            ('gnome-shell','/usr/bin/gnome-shell'),
            ('gnome-session','/usr/libexec/gnome-session-binary'),
        ):
            self.assertEqual(desktop_allowlist_id(path),allowlist_id)
            self.assertTrue(is_allowed_desktop_executable(path))
            details=classify_cuda_processes(f'445, {path}, 5 MiB', current_pid=999, confirmed_desktop={445:path})
            self.assertEqual(details[0]['classification'],'allowed_desktop_operator_confirmed')
            self.assertEqual(details[0]['desktop_allowlist_id'],allowlist_id)

    def test_desktop_without_operator_confirmation_is_blocked(self):
        path='/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration'
        details=classify_cuda_processes(f'445, {path}, 5 MiB', current_pid=999)
        self.assertEqual(details[0]['classification'],'blocked_desktop_unconfirmed')
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':f'445, {path}, 5 MiB', 'process_details':details})

    def test_confirmation_must_match_current_pid_and_path(self):
        path='/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration'
        with self.assertRaisesRegex(ValueError,'outside the allowlist'):
            parse_desktop_confirmations(['445=/snap/other/391/usr/bin/other'])
        details=classify_cuda_processes(f'445, {path}, 5 MiB', current_pid=999, confirmed_desktop={446:path})
        self.assertEqual(details[0]['classification'],'blocked_desktop_unconfirmed')

    def test_missing_process_information_is_blocked(self):
        details=classify_cuda_processes('806157, [Not Found], 218 MiB', current_pid=999)
        self.assertEqual(details[0]['classification'],'blocked_unverifiable_process')
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':'806157, [Not Found], 218 MiB', 'process_details':details})

    def test_stale_operator_confirmation_is_blocked(self):
        path='/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration'
        with patch('uniform_build_repeat.subprocess.run', side_effect=[
            SimpleNamespace(stdout='GPU snapshot'), SimpleNamespace(stdout='')]):
            state=snapshot({445:path})
        self.assertEqual(state['process_guard']['unmatched_confirmations'],[{'pid':445,'reported_path':path}])
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle(state)

    def test_competing_workload_is_recorded_as_external_violation(self):
        with patch('uniform_build_repeat.subprocess.run', side_effect=[
            SimpleNamespace(stdout='GPU snapshot'), SimpleNamespace(stdout='446, python, 460 MiB')]):
            state=snapshot()
        self.assertTrue(state['process_guard']['external_workload_detected'])
        self.assertEqual(state['process_guard']['blocked_processes'][0]['classification'],
                         'blocked_non_allowlisted_process')

    def test_missing_telemetry_is_recorded_as_limited(self):
        with patch('uniform_build_repeat.subprocess.run', side_effect=[
            SimpleNamespace(stdout=None), SimpleNamespace(stdout=None)]):
            state=snapshot()
        self.assertEqual(state['process_guard']['telemetry_status'],'limited')
        self.assertTrue(state['process_guard']['external_workload_detected'])
        self.assertEqual(state['process_details'][0]['classification'],'blocked_unverifiable')

    def test_low_memory_training_is_still_blocked(self):
        details=classify_cuda_processes(
            '445, python, 64 MiB', current_pid=999)
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':'445, python, 64 MiB', 'process_details':details})
        self.assertEqual(details[0]['classification'],'blocked_non_allowlisted_process')

    def test_desktop_name_with_different_executable_is_blocked(self):
        details=classify_cuda_processes(
            '445, /usr/local/bin/snapd-desktop-integration, 5 MiB', current_pid=999)
        self.assertFalse(details[0]['allowed'])
        self.assertEqual(details[0]['classification'],'blocked_non_allowlisted_process')
        with self.assertRaisesRegex(RuntimeError,'Other CUDA'):
            ensure_idle({'processes':'445, /usr/local/bin/snapd-desktop-integration, 5 MiB', 'process_details':details})

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
