from __future__ import annotations

import hashlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from precision_head_calibration_order import (  # noqa: E402
    CalibrationOrderError,
    ManifestIndexedDataset,
    build_manifest_ordered_calibration_dataloader,
    verify_pinned_source_contract,
)


class _TinyDataset:
    def __init__(self, paths: list[Path]):
        self.im_files = [str(path) for path in paths]
        self.collate_fn = lambda rows: rows
        self.transforms = SimpleNamespace(transforms=[])
        self.augment = False
        self.rect = False

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        return {"im_file": self.im_files[index]}


class CalibrationOrderContractTests(unittest.TestCase):
    def make_fake_inputs(self, discovered: list[Path]):
        class LoaderExporter:
            def __init__(self):
                self.args = SimpleNamespace(data="fixture.yaml", split="val", batch=1, fraction=1.0, rect=False)
                self.imgsz = (640, 640)

        seen: dict[str, object] = {}

        def check_data(_path, split):
            self.assertEqual(split, "val")
            return {"val": "fixture-images"}

        def build_dataset(cfg, img_path, batch, _data, *, mode, fraction):
            self.assertEqual(cfg.imgsz, 640)
            self.assertEqual((img_path, batch, mode, fraction), ("fixture-images", 1, "val", 1.0))
            return _TinyDataset(discovered)

        def build_loader(dataset, *, batch, workers, shuffle, drop_last):
            seen.update({"dataset": dataset, "batch": batch, "workers": workers, "shuffle": shuffle, "drop_last": drop_last})
            return dataset

        return LoaderExporter(), check_data, build_dataset, build_loader, seen

    def test_manifest_mapping_reorders_discovery_and_explicitly_disables_shuffle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = [root / f"{index:02d}.jpg" for index in range(4)]
            for path in canonical:
                path.write_bytes(path.name.encode())
            discovered = list(reversed(canonical))
            exporter, check_data, build_dataset, build_loader, seen = self.make_fake_inputs(discovered)
            ids = [f"train/images/{path.name}" for path in canonical]
            loader, audit = build_manifest_ordered_calibration_dataloader(
                exporter,
                ordered_image_ids=ids,
                expected_image_paths=canonical,
                check_det_dataset=check_data,
                build_yolo_dataset=build_dataset,
                build_dataloader=build_loader,
                expected_count=4,
            )
            self.assertIsInstance(loader, ManifestIndexedDataset)
            self.assertEqual(list(loader.indices), [3, 2, 1, 0])
            self.assertFalse(seen["shuffle"])
            self.assertEqual(seen["batch"], 1)
            self.assertEqual(seen["workers"], 0)
            self.assertTrue(seen["drop_last"])
            self.assertEqual(audit["yield_order_ids_sha256"], hashlib.sha256(__import__("json").dumps(ids, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())

    def test_rejects_missing_extra_and_duplicate_dataset_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = [root / f"{index:02d}.jpg" for index in range(4)]
            for path in canonical + [root / "extra.jpg"]:
                path.write_bytes(path.name.encode())
            ids = [f"train/images/{path.name}" for path in canonical]
            for discovered in (canonical[:-1], canonical + [root / "extra.jpg"], [canonical[0], canonical[0], *canonical[2:]]):
                exporter, check_data, build_dataset, build_loader, _seen = self.make_fake_inputs(discovered)
                with self.subTest(discovered=[path.name for path in discovered]):
                    with self.assertRaises(CalibrationOrderError):
                        build_manifest_ordered_calibration_dataloader(
                            exporter,
                            ordered_image_ids=ids,
                            expected_image_paths=canonical,
                            check_det_dataset=check_data,
                            build_yolo_dataset=build_dataset,
                            build_dataloader=build_loader,
                            expected_count=4,
                        )

    def test_rejects_invalid_manifest_duplicates_and_missing_input_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = [root / f"{index:02d}.jpg" for index in range(4)]
            for path in canonical[:-1]:
                path.write_bytes(path.name.encode())
            exporter, check_data, build_dataset, build_loader, _seen = self.make_fake_inputs(canonical)
            ids = [f"train/images/{path.name}" for path in canonical]
            with self.assertRaises(CalibrationOrderError):
                build_manifest_ordered_calibration_dataloader(
                    exporter,
                    ordered_image_ids=[ids[0], ids[0], *ids[2:]],
                    expected_image_paths=canonical,
                    check_det_dataset=check_data,
                    build_yolo_dataset=build_dataset,
                    build_dataloader=build_loader,
                    expected_count=4,
                )
            with self.assertRaises(FileNotFoundError):
                build_manifest_ordered_calibration_dataloader(
                    exporter,
                    ordered_image_ids=ids,
                    expected_image_paths=canonical,
                    check_det_dataset=check_data,
                    build_yolo_dataset=build_dataset,
                    build_dataloader=build_loader,
                    expected_count=4,
                )

    def test_rejects_changed_batch_fraction_or_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = [root / f"{index:02d}.jpg" for index in range(4)]
            for path in canonical:
                path.write_bytes(path.name.encode())
            ids = [f"train/images/{path.name}" for path in canonical]
            for field, value in (("batch", 2), ("fraction", 0.5), ("split", "train")):
                exporter, check_data, build_dataset, build_loader, _seen = self.make_fake_inputs(canonical)
                setattr(exporter.args, field, value)
                with self.subTest(field=field):
                    with self.assertRaises(CalibrationOrderError):
                        build_manifest_ordered_calibration_dataloader(
                            exporter,
                            ordered_image_ids=ids,
                            expected_image_paths=canonical,
                            check_det_dataset=check_data,
                            build_yolo_dataset=build_dataset,
                            build_dataloader=build_loader,
                            expected_count=4,
                        )

    def test_real_pinned_ultralytics_loader_returns_all_images_in_manifest_order(self):
        try:
            import torch
            import ultralytics
            import yaml
            from PIL import Image
            from ultralytics.engine import exporter as exporter_module
            from ultralytics.engine.exporter import Exporter
        except ImportError as exc:
            self.skipTest(f"real Ultralytics integration dependencies unavailable: {exc}")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1", "run this integration with CUDA hidden")
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertEqual(ultralytics.__version__, "8.4.102")
        verify_pinned_source_contract(
            ultralytics.__version__, Exporter,
            exporter_module.build_yolo_dataset,
            exporter_module.check_det_dataset,
            exporter_module.build_dataloader,
        )
        self.assertTrue(inspect.signature(exporter_module.build_dataloader).parameters["shuffle"].default)
        self.assertNotIn("shuffle=", inspect.getsource(Exporter.get_int8_calibration_dataloader))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            paths = []
            for index in range(8):
                path = images / f"{index:02d}.jpg"
                Image.new("RGB", (31 + index, 29 + index), color=(index * 20, 40, 80)).save(path)
                (labels / f"{index:02d}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
                paths.append(path.resolve())
            manifest_paths = list(reversed(paths))
            yaml_path = root / "dataset.yaml"
            yaml_path.write_text(yaml.safe_dump({"path": str(root), "train": "images", "val": "images", "names": {0: "sign"}, "nc": 1}, sort_keys=False), encoding="utf-8")
            exporter = Exporter(overrides={"format": "engine", "data": str(yaml_path), "imgsz": 640, "batch": 1, "fraction": 1.0, "split": "val", "rect": False, "device": "cpu"})
            exporter.imgsz = (640, 640)
            exporter.model = SimpleNamespace(task="detect")
            ids = [f"train/images/{path.name}" for path in manifest_paths]
            loader, evidence = build_manifest_ordered_calibration_dataloader(
                exporter,
                ordered_image_ids=ids,
                expected_image_paths=manifest_paths,
                check_det_dataset=exporter_module.check_det_dataset,
                build_yolo_dataset=exporter_module.build_yolo_dataset,
                build_dataloader=exporter_module.build_dataloader,
                expected_count=8,
            )
            observed = []
            iterator = iter(loader)
            for batch in iterator:
                observed.append(Path(batch["im_file"][0]).resolve())
                self.assertEqual(tuple(batch["img"].shape), (1, 3, 640, 640))
                self.assertEqual(batch["img"].dtype, torch.uint8)
            self.assertEqual(observed, manifest_paths)
            self.assertEqual(len(observed), 8)
            self.assertFalse(evidence["shuffle"])


if __name__ == "__main__":
    unittest.main()
