import copy
import json
import sys
import tempfile
import unittest
import zipfile
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_cctsdb_measurement import load_records, load_labels, load_xml, match_rows, size_bin, sha256, lf_sha256, audit, replay_ultralytics


class MeasurementAuditTests(unittest.TestCase):
    def record(self):
        return {"image": "a.jpg", "orig_shape": [100, 100], "xyxy": [[1, 2, 3, 4]], "confidence": [0.9], "class_id": [0]}

    def test_duplicate_id_rejected(self):
        r = self.record()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_records({"records": [r, r]})

    def test_duplicate_stem_rejected(self):
        r = self.record()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_records({"records": [r, dict(r, image="a.png")]})

    def test_lengths_nan_class_shape_path_rejected(self):
        for key, value in [("confidence", []), ("confidence", [float("nan")]), ("class_id", [3]), ("orig_shape", [0, 10]), ("image", "../a.jpg"), ("xyxy", [[2, 2, 1, 4]])]:
            r = self.record()
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_records({"records": [r]})

    def test_empty_detections_valid(self):
        r = dict(self.record(), xyxy=[], confidence=[], class_id=[])
        self.assertEqual(len(load_records({"records": [r]})), 1)

    def test_zero_area_clipped_detection_retained(self):
        r = dict(self.record(), xyxy=[[1, 0, 3, 0]])
        self.assertEqual(load_records({"records": [r]})["a.jpg"]["xyxy"], r["xyxy"])

    def test_boundaries(self):
        self.assertEqual([size_bin(v) for v in (210, 210.001, 400, 400.001, 1000, 1000.001, 2000, 2000.001)], ["xs", "s", "s", "m", "m", "l", "l", "xl"])

    def test_matching_reordered_and_duplicate(self):
        a, b = (0, [0, 0, 10, 10]), (1, [1, 1, 20, 20])
        self.assertIsNotNone(match_rows([a, b], [b, a], 0.05))
        self.assertIsNone(match_rows([a, a], [a, b], 0.05))
        self.assertIsNone(match_rows([a], [], 0.05))

    def test_bipartite_not_greedy(self):
        left = [(0, [0.04, 0, 1, 1]), (0, [0, 0, 1, 1])]
        right = [(0, [0, 0, 1, 1]), (0, [0.08, 0, 1, 1])]
        self.assertIsNotNone(match_rows(left, right, 0.05))

    def test_label_roundtrip_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("1 0.5 0.5 0.2 0.4\n")
            self.assertEqual(load_labels(p, [100, 200]), [(1, [80, 30, 120, 70])])
            p.write_text("0 0.5 0.5 0 0.1\n")
            with self.assertRaises(ValueError):
                load_labels(p, [100, 200])
            with self.assertRaises(FileNotFoundError):
                load_labels(Path(d) / "missing.txt", [100, 200])

    def test_xml_member_selection_and_duplicate(self):
        xml = '<annotation><filename>wrong.jpg</filename><size><height>100</height><width>100</width></size><object><name>prohibitory</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object></annotation>'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "xml.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("xml/a.xml", xml)
                z.writestr("xml/unrelated.xml", "invalid XML outside requested split")
            result = load_xml(p, {"a.jpg": self.record()})
            self.assertEqual(result["a.jpg"]["rows"], [(0, [1, 2, 3, 4])])
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_xml(p, {"missing.jpg": self.record()})
            with zipfile.ZipFile(p, "a") as z:
                z.writestr("another/a.xml", xml)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_xml(p, {"a.jpg": self.record()})

    def test_crlf_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            lf, crlf = Path(d)/"lf", Path(d)/"crlf"
            lf.write_bytes(b'{"a": 1}\n')
            crlf.write_bytes(b'{"a": 1}\r\n')
            self.assertNotEqual(sha256(lf), sha256(crlf))
            self.assertEqual(sha256(lf), lf_sha256(crlf))

    def test_integration_and_hash_rejection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"images").mkdir()
            (root/"labels").mkdir()
            Image.new("RGB", (100, 100)).save(root/"images/a.jpg")
            (root/"labels/a.txt").write_text("0 0.02 0.03 0.02 0.02\n")
            p, e = root/"p.json", root/"e.json"
            payload = {"records": [self.record()], "model_sha256": "abc", "data_sha256": "def"}
            p.write_text(json.dumps(payload))
            summary = {"model_sha256": "abc", "data_sha256": "def", "per_image_predictions": {"sha256": sha256(p)}}
            e.write_text(json.dumps(summary))
            r = audit(p, e, root)
            self.assertEqual(r["instances"], 1)
            self.assertEqual(r["status"], "review_required")
            summary["model_sha256"] = "other"
            e.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "model_sha256"):
                audit(p, e, root)

    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Optional pinned Ultralytics replay environment")
    def test_cpu_replay_perfect_and_empty(self):
        import ultralytics
        r = self.record()
        truth = {"a.jpg": [(0, [1, 2, 3, 4])]}
        result = replay_ultralytics({"a.jpg": r}, truth, ultralytics.__version__)
        self.assertGreater(result["map50"], 0.99)
        self.assertAlmostEqual(result["map50"], result["map50_95"])
        r.update(xyxy=[], confidence=[], class_id=[])
        result = replay_ultralytics({"a.jpg": r}, truth, ultralytics.__version__)
        self.assertEqual(result["map50"], 0)
        with self.assertRaisesRegex(ValueError, "requires recorded"):
            replay_ultralytics({"a.jpg": r}, truth, "not-installed")


if __name__ == "__main__":
    unittest.main()
