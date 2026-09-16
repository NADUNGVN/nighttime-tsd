import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import audit_precision_head_confirmation_graph as audit
import test_precision_head_confirmation_graph as graph_tests
import prepare_precision_head_confirmation_graph as graph


class PreservedGraphAuditTests(unittest.TestCase):
    def _accepted(self, model):
        return {
            "root": "accepted-readiness",
            "accepted_git_commit": "accepted-commit",
            "accepted_execution_commit": "accepted-execution",
            "files": {},
            "manifest": {"model_contracts": [model]},
        }

    def test_audit_reads_existing_graph_only_and_persists_unresolved_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "models" / "yolov8n"
            source.mkdir(parents=True)
            onnx_path = source / "model.onnx"
            onnx_path.write_bytes(b"preserved-onnx")
            (source / "model_prepare.json").write_text(
                json.dumps({"status": "completed", "export": {"onnx_sha256": "historical"}}),
                encoding="utf-8",
            )
            (root / "source" / "prepare_plan.json").write_text("{}", encoding="utf-8")
            fixture, model = graph_tests.graph_fixture("yolov8n", shared_downstream=True)
            fixture["nodes"][-1]["op_type"] = "UnknownSelection"
            args = SimpleNamespace(
                model="yolov8n",
                source_root=Path("source"),
                readiness_root=Path("readiness"),
                out_dir=Path("audit"),
            )
            accepted = self._accepted(model)
            binding = {"path": str(root / "config.json"), "sha256": "config", "semantic_sha256": "semantic"}
            with patch.object(audit.graph, "validate_readiness_artifact", return_value=accepted), \
                 patch.object(audit.graph, "validate_config_binding", return_value=binding), \
                 patch.object(audit.graph, "_load_onnx", return_value=(fixture, {"node_count": len(fixture["nodes"])})), \
                 patch.object(audit.graph, "default_exporter", Mock()) as exporter, \
                 patch.object(audit.graph, "prepare_model", Mock()) as prepare_model:
                result = audit.run_audit(args, root)

            self.assertEqual(
                result,
                0,
                (root / "audit/graph_audit_manifest.json").read_text(encoding="utf-8")
                if (root / "audit/graph_audit_manifest.json").is_file() else "manifest missing",
            )
            self.assertFalse(exporter.called)
            self.assertFalse(prepare_model.called)
            output = root / "audit"
            self.assertTrue((output / "audit_plan.json").is_file())
            self.assertTrue((output / "graph_audit_manifest.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            row = json.loads((output / "models/yolov8n/graph_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "audit_only_completed")
            self.assertEqual(row["mapping_status"], "mapping_unresolved")
            self.assertTrue(row["mapping_errors"])
            self.assertTrue(row["audit_flags"]["audit_only"])
            self.assertFalse(row["audit_flags"]["export_performed"])
            self.assertFalse(row["audit_flags"]["build_performed"])
            self.assertFalse(row["audit_flags"]["scored_run_authorized"])
            before = row["provenance"]["onnx"]["before"]["sha256"]
            after = row["provenance"]["onnx"]["after"]["sha256"]
            self.assertEqual(before, after)
            self.assertTrue(row["provenance"]["onnx"]["unchanged_during_audit"])
            self.assertTrue(onnx_path.read_bytes() == b"preserved-onnx")

    def test_missing_existing_onnx_stops_without_creating_output_or_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source/models/yolov8n").mkdir(parents=True)
            args = SimpleNamespace(
                model="yolov8n",
                source_root=Path("source"),
                readiness_root=Path("readiness"),
                out_dir=Path("audit"),
            )
            with patch.object(audit.graph, "default_exporter", Mock()) as exporter:
                with self.assertRaises(FileNotFoundError):
                    audit.run_audit(args, root)
            self.assertFalse(exporter.called)
            self.assertFalse((root / "audit").exists())

    def test_audit_source_has_no_model_dispatch_or_exporter_call(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".export(", source)
        self.assertNotIn("prepare_model(", source)
        self.assertNotIn("YOLO(", source)
        self.assertIn("audit_only", source)


if __name__ == "__main__":
    unittest.main()
