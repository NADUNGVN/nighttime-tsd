import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from edge_readiness.collect_inventory import (  # noqa: E402
    BEGIN,
    END,
    REMOTE_SCRIPT,
    command_status,
    parse_remote_output,
    validate_target,
)


class EdgeReadinessTests(unittest.TestCase):
    def test_target_validation_is_alias_safe(self):
        self.assertEqual(validate_target("E1"), "E1")
        self.assertEqual(validate_target("orin-nano_super"), "orin-nano_super")
        for value in ("../E1", "E1 device", "E1;rm", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_target(value)

    def test_cpu_fixture_parses_ok_missing_and_permission_states(self):
        raw = "\n".join(
            [
                f"{BEGIN} hostname",
                "fixture-host",
                f"{END} hostname 0",
                f"{BEGIN} nvidia_smi",
                f"{END} nvidia_smi 127",
                f"{BEGIN} power_mode",
                "permission denied",
                f"{END} power_mode 126",
            ]
        )
        records = parse_remote_output(raw)
        self.assertEqual(records["hostname"]["status"], "ok")
        self.assertEqual(records["nvidia_smi"]["status"], "missing")
        self.assertEqual(records["power_mode"]["status"], "permission_denied")

    def test_parser_rejects_unterminated_fixture(self):
        with self.assertRaisesRegex(ValueError, "unterminated"):
            parse_remote_output(f"{BEGIN} hostname\nfixture-host")

    def test_command_status(self):
        self.assertEqual(command_status(0), "ok")
        self.assertEqual(command_status(127), "missing")
        self.assertEqual(command_status(126), "permission_denied")
        self.assertEqual(command_status(2), "unavailable")
        self.assertEqual(command_status(124), "timeout")
        self.assertEqual(command_status(None, timed_out=True), "timeout")

    def test_remote_script_has_no_mutating_or_benchmark_actions(self):
        script = REMOTE_SCRIPT.lower()
        forbidden = (
            "apt", "apt-get", "pip", "sudo", "reboot", "poweroff", "shutdown",
            "nvpmodel -m", "jetson_clocks --fan", "trtexec --onnx", "hailo\n",
            "benchmark", "inference", "compile", "calibration",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, script)


if __name__ == "__main__":
    unittest.main()
