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
    validate_ssh_alias,
    validate_target,
)
from edge_readiness.collect_telemetry import (  # noqa: E402
    REMOTE_SCRIPT as TELEMETRY_REMOTE_SCRIPT,
    parse_channel_observations,
    parse_sample_blocks,
    summarize_samples,
)


class EdgeReadinessTests(unittest.TestCase):
    def test_target_validation_is_alias_safe(self):
        self.assertEqual(validate_target("E1"), "E1")
        self.assertEqual(validate_target("orin-nano_super"), "orin-nano_super")
        for value in ("../E1", "E1 device", "E1;rm", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_target(value)

    def test_ssh_alias_rejects_options_and_shellish_values(self):
        self.assertEqual(validate_ssh_alias("pi5"), "pi5")
        self.assertEqual(validate_ssh_alias("edge.host-1"), "edge.host-1")
        for value in ("-oProxyCommand=x", "--", "pi 5", "pi;echo", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ssh_alias(value)

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

    def test_parser_rejects_duplicate_unexpected_and_missing_markers(self):
        raw = "\n".join(
            [
                f"{BEGIN} hostname", "one", f"{END} hostname 0",
                f"{BEGIN} hostname", "two", f"{END} hostname 0",
                f"{BEGIN} surprise", "x", f"{END} surprise 0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_remote_output(raw, {"hostname", "memory"})
        unique = "\n".join(
            [f"{BEGIN} hostname", "one", f"{END} hostname 0", f"{BEGIN} surprise", "x", f"{END} surprise 0"]
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            parse_remote_output(unique, {"hostname", "memory"})
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_remote_output(f"{BEGIN} hostname\none\n{END} hostname 0", {"hostname", "memory"})

    def test_parser_preserves_partial_timeout_evidence(self):
        raw = f"{BEGIN} hostname\nfixture-host"
        records = parse_remote_output(raw, {"hostname", "memory"}, strict=False)
        self.assertEqual(records["hostname"]["status"], "partial")
        self.assertIsNone(records["hostname"]["returncode"])

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

    def test_telemetry_parser_preserves_raw_sources_and_device_timestamps(self):
        raw = "\n".join(
            [
                "__E2L1_SOURCE__ command=source_inventory",
                "command:tegrastats=/usr/bin/tegrastats",
                "__E2L1_SAMPLE__ device_monotonic_ns=100",
                "__E2L1_SOURCE__ source=free_bytes",
                "Mem: 100 20 80 1 19 79",
                "__E2L1_SAMPLE__ device_monotonic_ns=110",
                "__E2L1_SOURCE__ source=thermal_sysfs",
                "thermal_zone0/temp: 42000",
            ]
        )
        samples = parse_sample_blocks(raw)
        self.assertEqual([sample["device_monotonic_ns"] for sample in samples], [100, 110])
        self.assertIn("Mem: 100 20 80", samples[0]["raw"])
        summary = summarize_samples(raw)
        self.assertEqual(summary["timestamp_origin"].split(";")[0], "device_monotonic_ns from device python3 time.monotonic_ns")
        self.assertEqual(summary["timestamp_order"], "strictly_increasing")
        self.assertEqual(summary["max_gap_ns"], 10)

    def test_telemetry_parser_handles_actual_tegrastats_labels_without_summing_rails(self):
        raw = "\n".join(
            [
                "__E2L1_SAMPLE__ device_monotonic_ns=100",
                "__E2L1_SOURCE__ source=tegrastats",
                "RAM 2196/7607MB (lfb 17x4MB) SWAP 1/3804MB CPU [0%@729] cpu@50.5C gpu@51.0C VDD_IN 4672mW/4672mW VDD_SOC 1437mW/1437mW",
                "__E2L1_SOURCE__ source=vcgencmd",
                "temp=49.4'C",
                "throttled=0x0",
            ]
        )
        channels = parse_channel_observations(raw)
        self.assertEqual(channels["memory"][0]["unit"], "MB")
        self.assertEqual({observation["label"] for observation in channels["power"]}, {"VDD_IN", "VDD_SOC"})
        self.assertTrue(all(observation["boundary"] == "unknown_rail_boundary" for observation in channels["power"]))
        self.assertEqual(channels["throttle"][0]["raw_value"], "0x0")

    def test_telemetry_remote_script_is_bounded_read_only(self):
        script = TELEMETRY_REMOTE_SCRIPT.lower()
        forbidden = ("apt", "apt-get", "pip", "sudo", "reboot", "poweroff", "shutdown", "nvpmodel -m", "jetson_clocks", "trtexec", "hailo", "inference", "benchmark", "kill", "pkill", "systemctl")
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, script)
        self.assertIn("timeout 2s tegrastats", script)


if __name__ == "__main__":
    unittest.main()
