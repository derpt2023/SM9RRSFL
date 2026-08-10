import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord
from sm9rrsfl.visualization import generate_visualizations


class VisualizationTest(unittest.TestCase):
    def test_generates_dashboard_and_svg_files(self):
        results = [
            _fake_result("sm9rrs", 0.0, 1.2, 3.4),
            _fake_result("vert", 0.0, 1.1, 3.0),
            _fake_result("fedredefense", 0.0, 2.0, 4.0),
            _fake_result("krum", 0.0, 0.8, 2.1),
            _fake_result("ding13", 0.1, 0.6, 1.9),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dashboard = generate_visualizations(results, tmp)
            self.assertTrue(dashboard.exists())
            self.assertTrue((Path(tmp) / "plots" / "accuracy_comparison.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "runtime_overhead.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "runtime_without_crypto.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "memory_overhead.svg").exists())
            comparison = (
                Path(tmp) / "plots" / "accuracy_comparison.svg"
            ).read_text(encoding="utf-8")
            self.assertIn(">Ours<", comparison)
            self.assertIn(">VERT<", comparison)
            self.assertIn(">FedREDefense<", comparison)
            self.assertIn("(a) 恶意节点比例 0%", comparison)
            self.assertIn("(b) 恶意节点比例 10%", comparison)
            self.assertIn('data-series="accuracy"', comparison)
            svg = ET.fromstring(comparison)
            namespace = "{http://www.w3.org/2000/svg}"
            ours = next(
                node
                for node in svg.iter(f"{namespace}polyline")
                if node.get("data-series") == "accuracy"
                and node.get("data-method") == "Ours"
                and node.get("data-malicious-ratio") == "0"
            )
            self.assertEqual(len(ours.get("points").split()), 3)
            self.assertNotIn('data-series="accuracy-padded"', comparison)
            runtime = (
                Path(tmp) / "plots" / "runtime_overhead.svg"
            ).read_text(encoding="utf-8")
            self.assertIn(">TAD<", runtime)
            self.assertNotIn("文献 [13]", runtime)
            self.assertNotIn(">SM9-RRS-FL<", comparison)
            fair_runtime = (
                Path(tmp) / "plots" / "runtime_without_crypto.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("扣除密码协议", fair_runtime)

    def test_generates_partition_specific_files_for_multiple_scenarios(self):
        results = [
            _fake_result("sm9rrs", 0.0, 1.2, 3.4, partition="iid"),
            _fake_result("krum", 0.0, 0.8, 2.1, partition="iid"),
            _fake_result("sm9rrs", 0.0, 1.4, 3.6, partition="dirichlet"),
            _fake_result("krum", 0.0, 0.9, 2.3, partition="dirichlet"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            generate_visualizations(results, tmp)
            self.assertTrue((Path(tmp) / "plots" / "iid_accuracy_comparison.svg").exists())
            self.assertTrue(
                (Path(tmp) / "plots" / "dirichlet_alpha_0_5_accuracy_comparison.svg").exists()
            )

    def test_generates_client_count_comparison_files(self):
        results = [
            _fake_result("sm9rrs", 0.0, 1.2, 3.4, num_clients=20),
            _fake_result("krum", 0.0, 0.8, 2.1, num_clients=20),
            _fake_result("sm9rrs", 0.0, 2.4, 4.4, num_clients=50),
            _fake_result("krum", 0.0, 1.6, 3.1, num_clients=50),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            generate_visualizations(results, tmp)
            self.assertTrue((Path(tmp) / "plots" / "iid_clients_020_accuracy_comparison.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "iid_clients_050_accuracy_comparison.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "client_count_accuracy_ratio_000.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "client_count_runtime_ratio_000.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "client_count_memory_ratio_000.svg").exists())


def _fake_result(method, ratio, runtime, memory, partition="iid", num_clients=20):
    config = ExperimentConfig(
        method=method,
        malicious_ratio=ratio,
        partition=partition,
        num_clients=num_clients,
    )
    records = [
        RoundRecord(method, ratio, 0, 0.1, 0.9, 0, 0, 0, 0, 0, ""),
        RoundRecord(method, ratio, 1, 0.4, 0.6, 3, 0, 0, 0, 0, ""),
        RoundRecord(method, ratio, 2, 0.7, 0.3, 3, 0, 0, 0, 0, ""),
    ]
    return ExperimentResult(
        config=config,
        records=records,
        final_accuracy=0.7,
        final_error=0.3,
        stopped_round=2,
        malicious_clients=tuple(),
        blacklisted_clients=tuple(),
        runtime_seconds=runtime,
        peak_memory_mb=memory,
    )


if __name__ == "__main__":
    unittest.main()
