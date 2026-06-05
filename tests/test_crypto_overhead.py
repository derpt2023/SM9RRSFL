import tempfile
import unittest
from pathlib import Path

from sm9rrsfl.benchmarks.crypto_overhead import CryptoOverheadConfig, run_benchmarks, write_outputs


class CryptoOverheadBenchmarkTest(unittest.TestCase):
    def test_simulated_benchmark_records_sign_and_verify_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = CryptoOverheadConfig(
                client_counts=(3, 5),
                iterations=2,
                warmup=1,
                update_size=8,
                crypto_mode="simulated",
                output_dir=Path(tmp),
                seed=7,
            )

            result = run_benchmarks(config)
            dashboard = write_outputs(config, result)

            self.assertEqual(len(result.summaries), 2)
            self.assertEqual(len(result.samples), 4)
            for summary in result.summaries:
                self.assertEqual(summary.verify_successes, 2)
                self.assertGreaterEqual(summary.sign_mean_ms, 0.0)
                self.assertGreaterEqual(summary.verify_mean_ms, 0.0)
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            self.assertTrue((Path(tmp) / "samples.csv").exists())
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertEqual(dashboard, Path(tmp) / "visualizations.html")
            self.assertTrue((Path(tmp) / "visualizations.html").exists())
            self.assertTrue((Path(tmp) / "plots" / "mean_operation_overhead.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "signature_overhead.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "verification_overhead.svg").exists())
            self.assertTrue((Path(tmp) / "plots" / "setup_and_cache_overhead.svg").exists())


if __name__ == "__main__":
    unittest.main()
