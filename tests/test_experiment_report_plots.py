import copy
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer

from experiment_report_plots import (
    COLORS, _curve_figure, _overhead_figure, _prepare, _summary_figure, render_figures,
)


def _report_data(n=3):
    seeds = list(range(501, 501+n))
    methods = list(COLORS)
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
    data = {
        "dataset_label": "Synthetic dataset", "seeds": seeds, "rounds": 2,
        "attack_start": 1, "target_source": 2, "target_label": 4, "target_count": 17,
        "methods": methods,
        "groups": [{"slug": "iid", "label": "IID", "partition": "iid",
                    "dirichlet_alpha": .5, "num_clients": 10, "ratios": ratios}],
        "scenarios": [], "curves": [], "runs": [], "health_failed_runs": 1,
    }
    for method in methods:
        for ratio in ratios:
            key = {"group_slug": "iid", "method": method, "malicious_ratio": ratio}
            row = dict(key, n=n, failed_runs=int(method == "vert" and ratio == .2),
                       failure_reasons="synthetic health failure" if method == "vert" and ratio == .2 else "")
            for metric in ("final_accuracy", "final_asr", "attack_accuracy", "attack_asr"):
                row[metric + "_mean"] = .8 if "accuracy" in metric else .1
                row[metric + "_sd"] = .03 if n > 1 else None
                if ratio == 0 and metric.startswith("attack_"):
                    row[metric + "_mean"] = row[metric + "_sd"] = None
            for metric in ("runtime_seconds", "runtime_without_crypto_seconds", "crypto_wall_seconds", "peak_rss_mib"):
                row[metric + "_mean"] = 10.
                row[metric + "_sd"] = 2. if n > 1 else None
            data["scenarios"].append(row)
            for t in range(3):
                data["curves"].append(dict(key, round=t, accuracy_mean=.2+.3*t,
                                           accuracy_sd=.03 if n > 1 else None, asr_mean=.1,
                                           asr_sd=.02 if n > 1 else None))
            for i, seed in enumerate(seeds):
                data["runs"].append(dict(key, seed=seed, runtime_seconds=8.+i*2,
                                         runtime_without_crypto_seconds=8.+i*2, peak_rss_mib=8.+i*2))
    return data


class ExperimentReportPlotsTest(unittest.TestCase):
    def test_single_seed_has_no_sd_bands_or_errorbars(self):
        data = _report_data(1)
        scenarios, curves, runs = _prepare(data)
        group = data["groups"][0]
        figures = []
        try:
            curve, _ = _curve_figure(data, group, "accuracy", scenarios, curves)
            figures.append(curve)
            self.assertEqual(len(curve.axes), 6)
            self.assertTrue(all(not ax.collections for ax in curve.axes))
            self.assertTrue(any("Single seed 501" in text.get_text() for text in curve.texts))
            for make in (
                lambda: _summary_figure(data, group, scenarios),
                lambda: _overhead_figure(data, group, "runtime_seconds", scenarios, runs),
            ):
                figure, _ = make()
                figures.append(figure)
                self.assertFalse(any(isinstance(container, ErrorbarContainer)
                                     for ax in figure.axes for container in ax.containers))
        finally:
            for figure in figures:
                plt.close(figure)

    def test_full_export_preserves_palette_failed_observations_and_pdf_pages(self):
        data = _report_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = render_figures(data, root, png_dpi=35)
            self.assertEqual(len(entries), 6)
            self.assertEqual([entry["pdf_page"] for entry in entries], list(range(1, 7)))
            self.assertTrue((root / "mean_figures.pdf").read_bytes().startswith(b"%PDF-"))
            for entry in entries:
                self.assertTrue((root / entry["png"]).read_bytes().startswith(b"\x89PNG"))
                svg = (root / entry["svg"]).read_text()
                for color in COLORS.values():
                    self.assertIn(color.lower(), svg.lower())
                self.assertIn("Health-failed runs retained: 1/90 (VERT 1)", svg)
                self.assertNotIn("TAD at 80%", svg)
            accuracy = (root / entries[0]["svg"]).read_text()
            self.assertIn("PolyCollection", accuracy)
            self.assertIn("Attack starts at round 1", accuracy)
            self.assertIn("Synthetic dataset", accuracy)
            rss = (root / entries[-1]["svg"]).read_text()
            self.assertIn("not GPU memory", rss)

    def test_per_seed_export_omits_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = render_figures(_report_data(1), Path(directory), include_pdf=False, png_dpi=35)
            self.assertEqual(len(entries), 6)
            self.assertTrue(all(entry["pdf_page"] is None for entry in entries))
            self.assertFalse((Path(directory) / "mean_figures.pdf").exists())

    def test_clean_scenario_has_no_attack_window_and_is_not_plotted_in_window_panels(self):
        data = _report_data()
        scenarios, _, _ = _prepare(data)
        figure, _ = _summary_figure(data, data["groups"][0], scenarios)
        try:
            self.assertIn(0, figure.axes[0].lines[0].get_xdata())
            for ax in figure.axes[1:]:
                self.assertNotIn(0, ax.lines[0].get_xdata())
        finally:
            plt.close(figure)
        attacked = next(row for row in data["scenarios"] if row["malicious_ratio"] > 0)
        attacked["attack_accuracy_mean"] = attacked["attack_accuracy_sd"] = None
        with self.assertRaises(ValueError):
            _prepare(data)

    def test_footer_with_failures_in_every_method_fits_inside_all_figure_layouts(self):
        data = _report_data()
        for row in data["scenarios"]:
            row["failed_runs"] = row["n"]
        scenarios, curves, runs = _prepare(data)
        group = data["groups"][0]
        for make in (
            lambda: _curve_figure(data, group, "accuracy", scenarios, curves),
            lambda: _summary_figure(data, group, scenarios),
            lambda: _overhead_figure(data, group, "runtime_without_crypto_seconds", scenarios, runs),
        ):
            figure, _ = make()
            try:
                figure.canvas.draw()
                footer = next(text for text in figure.texts if text.get_position()[1] == .033)
                bounds = footer.get_window_extent(figure.canvas.get_renderer())
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, figure.bbox.width)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLess(bounds.y1, min(ax.bbox.y0 for ax in figure.axes))
            finally:
                plt.close(figure)

    def test_incomplete_or_nonfinite_data_fail_before_export(self):
        cases = []
        missing = _report_data()
        missing["curves"].pop()
        cases.append(missing)
        nonfinite = _report_data()
        nonfinite["scenarios"][0]["runtime_seconds_mean"] = float("nan")
        cases.append(nonfinite)
        wrong_sd = _report_data(1)
        wrong_sd["scenarios"][0]["final_accuracy_sd"] = 0.
        cases.append(wrong_sd)
        for data in cases:
            with self.subTest(data=data["seeds"]), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "new-report"
                with self.assertRaises(ValueError):
                    render_figures(copy.deepcopy(data), destination, png_dpi=35)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
