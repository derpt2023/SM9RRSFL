"""Render experiment reports from validated, seed-aggregated plain dictionaries.

This module only plots supplied observations. It does not train, load checkpoints,
exclude failed runs, or recompute experiment statistics. Probability inputs use
the 0--1 scale; their plots use percentages. Standard deviations are sample SDs.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


LABELS = {
    "sm9rrs": "Ours", "vert": "VERT", "alignins": "AlignIns",
    "krum": "Krum", "ding13": "TAD", "fedavg": "FedAvg",
}
# These exact colors also identify the methods in the original seed reports.
COLORS = dict(zip(LABELS, (
    "#8DBAD9", "#E99092", "#8ADDE4", "#DCDD8F", "#FEBD85", "#C4A9A2",
)))
STYLES = dict(zip(LABELS, (
    "-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (8, 3)),
)))
MARKERS = dict(zip(LABELS, ("o", "s", "^", "D", "v", "x")))
PROBABILITY_METRICS = ("final_accuracy", "final_asr", "attack_accuracy", "attack_asr")
OVERHEAD_METRICS = (
    "runtime_seconds", "runtime_without_crypto_seconds", "crypto_wall_seconds", "peak_rss_mib",
)


def _key(row):
    return row["group_slug"], row["method"], float(row["malicious_ratio"])


def _finite(value, field, *, probability=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(number) or number < 0 or (probability and number > 1):
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def _check_sd(value, n, field):
    if n == 1:
        if value is not None:
            raise ValueError(f"{field} must be None for a single seed")
    elif value is None:
        raise ValueError(f"{field} is missing for n={n}")
    else:
        _finite(value, field)


def _prepare(data):
    """Fail before writing files if the plot matrix or observations are invalid."""
    methods, groups = data["methods"], data["groups"]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a nonempty list without duplicates")
    if set(methods) - set(LABELS):
        raise ValueError(f"No plotting style for methods: {sorted(set(methods) - set(LABELS))}")
    seeds = data["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list without duplicates")
    rounds = int(data["rounds"])
    if rounds < 0 or rounds != data["rounds"]:
        raise ValueError("rounds must be a nonnegative integer")
    if not groups or len({g["slug"] for g in groups}) != len(groups):
        raise ValueError("groups must be nonempty and have unique slugs")
    expected = set()
    for group in groups:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", group["slug"]):
            raise ValueError(f"Unsafe group slug: {group['slug']!r}")
        ratios = [float(r) for r in group["ratios"]]
        if not ratios or ratios != sorted(set(ratios)):
            raise ValueError("Group ratios must be nonempty, increasing and unique")
        for ratio in ratios:
            _finite(ratio, "malicious_ratio", probability=True)
            expected.update((group["slug"], method, ratio) for method in methods)
    scenarios = {}
    for row in data["scenarios"]:
        key = _key(row)
        if key in scenarios:
            raise ValueError(f"Duplicate scenario: {key}")
        n = int(row["n"])
        failed = int(row["failed_runs"])
        if n != row["n"] or not 1 <= n <= len(seeds) or not 0 <= failed <= n:
            raise ValueError(f"Invalid seed/failure count for {key}")
        for metric in PROBABILITY_METRICS + OVERHEAD_METRICS:
            # A clean scenario has no attack window. Do not invent a window
            # statistic merely because rounds share the same numerical range.
            if (key[2] == 0 and metric in ("attack_accuracy", "attack_asr")
                    and row[metric + "_mean"] is None and row[metric + "_sd"] is None):
                continue
            _finite(row[metric + "_mean"], metric, probability=metric in PROBABILITY_METRICS)
            _check_sd(row[metric + "_sd"], n, metric + "_sd")
        scenarios[key] = row
    if set(scenarios) != expected:
        raise ValueError("Scenario rows do not match the group/method/ratio matrix")
    curves = {key: {} for key in expected}
    for row in data["curves"]:
        key, t = _key(row), int(row["round"])
        if key not in curves or t in curves[key] or t != row["round"]:
            raise ValueError(f"Unexpected or duplicate curve row: {key}, round {t}")
        for metric in ("accuracy", "asr"):
            _finite(row[metric + "_mean"], metric, probability=True)
            _check_sd(row[metric + "_sd"], scenarios[key]["n"], metric + "_sd")
        curves[key][t] = row
    if any(set(rows) != set(range(rounds + 1)) for rows in curves.values()):
        raise ValueError("Each curve must contain every round from 0 through rounds")
    runs = {key: {} for key in expected}
    for row in data["runs"]:
        key, seed = _key(row), row["seed"]
        if key not in runs or seed in runs[key] or seed not in seeds:
            raise ValueError(f"Unexpected or duplicate individual run: {key}, seed {seed}")
        for metric in ("runtime_seconds", "runtime_without_crypto_seconds", "peak_rss_mib"):
            _finite(row[metric], metric)
        runs[key][seed] = row
    if any(len(rows) != scenarios[key]["n"] for key, rows in runs.items()):
        raise ValueError("Individual run counts do not match scenario sample sizes")
    return scenarios, curves, runs


def _group_scenarios(group, scenarios):
    return [row for key, row in scenarios.items() if key[0] == group["slug"]]


def _sample_note(data, group, scenarios):
    counts = sorted({row["n"] for row in _group_scenarios(group, scenarios)})
    if counts == [1]:
        seed = f" {data['seeds'][0]}" if len(data["seeds"]) == 1 else " per scenario"
        return f"Single seed{seed}; sample SD unavailable"
    count = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    seeds = ", ".join(str(seed) for seed in data["seeds"])
    return f"Mean ± 1 sample SD; seeds {seeds} (n={count})"


def _health_note(group, scenarios):
    rows = _group_scenarios(group, scenarios)
    total = sum(row["n"] for row in rows)
    failed = sum(row["failed_runs"] for row in rows)
    if not failed:
        return f"Health-failed runs: 0/{total}. All observations retained."
    counts = {}
    for row in rows:
        if row["failed_runs"]:
            counts[row["method"]] = counts.get(row["method"], 0) + row["failed_runs"]
    details = ", ".join(f"{LABELS[m]} {counts[m]}" for m in LABELS if m in counts)
    return f"Health-failed runs retained: {failed}/{total} ({details})."


def _handles(methods):
    return [Line2D([0], [0], color=COLORS[m], linestyle=STYLES[m],
                   marker=MARKERS[m], markersize=3, linewidth=1.3, label=LABELS[m])
            for m in methods]


def _style_axis(ax, *, probability=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DEE3", linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)
    if probability:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])


def _footer(fig, lines, *, x=.08, width=116):
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width))
    fig.text(x, .033, "\n".join(wrapped), fontsize=6.7,
             color="#555555", linespacing=1.4, va="bottom")


def _title(data, group, suffix):
    return f"{data['dataset_label']} | {group['label']} | {suffix}"


def _curve_figure(data, group, metric, scenarios, curves):
    ratios, methods = group["ratios"], data["methods"]
    ncols = min(3, len(ratios) + 1)
    nrows = math.ceil((len(ratios) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.05*nrows + .9), squeeze=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.17, top=.84, wspace=.31, hspace=.57)
    is_accuracy = metric == "accuracy"
    title = _title(data, group, "Accuracy" if is_accuracy else "ASR")
    fig.suptitle(title, y=.975, fontsize=10.5)
    fig.text(.5, .922, _sample_note(data, group, scenarios), ha="center", fontsize=8, color="#555555")
    t = np.arange(data["rounds"] + 1)
    for i, ratio in enumerate(ratios):
        ax = axes.flat[i]
        for j, method in enumerate(methods):
            key = (group["slug"], method, float(ratio))
            rows = curves[key]
            mean = np.array([rows[r][metric + "_mean"] for r in t]) * 100
            if scenarios[key]["n"] > 1:
                sd = np.array([rows[r][metric + "_sd"] for r in t]) * 100
                ax.fill_between(t, np.clip(mean-sd, 0, 100), np.clip(mean+sd, 0, 100),
                                color=COLORS[method], alpha=.20, linewidth=0, zorder=1)
            ax.plot(t, mean, color=COLORS[method], linestyle=STYLES[method],
                    linewidth=1.35 if method == "sm9rrs" else 1.05,
                    marker=MARKERS[method], markersize=2.2,
                    markevery=(j % max(1, len(t)), max(1, len(t)//6)),
                    markerfacecolor="white", markeredgewidth=.65,
                    zorder=6 if method == "sm9rrs" else 5 if method == "vert" else 3)
        if ratio > 0 and 0 <= data["attack_start"] <= data["rounds"]:
            ax.axvline(data["attack_start"], color="#777777", linestyle=(0, (2, 3)), linewidth=.7, zorder=0)
        ratio_label = "0% (clean)" if ratio == 0 else f"{ratio*100:g}% malicious"
        ax.set_title(f"({chr(97+i)}) {ratio_label}", fontsize=8.8, loc="left", pad=6)
        ax.set_xlim(0, max(1, data["rounds"]))
        ax.set_xlabel("Communication round", labelpad=3)
        if i % ncols == 0:
            ax.set_ylabel("Test accuracy (%)" if is_accuracy else "Target-label rate / ASR (%)", labelpad=3)
        _style_axis(ax, probability=True)
    legend_ax = axes.flat[len(ratios)]
    legend_ax.axis("off")
    legend_ax.legend(handles=_handles(methods), loc="upper left", frameon=False, ncol=2,
                     fontsize=7.8, handlelength=2.3, columnspacing=1, labelspacing=.8)
    note = ("Shading: ±1 sample SD;\nabsent for a single seed.\n"
            f"Attack starts at round {data['attack_start']}\n(dotted line); clean at 0%.")
    if not is_accuracy:
        note += f"\nTarget: {data['target_source']} to {data['target_label']} (n={data['target_count']})."
    legend_ax.text(.02, .43, note, transform=legend_ax.transAxes, fontsize=7,
                   va="top", linespacing=1.45, color="#444444")
    for ax in list(axes.flat)[len(ratios)+1:]:
        ax.axis("off")
    _footer(fig, ["SD bands clipped to 0-100%; no smoothing. Clean target-label rate is not attack success."
                  if not is_accuracy else "SD bands clipped to 0-100%; no smoothing.",
                  _health_note(group, scenarios)])
    return fig, title


def _summary_figure(data, group, scenarios):
    methods = data["methods"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.08, right=.985, bottom=.22, top=.78, wspace=.27, hspace=.65)
    title = _title(data, group, "Final and attack-window results")
    fig.suptitle(title, y=.976, fontsize=10.3)
    fig.text(.5, .926, _sample_note(data, group, scenarios), ha="center", fontsize=7.8, color="#555555")
    fig.legend(handles=_handles(methods), loc="upper center", bbox_to_anchor=(.53, .885),
               ncol=min(6, len(methods)), frameon=False, handlelength=2.2, columnspacing=1, fontsize=7.8)
    for i, (metric, subtitle, ylabel) in enumerate([
        ("final_accuracy", f"Final accuracy (round {data['rounds']})", "Test accuracy (%)"),
        ("final_asr", f"Final ASR (round {data['rounds']})", "ASR (%)"),
        ("attack_accuracy", f"Attack-window accuracy ({data['attack_start']}-{data['rounds']})", "Test accuracy (%)"),
        ("attack_asr", f"Attack-window ASR ({data['attack_start']}-{data['rounds']})", "ASR (%)"),
    ]):
        ax = axes.flat[i]
        ratios = [r for r in group["ratios"] if metric == "final_accuracy" or r > 0]
        x = np.array(ratios) * 100
        for method in methods:
            rows = [scenarios[(group["slug"], method, float(r))] for r in ratios]
            mean = np.array([row[metric + "_mean"] for row in rows]) * 100
            ax.plot(x, mean, color=COLORS[method], linestyle=STYLES[method], linewidth=1.2,
                    marker=MARKERS[method], markersize=3, markerfacecolor="white", markeredgewidth=.7,
                    zorder=6 if method == "sm9rrs" else 5 if method == "vert" else 3)
            indices = [j for j, row in enumerate(rows) if row["n"] > 1]
            if indices:
                ax.errorbar(x[indices], mean[indices],
                            yerr=[rows[j][metric + "_sd"]*100 for j in indices],
                            fmt="none", ecolor=COLORS[method], elinewidth=.55, capsize=1.6, capthick=.7)
        ax.set_title(f"({chr(97+i)}) {subtitle}", loc="left", fontsize=8.1)
        ax.set_xticks(x)
        if len(x):
            ax.set_xlim(float(x.min())-3, float(x.max())+3)
        else:
            ax.text(.5, .5, "No attacked scenarios", ha="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Malicious clients (%)")
        ax.set_ylabel(ylabel)
        _style_axis(ax, probability=True)
    _footer(fig, ["Error bars: ±1 sample SD (not a confidence interval); absent for a single seed.",
                  "Attack-window panels include attacked scenarios only; final accuracy includes the clean reference.",
                  _health_note(group, scenarios)])
    return fig, title


def _overhead_figure(data, group, metric, scenarios, runs):
    methods, ratios = data["methods"], group["ratios"]
    memory = metric == "peak_rss_mib"
    suffix, ylabel = {
        "runtime_seconds": ("Recorded runtime", "Runtime (s)"),
        "runtime_without_crypto_seconds": ("Recorded runtime minus crypto spans", "Runtime excluding crypto (s)"),
        "peak_rss_mib": ("Recorded process peak RSS", "Process peak RSS (MiB)"),
    }[metric]
    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    fig.subplots_adjust(left=.10, right=.985, bottom=.27, top=.75)
    title = _title(data, group, suffix)
    fig.suptitle(title, y=.975, fontsize=10.1)
    fig.text(.5, .922, _sample_note(data, group, scenarios), ha="center", fontsize=7.8, color="#555555")
    fig.legend(handles=_handles(methods), loc="upper center", bbox_to_anchor=(.54, .88),
               ncol=min(6, len(methods)), frameon=False, handlelength=2, columnspacing=1, fontsize=7.8)
    centers = np.arange(len(ratios))
    width = .75 / len(methods)
    offsets = np.linspace(-width*.2, width*.2, len(data["seeds"])) if len(data["seeds"]) > 1 else [0]
    seed_offsets = dict(zip(data["seeds"], offsets))
    for j, method in enumerate(methods):
        x = centers + (j-(len(methods)-1)/2)*width
        rows = [scenarios[(group["slug"], method, float(r))] for r in ratios]
        means = np.array([row[metric + "_mean"] for row in rows])
        ax.bar(x, means, width=width*.92, color=COLORS[method], edgecolor="white", linewidth=.3)
        indices = [k for k, row in enumerate(rows) if row["n"] > 1]
        if indices:
            ax.errorbar(x[indices], means[indices], yerr=[rows[k][metric + "_sd"] for k in indices],
                        fmt="none", ecolor="#222222", elinewidth=.6, capsize=1.2, capthick=.6)
        for k, ratio in enumerate(ratios):
            observations = runs[(group["slug"], method, float(ratio))]
            for seed, row in observations.items():
                ax.scatter(x[k]+seed_offsets[seed], row[metric], s=5,
                           color="#222222", linewidths=0, alpha=.6, zorder=4)
    ax.set_xticks(centers, [f"{ratio*100:g}%" for ratio in ratios])
    ax.set_xlabel("Malicious client ratio")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="plain")
    _style_axis(ax)
    note = "Bars: seed mean; dots: individual seeds; error bars: ±1 sample SD when n >= 2."
    if memory:
        definition = "Process lifetime peak RSS, not GPU memory or algorithm-only allocation."
    elif metric == "runtime_without_crypto_seconds":
        definition = "Checkpoint I/O excluded; crypto spans subtracted, not a rerun with cryptography disabled."
    else:
        definition = "Recorded task wall time excluding checkpoint I/O; concurrent work can affect timing."
    _footer(fig, [note, definition, "Observational logs, not a controlled benchmark.",
                  _health_note(group, scenarios)], x=.10, width=112)
    return fig, title


def render_figures(data: dict, output: Path, *, include_pdf=True, png_dpi=300) -> list[dict]:
    """Write six SVG/PNG figures per group and optionally one combined vector PDF.

    Returned ``svg`` and ``png`` paths are relative to ``output``. ``pdf_page`` is
    one-based, or None if PDF export is disabled. Exceptions are deliberately not
    suppressed, so callers cannot accidentally report incomplete exports as done.
    """
    if not isinstance(png_dpi, (int, float)) or not math.isfinite(png_dpi) or png_dpi <= 0:
        raise ValueError("png_dpi must be positive and finite")
    scenarios, curves, runs = _prepare(data)
    output = Path(output)
    for subdir in ("svg", "png"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    settings = {
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white", "axes.unicode_minus": False,
    }
    pdf_context = PdfPages(output / "mean_figures.pdf", metadata={
        "Title": f"{data['dataset_label']} experiment seed reports",
        "Subject": "Seed means and sample standard deviations; original observations retained",
    }) if include_pdf else nullcontext(None)
    entries = []
    with plt.rc_context(settings), pdf_context as pdf:
        for group in data["groups"]:
            specs = [
                ("accuracy_comparison", _curve_figure, (data, group, "accuracy", scenarios, curves)),
                ("asr_comparison", _curve_figure, (data, group, "asr", scenarios, curves)),
                ("final_and_attack_window", _summary_figure, (data, group, scenarios)),
                ("runtime_without_crypto", _overhead_figure, (data, group, "runtime_without_crypto_seconds", scenarios, runs)),
                ("runtime_overhead", _overhead_figure, (data, group, "runtime_seconds", scenarios, runs)),
                ("memory_overhead", _overhead_figure, (data, group, "peak_rss_mib", scenarios, runs)),
            ]
            for suffix, draw, args in specs:
                fig, title = draw(*args)
                try:
                    name = f"{group['slug']}_{suffix}"
                    svg_path, png_path = f"svg/{name}.svg", f"png/{name}.png"
                    fig.savefig(output / svg_path)
                    fig.savefig(output / png_path, dpi=png_dpi)
                    if pdf is not None:
                        pdf.savefig(fig)
                    entries.append({"name": name, "title": title, "svg": svg_path,
                                    "png": png_path, "pdf_page": len(entries)+1 if pdf is not None else None})
                finally:
                    plt.close(fig)
    return entries
