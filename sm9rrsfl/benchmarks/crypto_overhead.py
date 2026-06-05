"""Micro-benchmark SM9-RRS signing and verification overhead."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from html import escape
import json
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

import numpy as np

from sm9rrsfl.crypto import RRSPacket, SM9RRSContext, digest_update, sm3_hex_text


DEFAULT_CLIENT_COUNTS = (20, 50, 100)
DEFAULT_OUTPUT_DIR = Path("outputs/crypto_overhead")


@dataclass(frozen=True)
class CryptoOverheadConfig:
    client_counts: tuple[int, ...] = DEFAULT_CLIENT_COUNTS
    iterations: int = 20
    warmup: int = 3
    update_size: int = 4096
    crypto_mode: str = "sm9"
    accumulator_mode: str = "dynamic"
    ring_size: int = 5
    strict_ring_verify: bool = False
    precompute_sign_cache: bool = True
    task_id: str = "crypto-overhead"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    visualizations: bool = True
    seed: int = 42


@dataclass(frozen=True)
class ClientCountSummary:
    crypto_mode: str
    accumulator_mode: str
    num_clients: int
    iterations: int
    warmup: int
    update_size: int
    setup_ms: float
    sign_cache_ms: float
    sign_mean_ms: float
    sign_median_ms: float
    sign_p95_ms: float
    sign_std_ms: float
    verify_mean_ms: float
    verify_median_ms: float
    verify_p95_ms: float
    verify_std_ms: float
    verify_successes: int


@dataclass(frozen=True)
class OperationSample:
    crypto_mode: str
    accumulator_mode: str
    num_clients: int
    iteration: int
    signer: str
    sign_ms: float
    verify_ms: float
    verify_ok: bool


@dataclass(frozen=True)
class BenchmarkResult:
    summaries: list[ClientCountSummary]
    samples: list[OperationSample]


def main() -> None:
    config = parse_args()
    result = run_benchmarks(config)
    visualization_path = write_outputs(config, result)
    print_summary(result.summaries)
    print(f"wrote {config.output_dir / 'summary.csv'}")
    print(f"wrote {config.output_dir / 'samples.csv'}")
    print(f"wrote {config.output_dir / 'summary.json'}")
    if visualization_path is not None:
        print(f"wrote {visualization_path}")


def parse_args() -> CryptoOverheadConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-counts", nargs="+", type=int, default=list(DEFAULT_CLIENT_COUNTS))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--update-size", type=int, default=4096)
    parser.add_argument("--crypto-mode", choices=["sm9", "simulated"], default="sm9")
    parser.add_argument("--accumulator-mode", choices=["dynamic", "none"], default="dynamic")
    parser.add_argument("--ring-size", type=int, default=5)
    parser.add_argument("--strict-ring-verify", action="store_true")
    parser.add_argument("--no-precompute-sign-cache", action="store_true")
    parser.add_argument("--task-id", default="crypto-overhead")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return CryptoOverheadConfig(
        client_counts=tuple(args.client_counts),
        iterations=args.iterations,
        warmup=args.warmup,
        update_size=args.update_size,
        crypto_mode=args.crypto_mode,
        accumulator_mode=args.accumulator_mode,
        ring_size=args.ring_size,
        strict_ring_verify=args.strict_ring_verify,
        precompute_sign_cache=not args.no_precompute_sign_cache,
        task_id=args.task_id,
        output_dir=args.output_dir,
        visualizations=not args.no_visualizations,
        seed=args.seed,
    )


def run_benchmarks(config: CryptoOverheadConfig) -> BenchmarkResult:
    _validate_config(config)
    summaries = []
    samples = []
    for num_clients in config.client_counts:
        summary, client_samples = benchmark_client_count(config, num_clients)
        summaries.append(summary)
        samples.extend(client_samples)
    return BenchmarkResult(summaries=summaries, samples=samples)


def benchmark_client_count(
    config: CryptoOverheadConfig,
    num_clients: int,
) -> tuple[ClientCountSummary, list[OperationSample]]:
    client_ids = tuple(f"client-{idx}" for idx in range(num_clients))
    setup_started = perf_counter()
    context = SM9RRSContext(
        list(client_ids),
        ring_size=config.ring_size,
        crypto_mode=config.crypto_mode,
        accumulator_mode=config.accumulator_mode,
        strict_ring_verify=config.strict_ring_verify,
        seed=config.seed,
    )
    setup_ms = _elapsed_ms(setup_started)

    rng = np.random.default_rng(config.seed + num_clients)
    update = rng.normal(size=config.update_size).astype(np.float32)
    cache_ms = 0.0
    if config.precompute_sign_cache:
        cache_ms = precompute_sign_cache(context, client_ids, update, config)

    for index in range(config.warmup):
        identity = client_ids[index % num_clients]
        packet = build_unsigned_packet(context, identity, update, config, round_id=index + 1)
        signature = context._sign(identity, context._message_for(packet))
        signed_packet = replace(packet, signature=signature)
        if not context.verify_packet(signed_packet, update):
            raise RuntimeError(f"warmup verification failed for {identity}")

    sign_ms_values: list[float] = []
    verify_ms_values: list[float] = []
    samples: list[OperationSample] = []
    for index in range(config.iterations):
        identity = client_ids[index % num_clients]
        round_id = config.warmup + index + 1
        packet = build_unsigned_packet(context, identity, update, config, round_id=round_id)
        message = context._message_for(packet)

        sign_started = perf_counter()
        signature = context._sign(identity, message)
        sign_ms = _elapsed_ms(sign_started)

        signed_packet = replace(packet, signature=signature)
        verify_started = perf_counter()
        verify_ok = context.verify_packet(signed_packet, update)
        verify_ms = _elapsed_ms(verify_started)
        if not verify_ok:
            raise RuntimeError(f"verification failed for {identity}")

        sign_ms_values.append(sign_ms)
        verify_ms_values.append(verify_ms)
        samples.append(
            OperationSample(
                crypto_mode=config.crypto_mode,
                accumulator_mode=config.accumulator_mode,
                num_clients=num_clients,
                iteration=index + 1,
                signer=identity,
                sign_ms=sign_ms,
                verify_ms=verify_ms,
                verify_ok=verify_ok,
            )
        )

    summary = ClientCountSummary(
        crypto_mode=config.crypto_mode,
        accumulator_mode=config.accumulator_mode,
        num_clients=num_clients,
        iterations=config.iterations,
        warmup=config.warmup,
        update_size=config.update_size,
        setup_ms=setup_ms,
        sign_cache_ms=cache_ms,
        sign_mean_ms=mean(sign_ms_values),
        sign_median_ms=median(sign_ms_values),
        sign_p95_ms=_percentile(sign_ms_values, 0.95),
        sign_std_ms=pstdev(sign_ms_values) if len(sign_ms_values) > 1 else 0.0,
        verify_mean_ms=mean(verify_ms_values),
        verify_median_ms=median(verify_ms_values),
        verify_p95_ms=_percentile(verify_ms_values, 0.95),
        verify_std_ms=pstdev(verify_ms_values) if len(verify_ms_values) > 1 else 0.0,
        verify_successes=sum(1 for sample in samples if sample.verify_ok),
    )
    return summary, samples


def precompute_sign_cache(
    context: SM9RRSContext,
    client_ids: tuple[str, ...],
    update: np.ndarray,
    config: CryptoOverheadConfig,
) -> float:
    started = perf_counter()
    for index, identity in enumerate(client_ids):
        packet = build_unsigned_packet(context, identity, update, config, round_id=index + 1)
        context._sign(identity, context._message_for(packet))
    return _elapsed_ms(started)


def build_unsigned_packet(
    context: SM9RRSContext,
    identity: str,
    update: np.ndarray,
    config: CryptoOverheadConfig,
    *,
    round_id: int,
) -> RRSPacket:
    ring = tuple() if context.accumulator_mode == "dynamic" else context._sample_ring(identity)
    update_digest = digest_update(update)
    ring_accumulator, current_ring_digest, current_ring_size = context._ring_commitment(ring)
    link_tag = sm3_hex_text(f"link:{config.task_id}:{identity}")[:32]
    event_tag = sm3_hex_text(f"event:{config.task_id}:{round_id}:{update_digest}")[:32]
    trapdoor_plain = json.dumps(
        {"id": identity, "event": event_tag, "tag": link_tag},
        sort_keys=True,
        separators=(",", ":"),
    )
    trapdoor = context._encrypt_trapdoor(trapdoor_plain)
    signer_hint = identity if context.crypto_mode == "simulated" or context.accumulator_mode == "none" else ""
    return RRSPacket(
        task_id=config.task_id,
        round_id=round_id,
        ring=ring,
        ring_accumulator=ring_accumulator,
        ring_digest=current_ring_digest,
        ring_size=current_ring_size,
        link_tag=link_tag,
        event_tag=event_tag,
        update_digest=update_digest,
        trapdoor=trapdoor,
        signature=None,
        crypto_mode=context.crypto_mode,
        accumulator_mode=context.accumulator_mode,
        _signer_identity_hint=signer_hint,
    )


def write_outputs(config: CryptoOverheadConfig, result: BenchmarkResult) -> Path | None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.output_dir / "summary.csv", [asdict(row) for row in result.summaries])
    write_csv(config.output_dir / "samples.csv", [asdict(row) for row in result.samples])
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {**asdict(config), "output_dir": str(config.output_dir)},
                "summary": [asdict(row) for row in result.summaries],
                "samples": [asdict(row) for row in result.samples],
            },
            handle,
            indent=2,
        )
    if config.visualizations:
        return generate_visualizations(result.summaries, config.output_dir)
    return None


def generate_visualizations(summaries: list[ClientCountSummary], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str]] = []
    mean_path = plot_dir / "mean_operation_overhead.svg"
    _write_text(
        mean_path,
        _line_chart(
            title="不同客户端数量下的单次签名与验签均值",
            series={
                "单次签名均值": [(row.num_clients, row.sign_mean_ms) for row in summaries],
                "单次验签均值": [(row.num_clients, row.verify_mean_ms) for row in summaries],
            },
            x_label="客户端数量",
            y_label="耗时（ms）",
        ),
    )
    generated.append(("单次签名/验签均值对比", mean_path.relative_to(out).as_posix()))

    sign_path = plot_dir / "signature_overhead.svg"
    _write_text(
        sign_path,
        _line_chart(
            title="不同客户端数量下的签名开销",
            series={
                "签名均值": [(row.num_clients, row.sign_mean_ms) for row in summaries],
                "签名 P95": [(row.num_clients, row.sign_p95_ms) for row in summaries],
            },
            x_label="客户端数量",
            y_label="耗时（ms）",
        ),
    )
    generated.append(("签名开销", sign_path.relative_to(out).as_posix()))

    verify_path = plot_dir / "verification_overhead.svg"
    _write_text(
        verify_path,
        _line_chart(
            title="不同客户端数量下的验签开销",
            series={
                "验签均值": [(row.num_clients, row.verify_mean_ms) for row in summaries],
                "验签 P95": [(row.num_clients, row.verify_p95_ms) for row in summaries],
            },
            x_label="客户端数量",
            y_label="耗时（ms）",
        ),
    )
    generated.append(("验签开销", verify_path.relative_to(out).as_posix()))

    setup_path = plot_dir / "setup_and_cache_overhead.svg"
    _write_text(
        setup_path,
        _grouped_bar_chart(
            title="初始化与签名缓存预热开销",
            values={
                str(row.num_clients): {
                    "上下文初始化": row.setup_ms,
                    "签名缓存预热": row.sign_cache_ms,
                }
                for row in summaries
            },
            x_label="客户端数量",
            y_label="耗时（ms）",
        ),
    )
    generated.append(("初始化与缓存预热开销", setup_path.relative_to(out).as_posix()))

    dashboard = out / "visualizations.html"
    _write_text(dashboard, _dashboard_html(summaries, generated))
    return dashboard


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries: list[ClientCountSummary]) -> None:
    print(
        "clients setup_ms sign_cache_ms sign_mean_ms sign_p95_ms "
        "verify_mean_ms verify_p95_ms"
    )
    for row in summaries:
        print(
            f"{row.num_clients:7d} "
            f"{row.setup_ms:8.2f} "
            f"{row.sign_cache_ms:13.2f} "
            f"{row.sign_mean_ms:12.2f} "
            f"{row.sign_p95_ms:11.2f} "
            f"{row.verify_mean_ms:14.2f} "
            f"{row.verify_p95_ms:13.2f}"
        )


def _line_chart(
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    x_label: str,
    y_label: str,
) -> str:
    width, height = 900, 460
    left, right, top, bottom = 82, 170, 56, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [point for values in series.values() for point in values]
    min_x = min((point[0] for point in all_points), default=0.0)
    max_x = max((point[0] for point in all_points), default=1.0)
    if min_x == max_x:
        min_x = 0.0
    max_y = max((point[1] for point in all_points), default=1.0)
    y_max = max(max_y * 1.15, 1e-9)

    def sx(x_value: float) -> float:
        return left + ((x_value - min_x) / (max_x - min_x)) * plot_w

    def sy(y_value: float) -> float:
        return top + (1.0 - y_value / y_max) * plot_h

    parts = [_svg_open(width, height), _chart_title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label, width, height))
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>')
    for tick in range(0, 6):
        value = min_x + (max_x - min_x) * tick / 5
        x = sx(value)
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#475569">{value:.0f}</text>')

    for index, (label, points) in enumerate(series.items()):
        color = _series_color(index)
        point_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        dash_attr = ' stroke-dasharray="6 4"' if index == 1 else ""
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.8"{dash_attr}/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
            parts.append(
                f'<text x="{sx(x):.1f}" y="{sy(y) - 8:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#334155">{y:.2f}</text>'
            )
        legend_y = top + 24 + index * 24
        parts.append(f'<rect x="{width - right + 22}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 42}" y="{legend_y + 2}" font-size="13" fill="#0f172a">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _grouped_bar_chart(
    *,
    title: str,
    values: dict[str, dict[str, float]],
    x_label: str,
    y_label: str,
) -> str:
    width, height = 900, 460
    left, right, top, bottom = 82, 170, 56, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    groups = list(values.items())
    series_names = sorted({series for group in values.values() for series in group})
    max_value = max((value for group in values.values() for value in group.values()), default=1.0)
    y_max = max(max_value * 1.15, 1e-9)
    group_w = plot_w / max(1, len(groups))
    bar_w = min(42.0, group_w / max(1, len(series_names) + 1))

    def sy(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    parts = [_svg_open(width, height), _chart_title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label, width, height))
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>')

    for group_index, (group_label, group_values) in enumerate(groups):
        center = left + group_index * group_w + group_w / 2
        start = center - (len(series_names) * bar_w) / 2
        for series_index, series_name in enumerate(series_names):
            value = group_values.get(series_name, 0.0)
            x = start + series_index * bar_w
            y = sy(value)
            color = _series_color(series_index)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 5:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="{color}"/>'
            )
            if value > 0:
                parts.append(
                    f'<text x="{x + (bar_w - 5) / 2:.1f}" y="{y - 5:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="#334155">{value:.1f}</text>'
                )
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#475569">{escape(group_label)}</text>')

    for index, series_name in enumerate(series_names):
        color = _series_color(index)
        legend_y = top + 24 + index * 24
        parts.append(f'<rect x="{width - right + 22}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 42}" y="{legend_y + 2}" font-size="13" fill="#0f172a">{escape(series_name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _dashboard_html(
    summaries: list[ClientCountSummary],
    generated: list[tuple[str, str]],
) -> str:
    if summaries:
        first = summaries[0]
        mode = f"{first.crypto_mode} / {first.accumulator_mode}"
        detail = (
            f"迭代次数 {first.iterations}，预热 {first.warmup}，"
            f"更新向量长度 {first.update_size}。"
        )
    else:
        mode = ""
        detail = ""
    figures = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(path)}" alt="{escape(title)}"></section>'
        for title, path in generated
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SM9-RRS 签名与验签开销可视化</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    p {{ color: #475569; }}
    section {{ margin-top: 24px; padding: 18px; background: white; border: 1px solid #e2e8f0; border-radius: 8px; }}
    img {{ display: block; max-width: 100%; height: auto; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>SM9-RRS 签名与验签开销可视化</h1>
  <p>模式：<code>{escape(mode)}</code>。{escape(detail)}数据来自同目录下的 <code>summary.csv</code> 与 <code>samples.csv</code>。</p>
  {figures}
</main>
</body>
</html>
"""


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
    )


def _chart_title(title: str, width: int) -> str:
    return f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">{escape(title)}</text>'


def _axes(
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    x_label: str,
    y_label: str,
    width: int,
    height: int,
) -> str:
    return "\n".join(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155" stroke-width="1.4"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155" stroke-width="1.4"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#334155">{escape(x_label)}</text>',
            f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" fill="#334155">{escape(y_label)}</text>',
        ]
    )


def _series_color(index: int) -> str:
    palette = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c")
    return palette[index % len(palette)]


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _validate_config(config: CryptoOverheadConfig) -> None:
    if not config.client_counts:
        raise ValueError("client_counts must not be empty")
    if any(count < 1 for count in config.client_counts):
        raise ValueError("client_counts must contain only positive integers")
    if config.iterations < 1:
        raise ValueError("iterations must be positive")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if config.update_size < 1:
        raise ValueError("update_size must be positive")
    if config.ring_size < 1:
        raise ValueError("ring_size must be positive")


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


if __name__ == "__main__":
    main()
