#!/usr/bin/env python3
"""Generate two-metric quantitative analysis for issue34 Prediction_b.

This script compares four models on two non-bond geometry metrics:
1) Hausdorff distance
2) Matching distance

Outputs:
- per_sample_metrics.csv
- summary_metrics.csv
- paired_tests.csv
- seq_stats_comparison.csv
- plot_A_distributions.png
- plot_B_paired_differences.png
- plot_C_ecdf.png
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.stats import wilcoxon


plt.rcParams["font.size"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = shutil.which("latex") is not None
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


DIST_THRESHOLD = 0.35
BOOTSTRAP_ROUNDS = 5000
SEED = 7


@dataclass(frozen=True)
class ModelSpec:
    folder: str
    notation: str


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("Ref_Pure_C9_issue34", "F_U"),
    ModelSpec("Ref_issue34", "F_Vbar"),
    ModelSpec("PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C7_issue34", "F_Vtilde"),
    ModelSpec("PPAFM2Exp_CoAll_L20_L1_Elatest_C1_issue34", "F_Vdagger"),
)

LATEX_MODEL_LABELS: Dict[str, str] = {
    "F_U": r"$F_{\mathcal{U}}$",
    "F_Vbar": r"$F_{\overline{\mathcal{V}}}$",
    "F_Vtilde": r"$F_{\widetilde{\mathcal{V}}}$",
    "F_Vdagger": r"$F_{\mathcal{V}^{\dagger}}$",
}

# Format requested in plan: First - Second
COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("F_Vtilde", "F_U"),
    ("F_Vdagger", "F_U"),
    ("F_Vtilde", "F_Vbar"),
    ("F_Vdagger", "F_Vbar"),
)

METRICS = ("hausdorff", "matching_distance")

SIM_COLOR = "#ed9d2c"
EXP_COLOR = "#de461c"
DFT_COLOR = "#2ca3cf"
BG07_COLOR = "#479FB1"
BV17_COLOR = "#6E7CBC"
PURE_SIM_GRAY = "#7a7a7a"

MODEL_COLORS = {
    "F_U": PURE_SIM_GRAY,
    "F_Vbar": SIM_COLOR,
    "F_Vtilde": EXP_COLOR,
    "F_Vdagger": DFT_COLOR,
}

MODEL_LINESTYLES = {
    "F_U": "dashdot",
    "F_Vbar": "dotted",
    "F_Vtilde": "dashed",
    "F_Vdagger": "solid",
}

HIGHLIGHT_SAMPLE_IDS = (20, 21, 23, 17)
HIGHLIGHT_MARKERS = {
    20: ("^", "Prediction 1"),
    21: ("o", "Prediction 2"),
    23: ("s", "Prediction 3"),
    17: ("*", "Prediction 4"),
}
HIGHLIGHT_RIGHT_SHIFT_SPLIT = 0.06
HIGHLIGHT_RIGHT_SHIFT_SINGLE = 0.36

H_COLOR = BG07_COLOR
O_COLOR = EXP_COLOR


def blend_with_white(color: str, blend: float) -> Tuple[float, float, float]:
    """Return a lighter version of color by blending toward white.

    blend=0 keeps original color, blend=1 returns white.
    """
    r, g, b = to_rgb(color)
    return (
        r + (1.0 - r) * blend,
        g + (1.0 - g) * blend,
        b + (1.0 - b) * blend,
    )


def parse_xyz_positions(path: Path) -> np.ndarray:
    """Parse xyz file and return Nx3 coordinate array."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty xyz file: {path}")

    # First non-empty line is atom count. Data starts after header.
    n_atoms = int(lines[0].strip())
    rows = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        cols = s.split()
        if len(cols) < 4:
            continue
        rows.append([float(cols[1]), float(cols[2]), float(cols[3])])

    arr = np.asarray(rows, dtype=float)
    if arr.shape[0] != n_atoms:
        raise ValueError(
            f"Atom count mismatch in {path}: header={n_atoms}, parsed={arr.shape[0]}"
        )
    return arr


def parse_xyz_with_atomic_numbers(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse xyz file and return (atomic_numbers, Nx3 coordinates)."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty xyz file: {path}")

    n_atoms = int(lines[0].strip())
    atomic_numbers = []
    rows = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        cols = s.split()
        if len(cols) < 4:
            continue
        atomic_numbers.append(int(float(cols[0])))
        rows.append([float(cols[1]), float(cols[2]), float(cols[3])])

    z = np.asarray(atomic_numbers, dtype=int)
    xyz = np.asarray(rows, dtype=float)
    if xyz.shape[0] != n_atoms:
        raise ValueError(
            f"Atom count mismatch in {path}: header={n_atoms}, parsed={xyz.shape[0]}"
        )
    return z, xyz


def filter_element_positions(z: np.ndarray, xyz: np.ndarray, atomic_number: int) -> np.ndarray:
    """Return coordinates for one atomic number (1=H, 8=O)."""
    return xyz[z == atomic_number]


def hausdorff_distance(pred: np.ndarray, ref: np.ndarray) -> float:
    if len(pred) == 0 or len(ref) == 0:
        return float("nan")
    d1 = directed_hausdorff(pred, ref)[0]
    d2 = directed_hausdorff(ref, pred)[0]
    return float(max(d1, d2))


def matching_distance(pred: np.ndarray, ref: np.ndarray, threshold: float = DIST_THRESHOLD) -> float:
    """Match definition mirrors GraphStats logic for distance collection.

    For each reference atom, pick closest predicted atom within threshold and
    average those matched distances.
    """
    if len(pred) == 0 or len(ref) == 0:
        return float("nan")

    dist_mat = cdist(ref, pred, metric="euclidean")
    matched_distances = []
    for i in range(dist_mat.shape[0]):
        row = dist_mat[i]
        candidate_idx = np.where(row < threshold)[0]
        if len(candidate_idx) == 0:
            continue
        if len(candidate_idx) == 1:
            matched_distances.append(float(row[candidate_idx[0]]))
        else:
            local_best = candidate_idx[np.argmin(row[candidate_idx])]
            matched_distances.append(float(row[local_best]))

    if not matched_distances:
        return float("nan")
    return float(np.mean(matched_distances))


def find_sample_ids(predictions_dir: Path) -> List[int]:
    ids = []
    for path in predictions_dir.glob("*_graph_pred.xyz"):
        stem = path.name.replace("_graph_pred.xyz", "")
        ids.append(int(stem))
    return sorted(ids)


def read_seq_stats_mean(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            try:
                val = float(v)
            except ValueError:
                continue
            if k == "Mean Hausdorff distance":
                out["hausdorff"] = val
            elif k == "Mean matching distance":
                out["matching_distance"] = val
    return out


def bootstrap_ci(values: np.ndarray, func, rounds: int = BOOTSTRAP_ROUNDS) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stats = np.empty(rounds, dtype=float)
    for i in range(rounds):
        idx = rng.integers(0, n, size=n)
        stats[i] = func(values[idx])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def holm_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    clean = np.sort(clean)
    y = np.arange(1, len(clean) + 1, dtype=float) / max(1, len(clean))
    return clean, y


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_figure_multiformat(fig: plt.Figure, out_base: Path, dpi: int = 200) -> None:
    """Save one figure to PNG, SVG, and PDF with the same basename."""
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".svg"))
    fig.savefig(out_base.with_suffix(".pdf"))


def draw_split_violin(
    ax: plt.Axes,
    left_vals: np.ndarray,
    right_vals: np.ndarray,
    pos: float,
    width: float = 0.8,
    left_color: str = "tab:blue",
    right_color: str = "tab:orange",
) -> None:
    """Draw one split violin at x=pos; left half from left_vals, right half from right_vals."""
    if len(left_vals) > 0:
        vl = ax.violinplot([left_vals], positions=[pos], widths=width, showmeans=False, showextrema=False, showmedians=False)
        body = vl["bodies"][0]
        body.set_facecolor(left_color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], pos)

    if len(right_vals) > 0:
        vr = ax.violinplot([right_vals], positions=[pos], widths=width, showmeans=False, showextrema=False, showmedians=False)
        body = vr["bodies"][0]
        body.set_facecolor(right_color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.maximum(verts[:, 0], pos)


def draw_species_violin_scatter(
    ax: plt.Axes,
    h_vals: np.ndarray,
    o_vals: np.ndarray,
    center: float,
    rng: np.random.Generator,
    base_color: str,
    offset: float = 0.18,
    width: float = 0.28,
) -> None:
    """Draw side-by-side H/O violins for one model with scatter and medians."""
    x_h = center - offset
    x_o = center + offset
    color_h = blend_with_white(base_color, 0.50)
    color_o = blend_with_white(base_color, 0.15)

    if len(h_vals) > 0:
        vh = ax.violinplot([h_vals], positions=[x_h], widths=width, showmeans=False, showextrema=False, showmedians=False)
        body_h = vh["bodies"][0]
        body_h.set_facecolor(color_h)
        body_h.set_edgecolor("black")
        body_h.set_alpha(0.35)

        jitter_h = rng.uniform(-0.03, 0.03, size=len(h_vals))
        ax.scatter(np.full(len(h_vals), x_h) + jitter_h, h_vals, s=12, alpha=0.45, color=color_h)
        med_h = np.median(h_vals)
        ax.plot([x_h - 0.06, x_h + 0.06], [med_h, med_h], color="black", lw=2)

    if len(o_vals) > 0:
        vo = ax.violinplot([o_vals], positions=[x_o], widths=width, showmeans=False, showextrema=False, showmedians=False)
        body_o = vo["bodies"][0]
        body_o.set_facecolor(color_o)
        body_o.set_edgecolor("black")
        body_o.set_alpha(0.35)

        jitter_o = rng.uniform(-0.03, 0.03, size=len(o_vals))
        ax.scatter(np.full(len(o_vals), x_o) + jitter_o, o_vals, s=12, alpha=0.45, color=color_o)
        med_o = np.median(o_vals)
        ax.plot([x_o - 0.06, x_o + 0.06], [med_o, med_o], color="black", lw=2)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    issue34_dir = script_dir.parent
    output_dir = script_dir / "outputs"
    ensure_dir(output_dir)

    # Build model directories
    model_dirs: Dict[str, Path] = {}
    for spec in MODEL_SPECS:
        pred_dir = issue34_dir / spec.folder / "Prediction_b" / "predictions"
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing predictions directory: {pred_dir}")
        model_dirs[spec.notation] = pred_dir

    # Shared sample IDs for paired analysis
    sample_sets = {name: set(find_sample_ids(path)) for name, path in model_dirs.items()}
    shared_ids = sorted(set.intersection(*sample_sets.values()))
    if not shared_ids:
        raise RuntimeError("No shared sample IDs found across models")

    per_sample_rows = []
    for notation, pred_dir in model_dirs.items():
        for sid in shared_ids:
            pred_path = pred_dir / f"{sid}_graph_pred.xyz"
            ref_path = pred_dir / f"{sid}_graph_ref.xyz"
            pred_z, pred = parse_xyz_with_atomic_numbers(pred_path)
            ref_z, ref = parse_xyz_with_atomic_numbers(ref_path)

            h = hausdorff_distance(pred, ref)
            m = matching_distance(pred, ref, threshold=DIST_THRESHOLD)
            pred_h = filter_element_positions(pred_z, pred, 1)
            ref_h = filter_element_positions(ref_z, ref, 1)
            pred_o = filter_element_positions(pred_z, pred, 8)
            ref_o = filter_element_positions(ref_z, ref, 8)

            h_h = hausdorff_distance(pred_h, ref_h)
            h_o = hausdorff_distance(pred_o, ref_o)
            m_h = matching_distance(pred_h, ref_h, threshold=DIST_THRESHOLD)
            m_o = matching_distance(pred_o, ref_o, threshold=DIST_THRESHOLD)

            per_sample_rows.append(
                {
                    "model": notation,
                    "model_latex": LATEX_MODEL_LABELS[notation],
                    "sample_id": sid,
                    "hausdorff": h,
                    "matching_distance": m,
                    "hausdorff_H": h_h,
                    "hausdorff_O": h_o,
                    "matching_distance_H": m_h,
                    "matching_distance_O": m_o,
                }
            )

    per_sample_lookup = {
        (str(r["model"]), int(r["sample_id"])): r for r in per_sample_rows
    }

    # Summary metrics
    summary_rows = []
    for notation in model_dirs.keys():
        rows = [r for r in per_sample_rows if r["model"] == notation]
        for metric in METRICS:
            vals = np.asarray([r[metric] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            summary_rows.append(
                {
                    "model": notation,
                    "model_latex": LATEX_MODEL_LABELS[notation],
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "q1": float(q1),
                    "q3": float(q3),
                }
            )

    # Compare subset means to full-seq_stats means
    seq_rows = []
    folder_by_notation = {spec.notation: spec.folder for spec in MODEL_SPECS}
    for notation in model_dirs.keys():
        seq_file = issue34_dir / folder_by_notation[notation] / "Prediction_b" / "stats" / "seq_stats.csv"
        full_means = read_seq_stats_mean(seq_file)
        for metric in METRICS:
            subset_mean = next(
                r["mean"] for r in summary_rows if r["model"] == notation and r["metric"] == metric
            )
            seq_rows.append(
                {
                    "model": notation,
                    "model_latex": LATEX_MODEL_LABELS[notation],
                    "metric": metric,
                    "subset_mean_from_predictions_folder": subset_mean,
                    "full_mean_from_seq_stats": full_means.get(metric, float("nan")),
                }
            )

    # Statistical tests for paired differences
    test_rows = []
    raw_pvals = []
    pval_index = []

    # Index values by model + sample for fast alignment
    by_model_metric: Dict[Tuple[str, str], Dict[int, float]] = {}
    for notation in model_dirs.keys():
        model_rows = [r for r in per_sample_rows if r["model"] == notation]
        for metric in METRICS:
            by_model_metric[(notation, metric)] = {
                int(r["sample_id"]): float(r[metric])
                for r in model_rows
                if math.isfinite(float(r[metric]))
            }

    for first, second in COMPARISONS:
        for metric in METRICS:
            m1 = by_model_metric[(first, metric)]
            m2 = by_model_metric[(second, metric)]
            ids = sorted(set(m1.keys()) & set(m2.keys()))
            v1 = np.asarray([m1[i] for i in ids], dtype=float)
            v2 = np.asarray([m2[i] for i in ids], dtype=float)
            diffs = v1 - v2  # First - Second (negative means first is better, lower-is-better metrics)

            if len(diffs) == 0:
                pval = float("nan")
            elif np.allclose(diffs, 0.0):
                pval = 1.0
            else:
                pval = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)

            mean_diff = float(np.mean(diffs)) if len(diffs) else float("nan")
            median_diff = float(np.median(diffs)) if len(diffs) else float("nan")
            ci_lo, ci_hi = bootstrap_ci(diffs, np.median) if len(diffs) else (float("nan"), float("nan"))

            row = {
                "comparison": f"{first} - {second}",
                "comparison_latex": f"{LATEX_MODEL_LABELS[first]} - {LATEX_MODEL_LABELS[second]}",
                "metric": metric,
                "n_pairs": len(diffs),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "median_diff_ci95_low": ci_lo,
                "median_diff_ci95_high": ci_hi,
                "p_value_raw": pval,
                "p_value_holm": float("nan"),
            }
            test_rows.append(row)
            raw_pvals.append(pval)
            pval_index.append(len(test_rows) - 1)

    valid_mask = [math.isfinite(p) for p in raw_pvals]
    if any(valid_mask):
        valid_pvals = [p for p in raw_pvals if math.isfinite(p)]
        adjusted = holm_correction(valid_pvals)
        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                test_rows[pval_index[i]]["p_value_holm"] = adjusted[j]
                j += 1

    # Write csv outputs
    write_csv(
        output_dir / "per_sample_metrics.csv",
        per_sample_rows,
        [
            "model",
            "model_latex",
            "sample_id",
            "hausdorff",
            "matching_distance",
            "hausdorff_H",
            "hausdorff_O",
            "matching_distance_H",
            "matching_distance_O",
        ],
    )
    write_csv(
        output_dir / "summary_metrics.csv",
        summary_rows,
        ["model", "model_latex", "metric", "n", "mean", "std", "median", "q1", "q3"],
    )
    write_csv(
        output_dir / "paired_tests.csv",
        test_rows,
        [
            "comparison",
            "comparison_latex",
            "metric",
            "n_pairs",
            "mean_diff",
            "median_diff",
            "median_diff_ci95_low",
            "median_diff_ci95_high",
            "p_value_raw",
            "p_value_holm",
        ],
    )
    write_csv(
        output_dir / "seq_stats_comparison.csv",
        seq_rows,
        [
            "model",
            "model_latex",
            "metric",
            "subset_mean_from_predictions_folder",
            "full_mean_from_seq_stats",
        ],
    )

    # Plot A: distributions
    model_order = [s.notation for s in MODEL_SPECS]
    model_order_latex = [LATEX_MODEL_LABELS[m] for m in model_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    rng_plot_a = np.random.default_rng(SEED)
    pred_handles = [
        Line2D(
            [0],
            [0],
            marker=HIGHLIGHT_MARKERS[sid][0],
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label=HIGHLIGHT_MARKERS[sid][1],
        )
        for sid in HIGHLIGHT_SAMPLE_IDS
    ]
    for ax, metric, ylabel in zip(
        axes,
        METRICS,
        ("Hausdorff distance (H/O)", "Matching distance (H/O)"),
    ):
        x_positions = np.arange(1, len(model_order) + 1)
        for i, m in enumerate(model_order, start=1):
            vals_h = np.asarray(
                [r[f"{metric}_H"] for r in per_sample_rows if r["model"] == m],
                dtype=float,
            )
            vals_o = np.asarray(
                [r[f"{metric}_O"] for r in per_sample_rows if r["model"] == m],
                dtype=float,
            )
            vals_h = vals_h[np.isfinite(vals_h)]
            vals_o = vals_o[np.isfinite(vals_o)]
            draw_species_violin_scatter(
                ax,
                vals_h,
                vals_o,
                center=i,
                rng=rng_plot_a,
                base_color=MODEL_COLORS[m],
            )

            for sid in HIGHLIGHT_SAMPLE_IDS:
                row = per_sample_lookup.get((m, sid))
                if row is None:
                    continue
                y_h = float(row[f"{metric}_H"])
                y_o = float(row[f"{metric}_O"])
                marker_style, _ = HIGHLIGHT_MARKERS.get(sid, ("o", "Prediction"))
                if math.isfinite(y_h):
                    ax.scatter(
                        i - 0.18 + HIGHLIGHT_RIGHT_SHIFT_SPLIT,
                        y_h,
                        s=75,
                        marker=marker_style,
                        facecolors="none",
                        edgecolors=MODEL_COLORS[m],
                        linewidths=1.0,
                        zorder=8,
                    )
                if math.isfinite(y_o):
                    ax.scatter(
                        i + 0.18 + HIGHLIGHT_RIGHT_SHIFT_SPLIT,
                        y_o,
                        s=75,
                        marker=marker_style,
                        facecolors="none",
                        edgecolors=MODEL_COLORS[m],
                        linewidths=1.0,
                        zorder=8,
                    )

        ax.set_xlim(0.5, len(model_order) + 0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_order_latex)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(
            handles=[
                Patch(facecolor=blend_with_white("#666666", 0.50), edgecolor="black", alpha=0.35, label="H"),
                Patch(facecolor=blend_with_white("#666666", 0.15), edgecolor="black", alpha=0.35, label="O"),
                Patch(facecolor="none", edgecolor="black", label="black line: median"),
            ],
            frameon=False,
            loc="lower right",
        )
        pred_legend = ax.legend(handles=pred_handles, frameon=False, loc="upper right", fontsize=9)
        for txt in pred_legend.get_texts():
            txt.set_color("black")
        ax.add_artist(pred_legend)

    fig.suptitle("Plot A: Distribution Comparison on Prediction_b (shared samples)")
    save_figure_multiformat(fig, output_dir / "plot_A_distributions", dpi=200)
    plt.close(fig)

    # Plot B: paired differences
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    comp_labels = [f"{LATEX_MODEL_LABELS[a]}-{LATEX_MODEL_LABELS[b]}" for a, b in COMPARISONS]
    x = np.arange(1, len(COMPARISONS) + 1)

    rng = np.random.default_rng(SEED)
    comp_colors = [DFT_COLOR, BG07_COLOR, DFT_COLOR, BG07_COLOR]
    for ax, metric, title in zip(
        axes,
        METRICS,
        ("Hausdorff", "Matching"),
    ):
        all_diffs = []
        for first, second in COMPARISONS:
            m1 = by_model_metric[(first, metric)]
            m2 = by_model_metric[(second, metric)]
            ids = sorted(set(m1.keys()) & set(m2.keys()))
            diffs = np.asarray([m1[i] - m2[i] for i in ids], dtype=float)
            all_diffs.append(diffs)

        # jittered points + median line
        for i, diffs in enumerate(all_diffs):
            jitter = rng.uniform(-0.08, 0.08, size=len(diffs))
            ax.scatter(np.full(len(diffs), x[i]) + jitter, diffs, s=12, alpha=0.4, color=comp_colors[i])
            if len(diffs):
                med = np.median(diffs)
                ci_lo, ci_hi = bootstrap_ci(diffs, np.median)
                ax.plot([x[i] - 0.18, x[i] + 0.18], [med, med], color="black", lw=2)
                ax.vlines(x[i], ci_lo, ci_hi, color="black", lw=1.8)

        ax.axhline(0.0, color="red", linestyle="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_labels, rotation=0)
        ax.set_title(f"{title} Difference (First - Second)")
        ax.set_ylabel("Difference in distance")

    fig.suptitle("Plot B: Paired Differences (negative favors first model)")
    save_figure_multiformat(fig, output_dir / "plot_B_paired_differences", dpi=200)
    plt.close(fig)

    # Plot C: empirical CDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        METRICS,
        ("Hausdorff Distance", "Matching Distance"),
    ):
        for m in model_order:
            vals = np.asarray(
                [r[metric] for r in per_sample_rows if r["model"] == m],
                dtype=float,
            )
            xx, yy = ecdf(vals)
            ax.step(
                xx,
                yy,
                where="post",
                label=LATEX_MODEL_LABELS[m],
                color=MODEL_COLORS[m],
                lw=2,
                linestyle=MODEL_LINESTYLES[m],
            )
        ax.set_title(title)
        ax.set_xlabel("Distance")
        ax.set_ylabel("ECDF")
        ax.set_ylim(0.0, 1.0)
        #ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Plot C: Empirical CDF on Prediction_b (shared samples)")
    save_figure_multiformat(fig, output_dir / "plot_C_ecdf", dpi=200)
    plt.close(fig)

    # Plot A+C combined: top row distributions, bottom row ECDF
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5), constrained_layout=True)
    metric_titles = (r"$d_\text{H}(\mathcal{P}, \mathcal{R})$", r"$d_\text{M}(\mathcal{P}, \mathcal{R})$")
    metric_symbols = (r"$d_\text{H}$", r"$d_\text{M}$")

    # Top row: original Plot A style distributions (atom-agnostic)
    for j, (metric, label_text) in enumerate(zip(METRICS, metric_titles)):
        ax = axes[0, j]
        x_positions = np.arange(1, len(model_order) + 1)
        data = []
        for m in model_order:
            vals = np.asarray(
                [r[metric] for r in per_sample_rows if r["model"] == m],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            data.append(vals)

        parts = ax.violinplot(data, showmeans=False, showextrema=False, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_alpha(0.25)
            pc.set_facecolor(MODEL_COLORS[model_order[i]])

        ax.boxplot(data, labels=model_order_latex, showfliers=False)

        for i, m in enumerate(model_order, start=1):
            for sid in HIGHLIGHT_SAMPLE_IDS:
                y_val = by_model_metric[(m, metric)].get(sid)
                if y_val is None or not math.isfinite(y_val):
                    continue
                marker_style, _ = HIGHLIGHT_MARKERS.get(sid, ("o", "Prediction"))
                ax.scatter(
                    i + HIGHLIGHT_RIGHT_SHIFT_SINGLE,
                    y_val,
                    s=85,
                    marker=marker_style,
                    facecolors="none",
                    edgecolors=MODEL_COLORS[m],
                    linewidths=1.0,
                    zorder=10,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_order_latex)
        ax.set_ylabel(label_text + r" (Å)" )
        ax.tick_params(axis="x", rotation=0)
        pred_legend = ax.legend(handles=pred_handles, frameon=False, loc="upper right", fontsize=9)
        for txt in pred_legend.get_texts():
            txt.set_color("black")
        ax.add_artist(pred_legend)

    # Bottom row: Plot C style ECDF
    for j, (metric, symbol_text) in enumerate(zip(METRICS, metric_symbols)):
        ax = axes[1, j]
        for m in model_order:
            vals = np.asarray(
                [r[metric] for r in per_sample_rows if r["model"] == m],
                dtype=float,
            )
            xx, yy = ecdf(vals)
            ax.step(
                xx,
                yy,
                where="post",
                label=LATEX_MODEL_LABELS[m],
                color=MODEL_COLORS[m],
                lw=2,
                linestyle=MODEL_LINESTYLES[m],
            )
        ax.set_xlabel(symbol_text + " (Å)")
        ax.set_ylabel(r"$G(d_\text{H})$" if metric == "hausdorff" else r"$G(d_\text{M})$")
        #ax.set_ylim(0.0, 1.0)
        #ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="lower right")

    # Panel labels (force visible at top-left)
    panel_labels = (("a", axes[0, 0]), ("b", axes[0, 1]), ("c", axes[1, 0]), ("d", axes[1, 1]))
    for label, ax in panel_labels:
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="black",
            zorder=20,
            clip_on=False,
        )

    #fig.suptitle("Combined Plot A+C: Distribution and ECDF on Prediction_b")
    save_figure_multiformat(fig, output_dir / "plot_AC_combined", dpi=200)
    plt.close(fig)

    print(f"Done. Shared samples across models: {len(shared_ids)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
