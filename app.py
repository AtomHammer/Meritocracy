"""Meritocracy Workbench — three-tab UI: World, Library, Explorer.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sim import (
    ApplicationConfig,
    FilterConfig,
    QualityConfig,
    ReviewerAccuracyConfig,
    SigmoidConfig,
    WorldConfig,
    default_world,
)
from workbench import (
    ApplyFilterOrigin,
    FreshDrawOrigin,
    MixOrigin,
    Workbench,
    apply_filter,
    export_distribution_csv,
    fresh_draw,
    import_workbench_into,
    lineage_text,
    load_workbench,
    mix_distributions,
    root_relative_productivity,
    save_workbench,
)


SIGMOID_PRESETS = {
    "Bell (narrow)  k=0.6  σ≈0.14": 0.6,
    "Bell (broad)  k=1.0  σ≈0.20": 1.0,
}


def _is_default_world(world: WorldConfig) -> bool:
    """True if `world` is exactly the package default (helps decide whether a
    user's first edit should sweep away the named defaults)."""
    d = default_world()
    if len(world.quality_configs) != len(d.quality_configs):
        return False
    for a, b in zip(world.quality_configs, d.quality_configs):
        if (a.name != b.name or a.mu != b.mu or a.sigma != b.sigma
                or a.weight != b.weight or a.is_communication != b.is_communication):
            return False
    return True


def _generic_quality(i: int) -> QualityConfig:
    return QualityConfig(name=f"Q{i+1}", mu=0.0, sigma=1.0, weight=1.0)


def _std_to_concentration(mean: float, std: float) -> float:
    """Beta with given mean μ and per-reviewer std s has concentration
    κ = μ(1−μ)/s² − 1. Clamp to ≥1 so Beta always has a sensible shape."""
    var = max(std, 1e-6) ** 2
    max_var = mean * (1.0 - mean)
    var = min(var, max_var * 0.999)
    return max(1.0, max_var / var - 1.0)


def _concentration_to_std(mean: float, concentration: float) -> float:
    """Inverse of the above for showing the current value as a std."""
    return float(np.sqrt(mean * (1.0 - mean) / max(concentration + 1.0, 1e-6)))


def _resize_quality_set(
    current_qcs: list[QualityConfig],
    current_racs: list[ReviewerAccuracyConfig],
    current_jc: list[float],
    current_wc: list[float],
    new_n: int,
    is_default: bool,
) -> tuple[list[QualityConfig], list[ReviewerAccuracyConfig], list[float], list[float]]:
    """Asymmetric resize per the user's spec:
       - From the default world, reducing Nq wipes the named defaults
         (Q1..Q_new generic).
       - From the default world, increasing Nq keeps the defaults and
         appends generic Q_(old+1)..Q_new at the end.
       - Once customized, just preserve as much as possible (chop or extend)."""
    cur_n = len(current_qcs)
    if new_n == cur_n:
        return current_qcs, current_racs, current_jc, current_wc

    if is_default and new_n < cur_n:
        # Wipe the named defaults entirely
        return (
            [_generic_quality(i) for i in range(new_n)],
            [ReviewerAccuracyConfig() for _ in range(new_n)],
            [0.0] * new_n,
            [0.0] * new_n,
        )
    # Extend or chop preserving entries
    if new_n < cur_n:
        return (
            current_qcs[:new_n], current_racs[:new_n],
            current_jc[:new_n], current_wc[:new_n],
        )
    # new_n > cur_n
    pad_q = [_generic_quality(i) for i in range(cur_n, new_n)]
    pad_r = [ReviewerAccuracyConfig() for _ in range(cur_n, new_n)]
    return (
        list(current_qcs) + pad_q,
        list(current_racs) + pad_r,
        list(current_jc) + [0.0] * (new_n - cur_n),
        list(current_wc) + [0.0] * (new_n - cur_n),
    )


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Meritocracy Workbench", layout="wide")

if "workbench" not in st.session_state:
    st.session_state.workbench = Workbench()
if "selected_dist" not in st.session_state:
    st.session_state.selected_dist = None
if "selected_filter" not in st.session_state:
    st.session_state.selected_filter = None
if "show_apply" not in st.session_state:
    st.session_state.show_apply = False
if "show_mix" not in st.session_state:
    st.session_state.show_mix = False

wb: Workbench = st.session_state.workbench


def _format_dist(wb: Workbench, name: str) -> str:
    d = wb.distributions[name]
    if isinstance(d.origin, FreshDrawOrigin):
        kind = "fresh"
    elif isinstance(d.origin, ApplyFilterOrigin):
        kind = f"← {d.origin.filter_name}"
    elif isinstance(d.origin, MixOrigin):
        kind = f"mix({len(d.origin.distribution_names)})"
    else:
        kind = "?"
    return f"{name}   ·   n={d.n:,}   ·   {kind}"


def _format_filter(wb: Workbench, name: str) -> str:
    f = wb.filters[name]
    ss = "SS" if f.self_selection_enabled else "no-SS"
    return (f"{name}   ·   cutoff={f.application.cutoff:,}   ·   "
            f"R={f.application.resource:g}   ·   {ss}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def _skewness(arr):
    if arr.size < 3:
        return float("nan")
    mu, sigma = float(arr.mean()), float(arr.std())
    if sigma == 0:
        return 0.0
    return float(((arr - mu) ** 3).mean() / sigma ** 3)


def hist_potential(d, intermediate=None, parent_potential=None,
                   x_range=None, ideal_cutoff=None, ss_threshold=None,
                   color_by_path=False, stacked=False):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    arrays = [d.potential]
    if intermediate is not None:
        arrays.append(intermediate)
    if parent_potential is not None:
        arrays.append(parent_potential)
    if x_range is not None:
        lo, hi = x_range
    else:
        lo = float(min(a.min() for a in arrays))
        hi = float(max(a.max() for a in arrays))
    bins = np.linspace(lo, hi, 40)
    if parent_potential is not None:
        ax.hist(parent_potential, bins=bins, alpha=0.30,
                label=f"pre-SS (n={len(parent_potential)}, μ={parent_potential.mean():.2f})",
                color="gray")
    if intermediate is not None:
        ax.hist(intermediate, bins=bins, alpha=0.5,
                label=f"post-SS pre-reviewer (n={len(intermediate)}, μ={intermediate.mean():.2f}, skew={_skewness(intermediate):+.2f})",
                color="darkorange")
    if (color_by_path and d.path_label is not None
            and len(set(d.path_label.tolist())) >= 2):
        unique_paths = sorted(set(d.path_label.tolist()))
        path_arrays = [d.potential[d.path_label == p] for p in unique_paths]
        path_labels = [
            f"{p} (n={a.size}, μ={a.mean():.2f})"
            for p, a in zip(unique_paths, path_arrays)
        ]
        path_colors = [_PATH_COLORS[i % len(_PATH_COLORS)]
                        for i in range(len(unique_paths))]
        if stacked:
            ax.hist(path_arrays, bins=bins, stacked=True,
                    color=path_colors, label=path_labels)
        else:
            for arr, color, label in zip(path_arrays, path_colors, path_labels):
                ax.hist(arr, bins=bins, alpha=0.55, color=color, label=label)
    else:
        ax.hist(d.potential, bins=bins, alpha=0.65,
                label=f"final (n={d.n}, μ={d.potential.mean():.2f}, skew={_skewness(d.potential):+.2f})",
                color="seagreen")
    if ideal_cutoff is not None and not np.isnan(ideal_cutoff):
        ax.axvline(ideal_cutoff, color="black", linestyle="--", linewidth=2,
                   label=f"ideal cutoff = {ideal_cutoff:.2f}")
    if ss_threshold is not None and not np.isnan(ss_threshold):
        ax.axvline(ss_threshold, color="dimgray", linestyle=":", linewidth=1.5,
                   label=f"SS threshold = {ss_threshold:.2f}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("True Potential")
    ax.set_ylabel("Count")
    ax.set_title("True Potential")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False)
    fig.tight_layout()
    return fig


_PATH_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_args(parent_dist, applied_mask, survived_among_applied,
                color_by_path: bool):
    """Return a list of (mask, color, label) tuples for plotting layers.
    When color_by_path is True and parent has multi-source path_label,
    color points by their source path; otherwise color by survived/rejected."""
    if color_by_path and parent_dist.path_label is not None:
        labels_applied = parent_dist.path_label[applied_mask]
        unique_labels = sorted(set(labels_applied.tolist()))
        if len(unique_labels) >= 2:
            layers = []
            for i, lbl in enumerate(unique_labels):
                color = _PATH_COLORS[i % len(_PATH_COLORS)]
                mask = labels_applied == lbl
                surv_count = int((mask & survived_among_applied).sum())
                tot_count = int(mask.sum())
                layers.append((mask, color, f"{lbl}  ({surv_count}/{tot_count} survived)"))
            return layers
    # Fallback: survived (green) vs rejected (gray)
    surv = survived_among_applied
    rej = ~surv
    return [
        (rej, "gray", f"rejected ({int(rej.sum()):,})"),
        (surv, "seagreen", f"survived ({int(surv.sum()):,})"),
    ]


def scatter_observed_vs_potential(d, parent_dist, color_by_path=False):
    """Scatter of (true Potential, Observed score) for every applicant.
    Lines for cutoff_score and ideal_cutoff. Coloring: survived/rejected by
    default, or by path_label when color_by_path=True."""
    fa = d.filter_artifacts
    if fa is None or parent_dist is None:
        return None
    applied_idx = np.where(fa.applied_mask)[0]
    if applied_idx.size == 0:
        return None
    pot_for_applied = parent_dist.potential[fa.applied_mask]
    obs = fa.observed_scores
    survived_among_applied = fa.survived_mask[fa.applied_mask]

    # Compute ideal cutoff (K-th best by true Potential, in the parent pool)
    n = parent_dist.n
    k = fa.pool_size_after
    if 0 < k < n:
        order = np.argsort(-parent_dist.potential)
        ideal_cutoff_potential = float(parent_dist.potential[order[k - 1]])
    else:
        ideal_cutoff_potential = None

    fig, ax = plt.subplots(figsize=(7, 5))
    layers = _color_args(parent_dist, fa.applied_mask, survived_among_applied,
                          color_by_path)
    # Subsample if huge so the scatter doesn't render forever
    max_points = 25000
    if applied_idx.size > max_points:
        step = applied_idx.size // max_points
        sub_idx = np.arange(0, applied_idx.size, step)
        for mask, color, label in layers:
            sub_mask = mask[sub_idx]
            ax.scatter(pot_for_applied[sub_idx][sub_mask],
                       obs[sub_idx][sub_mask],
                       s=3, alpha=0.4, color=color, label=label)
        ax.text(0.02, 0.98, f"showing {len(sub_idx):,} of {applied_idx.size:,} applicants",
                transform=ax.transAxes, fontsize=7, va="top",
                color="dimgray", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    else:
        for mask, color, label in layers:
            ax.scatter(pot_for_applied[mask], obs[mask],
                       s=4, alpha=0.4, color=color, label=label)

    ax.axhline(fa.cutoff_score, color="red", linestyle="--", linewidth=1.5,
               label=f"observed cutoff = {fa.cutoff_score:.2f}")
    if ideal_cutoff_potential is not None:
        ax.axvline(ideal_cutoff_potential, color="black", linestyle="--", linewidth=1.5,
                   label=f"ideal cutoff = {ideal_cutoff_potential:.2f}")
    ax.set_xlabel("True Potential (in [0, 1])")
    ax.set_ylabel("Observed score (1-5, weighted average)")
    ax.set_title("Reviewer perception vs reality")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def scatter_per_quality(d, parent_dist, q_idx: int, q_name: str,
                         color_by_path: bool = False):
    """Scatter of (true quality value, reviewer-averaged 1-5 score for that
    quality) for every applicant. Quartile lines on both axes (no filter
    cutoffs — those are at the overall-score level, not per-quality)."""
    fa = d.filter_artifacts
    if fa is None or parent_dist is None or fa.per_quality_observed is None:
        return None
    if q_idx < 0 or q_idx >= fa.per_quality_observed.shape[1]:
        return None
    applied_idx = np.where(fa.applied_mask)[0]
    if applied_idx.size == 0:
        return None
    true_q = parent_dist.qualities[fa.applied_mask, q_idx]
    obs_q = fa.per_quality_observed[:, q_idx]   # 1-5 (averaged, may be float)
    survived_among_applied = fa.survived_mask[fa.applied_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    layers = _color_args(parent_dist, fa.applied_mask, survived_among_applied,
                          color_by_path)
    max_points = 25000
    if applied_idx.size > max_points:
        step = applied_idx.size // max_points
        sub_idx = np.arange(0, applied_idx.size, step)
        for mask, color, label in layers:
            sub_mask = mask[sub_idx]
            ax.scatter(true_q[sub_idx][sub_mask], obs_q[sub_idx][sub_mask],
                       s=3, alpha=0.4, color=color, label=label)
    else:
        for mask, color, label in layers:
            ax.scatter(true_q[mask], obs_q[mask],
                       s=4, alpha=0.4, color=color, label=label)

    # Quartile lines, both axes
    for q in (0.25, 0.5, 0.75):
        ax.axvline(np.quantile(true_q, q), color="lightgray",
                   linestyle=":", linewidth=0.8, zorder=0)
        ax.axhline(np.quantile(obs_q, q), color="lightgray",
                   linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"True {q_name}")
    ax.set_ylabel(f"Reviewer score for {q_name} (1–5, averaged)")
    ax.set_title(f"{q_name}: reviewer perception vs reality")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def scatter_productivity(d, parent_dist, color_by_path: bool = False):
    """Scatter of (true productivity, reviewer-averaged 1-5 productivity score)
    for applicants with a track record. Quartile lines, no cutoffs.

    Applicants with no opportunity (opportunity == 0) are left out: they have
    no record to review, and productivity is not part of their score."""
    fa = d.filter_artifacts
    if fa is None or parent_dist is None or fa.productivity_observed is None:
        return None
    has_record_among_applied = parent_dist.opportunity[fa.applied_mask] > 0
    n_no_record = int((~has_record_among_applied).sum())
    plot_mask = fa.applied_mask & (parent_dist.opportunity > 0)
    applied_idx = np.where(plot_mask)[0]
    if applied_idx.size == 0:
        return None
    true_prod = parent_dist.productivity[plot_mask]
    obs_prod = fa.productivity_observed[has_record_among_applied]
    survived_among_applied = fa.survived_mask[plot_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    layers = _color_args(parent_dist, plot_mask, survived_among_applied,
                          color_by_path)
    if n_no_record > 0:
        ax.text(0.98, 0.02,
                f"{n_no_record:,} applicants with no track record not shown",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    max_points = 25000
    if applied_idx.size > max_points:
        step = applied_idx.size // max_points
        sub_idx = np.arange(0, applied_idx.size, step)
        for mask, color, label in layers:
            sub_mask = mask[sub_idx]
            ax.scatter(true_prod[sub_idx][sub_mask], obs_prod[sub_idx][sub_mask],
                       s=3, alpha=0.4, color=color, label=label)
    else:
        for mask, color, label in layers:
            ax.scatter(true_prod[mask], obs_prod[mask],
                       s=4, alpha=0.4, color=color, label=label)
    for q in (0.25, 0.5, 0.75):
        if true_prod.size > 0:
            ax.axvline(np.quantile(true_prod, q), color="lightgray",
                       linestyle=":", linewidth=0.8, zorder=0)
        ax.axhline(np.quantile(obs_prod, q), color="lightgray",
                   linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xlabel("True productivity")
    ax.set_ylabel("Reviewer score for productivity (1–5, averaged)")
    ax.set_title("Productivity: reviewer perception vs reality")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def hist_filter_mistakes(d, parent_dist, x_range=None, color_by_path=False,
                          stacked=True):
    """Filter-mistakes panel.

    FN side (admit-failure relative to the absolute parent-pool ideal):
      - FN from SS: parent-top-K who didn't apply (lost to self-doubt).
      - FN from reviewer: parent-top-K who applied but got rejected.

    FP side (admit-error judged ONLY against what reviewers could choose
    from). The applicant-pool's K-th-best true Potential is the cutoff
    reviewers SHOULD have hit; admits below it are reviewer ranking errors.
    Admits between the applicant-pool ideal and the parent-pool ideal are
    "necessary admits" — slots that got filled by the next-best applicants
    because some deserving people didn't apply. Those aren't reviewer errors.
    """
    fa = d.filter_artifacts
    if fa is None or parent_dist is None:
        return None
    true_potential_before = parent_dist.potential
    survived_mask = fa.survived_mask
    applied_mask = fa.applied_mask
    k = fa.pool_size_after
    n = len(true_potential_before)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if k <= 0 or k >= n:
        ax.set_title("Filter mistakes (n/a)")
        return fig

    # Parent-pool ideal: top K by true Potential in the full parent pool.
    order_parent = np.argsort(-true_potential_before)
    deserving_parent = np.zeros(n, dtype=bool)
    deserving_parent[order_parent[:k]] = True
    parent_ideal_cutoff = float(true_potential_before[order_parent[k - 1]])

    # Applicant-pool ideal: top K by true Potential AMONG applicants.
    applicant_idx = np.where(applied_mask)[0]
    n_applied = applicant_idx.size
    if n_applied >= k:
        applicant_potentials = true_potential_before[applicant_idx]
        order_app = np.argsort(-applicant_potentials)
        applicant_ideal_cutoff = float(applicant_potentials[order_app[k - 1]])
        deserving_applicant = np.zeros(n, dtype=bool)
        deserving_applicant[applicant_idx[order_app[:k]]] = True
    else:
        # Under-subscription — every applicant is admitted; no reviewer FP possible.
        applicant_ideal_cutoff = (float(true_potential_before[applied_mask].min())
                                   if n_applied > 0 else parent_ideal_cutoff)
        deserving_applicant = applied_mask.copy()

    # Categories
    fn_from_ss = (~applied_mask) & deserving_parent
    fn_from_reviewer = applied_mask & (~survived_mask) & deserving_parent
    fp_from_reviewer = survived_mask & (~deserving_applicant)
    necessary_admits = survived_mask & deserving_applicant & (~deserving_parent)

    if x_range is not None:
        lo, hi = x_range
    else:
        lo, hi = float(true_potential_before.min()), float(true_potential_before.max())
    bins = np.linspace(lo, hi, 40)

    use_path = (color_by_path and parent_dist.path_label is not None
                and len(set(parent_dist.path_label.tolist())) >= 2)

    if use_path:
        # Replace category bars with per-path bars on each side (FP, FN).
        # Loses the FP-from-reviewer / necessary-admits split but gives a
        # path-by-path comparison. FP appears left of the cutoffs, FN right
        # — separation is positional, not by hatching when stacked.
        unique_paths = sorted(set(parent_dist.path_label.tolist()))
        fp_all = survived_mask & (~deserving_parent)
        fn_all = (~survived_mask) & deserving_parent
        # Per-path FP arrays
        fp_arrays, fp_labels, fp_colors = [], [], []
        for i, p in enumerate(unique_paths):
            mask = parent_dist.path_label == p
            fp_p = fp_all & mask
            if fp_p.any():
                fp_arrays.append(true_potential_before[fp_p])
                fp_labels.append(f"FP: {p} (n={int(fp_p.sum())})")
                fp_colors.append(_PATH_COLORS[i % len(_PATH_COLORS)])
        # FP bars: solid path color, no hatching. FN bars: same path color,
        # but hatched. This way path identity is read from color, FP-vs-FN
        # is read from texture — preventing the "two same-colored entries
        # in the legend" confusion.
        if fp_arrays:
            if stacked:
                ax.hist(fp_arrays, bins=bins, stacked=True,
                        color=fp_colors, label=fp_labels,
                        edgecolor="white", linewidth=0.3)
            else:
                for arr, color, label in zip(fp_arrays, fp_colors, fp_labels):
                    ax.hist(arr, bins=bins, alpha=0.55, color=color,
                            edgecolor="white", linewidth=0.3, label=label)
        # Per-path FN arrays
        fn_arrays, fn_labels, fn_colors = [], [], []
        for i, p in enumerate(unique_paths):
            mask = parent_dist.path_label == p
            fn_p = fn_all & mask
            if fn_p.any():
                fn_arrays.append(true_potential_before[fn_p])
                fn_labels.append(f"FN: {p} (n={int(fn_p.sum())})")
                fn_colors.append(_PATH_COLORS[i % len(_PATH_COLORS)])
        if fn_arrays:
            if stacked:
                ax.hist(fn_arrays, bins=bins, stacked=True,
                        color=fn_colors, label=fn_labels,
                        edgecolor="black", linewidth=0.4,
                        hatch="//")
            else:
                for arr, color, label in zip(fn_arrays, fn_colors, fn_labels):
                    ax.hist(arr, bins=bins, alpha=0.45, color=color,
                            edgecolor="black", linewidth=0.4, hatch="//",
                            label=label)
    elif fp_from_reviewer.any() or necessary_admits.any():
        # Default category bars: stacked or overlapped per `stacked` flag.
        if stacked:
            ax.hist(
                [true_potential_before[fp_from_reviewer],
                 true_potential_before[necessary_admits]],
                bins=bins, stacked=True,
                color=["crimson", "lightgray"],
                label=[
                    f"FP from reviewer (n={int(fp_from_reviewer.sum())})",
                    f"Filled, no better applicant (n={int(necessary_admits.sum())})",
                ],
            )
        else:
            if necessary_admits.any():
                ax.hist(true_potential_before[necessary_admits], bins=bins,
                        alpha=0.55, color="lightgray",
                        label=f"Filled, no better applicant (n={int(necessary_admits.sum())})")
            if fp_from_reviewer.any():
                ax.hist(true_potential_before[fp_from_reviewer], bins=bins,
                        alpha=0.65, color="crimson",
                        label=f"FP from reviewer (n={int(fp_from_reviewer.sum())})")
    # FN side: SS losses + reviewer rejections of deserving. Each sub-series
    # is included only when it actually has members, so a filter with no
    # self-selection (fn_from_ss is always empty) shows no dead "FN from SS"
    # legend entry. Skipped when use_path is on (per-path FN overlays already
    # drawn above).
    if not use_path:
        fn_series = []  # (values, color, alpha, label)
        if fn_from_ss.any():
            fn_series.append(
                (true_potential_before[fn_from_ss], "lightsteelblue", 0.55,
                 f"FN from SS — didn't apply (n={int(fn_from_ss.sum())})"))
        if fn_from_reviewer.any():
            fn_series.append(
                (true_potential_before[fn_from_reviewer], "royalblue", 0.65,
                 f"FN from reviewer — rejected (n={int(fn_from_reviewer.sum())})"))
        if fn_series:
            if stacked:
                ax.hist([s[0] for s in fn_series], bins=bins, stacked=True,
                        color=[s[1] for s in fn_series],
                        label=[s[3] for s in fn_series])
            else:
                for values, color, alpha, label in fn_series:
                    ax.hist(values, bins=bins, alpha=alpha, color=color,
                            label=label)

    # Cutoff lines
    ax.axvline(parent_ideal_cutoff, color="black", linestyle="--", linewidth=2,
               label=f"Absolute ideal cutoff = {parent_ideal_cutoff:.2f}")
    if abs(applicant_ideal_cutoff - parent_ideal_cutoff) > 1e-6:
        ax.axvline(applicant_ideal_cutoff, color="dimgray", linestyle="--",
                   linewidth=1.4,
                   label=f"Applicant-pool ideal = {applicant_ideal_cutoff:.2f}")
    if fa.self_selection_enabled and not np.isnan(fa.self_selection_threshold):
        ax.axvline(fa.self_selection_threshold, color="dimgray", linestyle=":",
                   linewidth=1.2,
                   label=f"SS threshold = {fa.self_selection_threshold:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_xlabel("True Potential")
    ax.set_ylabel("Count")
    ax.set_title("Filter mistakes")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False)
    fig.tight_layout()
    return fig


def lotka_loglog(full_productivity, n_initial, root_name):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    nonzero = full_productivity[full_productivity > 0]
    n_zero = int((full_productivity == 0).sum())
    if nonzero.size < 5:
        ax.set_title("Lotka (insufficient data)")
        return fig
    lo, hi = max(nonzero.min(), 1e-6), nonzero.max()
    bins = np.geomspace(lo, hi, 30)
    counts, edges = np.histogram(nonzero, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = counts > 0
    ax.loglog(centers[valid], counts[valid], "o-", color="steelblue",
              markersize=5, label="simulated")
    if valid.sum() >= 3:
        right_mask = valid & (centers >= np.median(centers[valid]))
        if right_mask.any():
            anchor_x = centers[right_mask][0]
            anchor_y = counts[right_mask][0]
            xline = centers[valid]
            ax.loglog(xline, anchor_y * (xline / anchor_x) ** -2, "--",
                      color="crimson", linewidth=1.4, label="Lotka slope = -2")
    ax.set_xlabel("Productivity (log)")
    ax.set_ylabel("Count of scientists (log)")
    ax.set_title(f"Final productivity rooted at '{root_name}'  ·  "
                 f"{nonzero.size:,} of {n_initial:,} have nonzero productivity")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------

st.title("Meritocracy Workbench")
st.caption(
    f"**{len(wb.distributions)}** distributions · **{len(wb.filters)}** filters · "
    f"world: **{len(wb.world.quality_configs)} qualities** "
    f"({', '.join(c.name for c in wb.world.quality_configs)})"
)

top_cols = st.columns([2, 5, 2, 2])
with top_cols[0]:
    buf = io.BytesIO()
    save_workbench(wb, buf)
    st.download_button(
        "💾 Save", data=buf.getvalue(),
        file_name="workbench.pkl",
        mime="application/octet-stream",
        width="stretch",
    )
with top_cols[1]:
    upload = st.file_uploader(
        "Upload a workbench (.pkl)", type=["pkl"],
        key="wb_upload", label_visibility="collapsed",
    )
do_replace = top_cols[2].button(
    "Replace ←", disabled=upload is None, width="stretch",
    help="Discard current workbench and load this one.",
)
do_merge = top_cols[3].button(
    "Merge +", disabled=upload is None, width="stretch",
    help="Merge contents into current workbench. Worlds must match.",
)

if do_replace and upload is not None:
    try:
        st.session_state.workbench = load_workbench(upload)
        st.session_state.selected_dist = None
        st.session_state.selected_filter = None
        st.success("Loaded.")
        st.rerun()
    except Exception as e:
        st.error(f"Load failed: {e}")
if do_merge and upload is not None:
    try:
        source = load_workbench(upload)
        renames = import_workbench_into(wb, source)
        msg = (f"Imported {len(source.distributions)} distributions "
               f"and {len(source.filters)} filters.")
        if renames:
            msg += f"  Renames: {renames}"
        st.success(msg)
        st.rerun()
    except Exception as e:
        st.error(f"Import failed: {e}")

st.divider()

tab_world, tab_library, tab_explorer = st.tabs(
    ["🌍 World", "📚 Library", "🔬 Explorer"]
)


# ---------------------------------------------------------------------------
# WORLD TAB
# ---------------------------------------------------------------------------

with tab_world:
    locked = wb.world_locked()
    if locked:
        st.warning(
            "World structure is **locked** because there are existing "
            "distributions or filters that depend on it. To change qualities "
            "or sigmoid, delete all distributions and filters first. "
            "(Reviewer accuracies, correlations, and SS settings can still be "
            "tweaked — they only affect *future* filter applications.)"
        )

    st.subheader("Quality space")
    st.caption(
        "The qualities every scientist has and every reviewer scores against. "
        "These define the world the simulation runs in. Defaults are "
        "literature-informed (NIH/NSF dimensions, IRR ~0.4–0.7). "
        "Changing the count from default replaces the named set; "
        "increasing it adds generic Qn at the end."
    )

    Nq_current = len(wb.world.quality_configs)
    # Nq input lives outside the form so we can rebuild form contents
    # immediately when the count changes.
    Nq = int(st.number_input(
        "Number of qualities", min_value=1, max_value=10,
        value=Nq_current, disabled=locked, key="world_nq_input",
    ))

    # max_score: lives outside the form too, since changes auto-rescale the
    # pile_mean defaults below. Text-box style (number_input).
    cur_max = int(getattr(wb.world, "max_score", 5))
    new_max = int(st.number_input(
        "Max reviewer score (integer 1..N)",
        min_value=2, max_value=20, value=cur_max, step=1,
        disabled=locked, key="world_max_score_input",
        help=("Reviewers emit integers from 1 to N for each quality and for "
              "productivity. Default 5 (NSF-like). 9 ≈ NIH-like. Higher N "
              "gives finer score gradations (good for robustness checks) "
              "but doesn't add signal beyond reviewer accuracy."),
    ))
    if new_max != cur_max and not locked:
        # Auto-scale every pile_mean by (new / cur) so the relative position
        # on the scale is preserved. Mutates world directly so the form
        # below renders with the scaled values; clears slider session keys
        # so they don't carry stale values.
        scale = new_max / cur_max
        wb.world.max_score = new_max
        for q, c in enumerate(wb.world.reviewer_accuracy_configs):
            wb.world.reviewer_accuracy_configs[q] = ReviewerAccuracyConfig(
                mean=c.mean, concentration=c.concentration,
                pile_mean=float(np.clip(c.pile_mean * scale, 1.0, new_max)),
                pile_sigma=c.pile_sigma,
            )
            st.session_state.pop(f"world_pile_m_{q}", None)
        wb.world.productivity_pile_mean = float(np.clip(
            wb.world.productivity_pile_mean * scale, 1.0, new_max
        ))
        st.session_state.pop("world_prod_pile_m", None)
        st.rerun()
    # Apply the asymmetric resize logic to derive what to render
    base_qcs, base_racs, base_jc, base_wc = _resize_quality_set(
        list(wb.world.quality_configs),
        list(wb.world.reviewer_accuracy_configs),
        list(wb.world.judgment_correlations),
        list(wb.world.weight_correlations),
        Nq,
        _is_default_world(wb.world),
    )

    with st.form("world_form"):
        new_quality_configs: list[QualityConfig] = []
        new_reviewer_accuracy_configs: list[ReviewerAccuracyConfig] = []
        new_judge_corrs: list[float] = []
        new_weight_corrs: list[float] = []

        for q in range(Nq):
            existing = base_qcs[q]
            existing_ra = base_racs[q]
            existing_jc = base_jc[q]
            existing_wc = base_wc[q]

            with st.expander(
                f"{existing.name}",
                expanded=(q == 0),
            ):
                qn = st.text_input("Name", value=existing.name,
                                    key=f"world_qn_{q}", disabled=locked)
                cols = st.columns(3)
                qmu = cols[0].number_input(
                    "μ", value=float(existing.mu), step=0.1,
                    key=f"world_qmu_{q}", disabled=locked,
                )
                qsigma = cols[1].number_input(
                    "σ", value=float(existing.sigma), min_value=0.001,
                    step=0.1, key=f"world_qsigma_{q}", disabled=locked,
                )
                qweight = cols[2].number_input(
                    "Weight in Potential",
                    value=float(existing.weight), step=0.1,
                    key=f"world_qw_{q}", disabled=locked,
                )

                qbi = st.checkbox(
                    "Biphasic (yes/no)", value=bool(existing.biphasic),
                    key=f"world_qbi_{q}", disabled=locked,
                )
                if qbi:
                    cols = st.columns(2)
                    qpyes = cols[0].slider("P(yes)", 0.01, 0.99,
                                            float(existing.p_yes),
                                            0.01, key=f"world_pyes_{q}",
                                            disabled=locked)
                    qdelta = cols[1].number_input(
                        "Delta", min_value=0.0, max_value=10.0,
                        value=float(existing.delta),
                        step=0.05, key=f"world_delta_{q}", disabled=locked,
                    )
                else:
                    qpyes, qdelta = 0.4, 1.0

                qcomm = st.checkbox(
                    "Communication-type quality (modifies perception of others)",
                    value=bool(existing.is_communication),
                    key=f"world_comm_{q}", disabled=locked,
                    help=("When checked, this quality is special: scientists "
                          "high on it get an additive perception bonus on "
                          "OTHER qualities (polish), and protection against "
                          "unlucky downward noise (protection). Default 'on' "
                          "for the Communication quality."),
                )
                if qcomm:
                    cols = st.columns(2)
                    polish = cols[0].slider(
                        "Polish strength",
                        0.0, 2.0,
                        float(existing.polish_strength),
                        0.05, key=f"world_polish_{q}", disabled=locked,
                    )
                    protection = cols[1].slider(
                        "Protection strength",
                        0.0, 2.0,
                        float(existing.protection_strength),
                        0.05, key=f"world_protect_{q}", disabled=locked,
                    )
                else:
                    polish, protection = 0.0, 0.0

                st.markdown("**Reviewer perception of this quality**")
                cols = st.columns(2)
                acc_m = cols[0].slider(
                    "Reviewer accuracy mean",
                    0.0, 1.0, float(existing_ra.mean), 0.01,
                    key=f"world_acc_m_{q}",
                )
                # Slider directly controls reviewer-to-reviewer std (the
                # actually-meaningful quantity); we convert to Beta concentration
                # internally.
                cur_std = _concentration_to_std(existing_ra.mean, existing_ra.concentration)
                acc_std = cols[1].slider(
                    "Spread of reviewer accuracy (σ across reviewers)",
                    0.005, 0.30, float(cur_std), 0.005,
                    key=f"world_acc_std_{q}",
                    help=("How much reviewers differ from each other in their "
                          "accuracy on this quality. 0.05 = tight agreement; "
                          "0.20 = wide variation."),
                )
                acc_c = _std_to_concentration(acc_m, acc_std)
                cols[1].caption(
                    f"→ Most reviewers within μ ± {2*acc_std:.2f} (κ ≈ {acc_c:.1f})."
                )
                jc = st.slider(
                    "Judgment correlation  (corr of reviewer's own value with their accuracy on this Q)",
                    -1.0, 1.0, float(existing_jc), 0.05,
                    key=f"world_jc_{q}",
                    help=("0 = no correlation (default). +1 = reviewers high "
                          "on this Q see it more clearly. -1 = Dunning-Kruger; "
                          "high-Q reviewers are worse at judging it."),
                )
                # Weight correlation is currently disabled. Field preserved
                # for forward compat; we may turn it back on later.
                wc = float(existing_wc)

                st.markdown("**Pile calibration (1-5 scoring)**")
                cols = st.columns(2)
                pile_mean = cols[0].slider(
                    f"Pile score mean (target avg of 1-{new_max} scores)",
                    1.0, float(new_max),
                    float(np.clip(existing_ra.pile_mean, 1.0, new_max)),
                    0.1, key=f"world_pile_m_{q}",
                    help=("Reviewers spread their pile across the 1-5 scale "
                          "via a Normal calibration. This sets the target "
                          "mean. NIH-like programs typically run ~3.8."),
                )
                pile_sigma = cols[1].slider(
                    "Pile score spread (σ on the 1-5 scale)",
                    0.2, 2.0, float(existing_ra.pile_sigma), 0.05,
                    key=f"world_pile_s_{q}",
                    help=("Larger σ uses more of the 1-5 range; smaller σ "
                          "concentrates scores near the pile mean."),
                )

                new_quality_configs.append(QualityConfig(
                    name=qn, mu=qmu, sigma=qsigma, weight=qweight,
                    biphasic=qbi, p_yes=qpyes, delta=qdelta,
                    is_communication=qcomm,
                    polish_strength=polish, protection_strength=protection,
                ))
                new_reviewer_accuracy_configs.append(
                    ReviewerAccuracyConfig(
                        mean=acc_m, concentration=acc_c,
                        pile_mean=pile_mean, pile_sigma=pile_sigma,
                    )
                )
                new_judge_corrs.append(jc)
                new_weight_corrs.append(wc)

        st.subheader("Sigmoid (raw → Potential ∈ [0,1])")
        cur_k = wb.world.sigmoid_config.k
        sigmoid_label = st.selectbox(
            "Potential shape",
            list(SIGMOID_PRESETS.keys()),
            index=0 if cur_k == 0.6 else 1,
            disabled=locked,
        )

        st.subheader("Self-assessment by scientists")
        cols = st.columns(2)
        ss_m = cols[0].slider("SS accuracy mean", 0.0, 1.0,
                                float(wb.world.ss_accuracy_mean), 0.01)
        cur_ss_std = _concentration_to_std(
            wb.world.ss_accuracy_mean, wb.world.ss_accuracy_concentration,
        )
        ss_std = cols[1].slider(
            "Spread of self-assessment accuracy (σ across scientists)",
            0.005, 0.30, float(cur_ss_std), 0.005,
            help="How much scientists differ in their self-assessment ability.",
        )
        ss_c = _std_to_concentration(ss_m, ss_std)
        cols[1].caption(f"→ Most scientists within μ ± {2*ss_std:.2f} (κ ≈ {ss_c:.1f}).")

        st.subheader("Reviewer perception of productivity")
        cols = st.columns(2)
        prod_m = cols[0].slider("Productivity reviewer accuracy mean",
                                  0.0, 1.0,
                                  float(wb.world.productivity_accuracy_mean), 0.01)
        cur_prod_std = _concentration_to_std(
            wb.world.productivity_accuracy_mean,
            wb.world.productivity_accuracy_concentration,
        )
        prod_std = cols[1].slider(
            "Spread of productivity-accuracy (σ across reviewers)",
            0.005, 0.30, float(cur_prod_std), 0.005,
            help="How much reviewers differ in their productivity assessment.",
        )
        prod_c = _std_to_concentration(prod_m, prod_std)
        cols[1].caption(f"→ Most reviewers within μ ± {2*prod_std:.2f} (κ ≈ {prod_c:.1f}).")

        st.markdown(f"**Productivity pile calibration (1-{new_max} scoring)**")
        cols = st.columns(2)
        prod_pile_m = cols[0].slider(
            "Productivity pile mean", 1.0, float(new_max),
            float(np.clip(wb.world.productivity_pile_mean, 1.0, new_max)), 0.1,
            key="world_prod_pile_m",
        )
        prod_pile_s = cols[1].slider(
            "Productivity pile spread (σ)", 0.2, 2.0,
            float(wb.world.productivity_pile_sigma), 0.05,
            key="world_prod_pile_s",
        )

        save = st.form_submit_button("Save world", type="primary")

    if save:
        new_world = WorldConfig(
            quality_configs=new_quality_configs,
            sigmoid_config=SigmoidConfig(k=SIGMOID_PRESETS[sigmoid_label]),
            reviewer_accuracy_configs=new_reviewer_accuracy_configs,
            judgment_correlations=new_judge_corrs,
            weight_correlations=new_weight_corrs,
            ss_accuracy_mean=ss_m,
            ss_accuracy_concentration=ss_c,
            productivity_accuracy_mean=prod_m,
            productivity_accuracy_concentration=prod_c,
            productivity_pile_mean=prod_pile_m,
            productivity_pile_sigma=prod_pile_s,
            max_score=new_max,
        )
        wb.world = new_world
        st.success("World updated.")
        st.rerun()

    st.divider()
    if st.button("Reset world to defaults", disabled=locked):
        wb.world = default_world()
        st.success("Reset.")
        st.rerun()


# ---------------------------------------------------------------------------
# LIBRARY TAB
# ---------------------------------------------------------------------------

with tab_library:
    dist_col, filter_col = st.columns(2)

    with dist_col:
        st.subheader("Distributions")
        names = wb.distribution_names()
        if not names:
            st.info("No distributions yet. Use **➕ New distribution** below.")
            st.session_state.selected_dist = None
        else:
            if st.session_state.selected_dist not in names:
                st.session_state.selected_dist = names[0]
                st.session_state.pop("selected_dist_radio", None)
            sel = st.radio(
                "Select", names,
                format_func=lambda x: _format_dist(wb, x),
                index=names.index(st.session_state.selected_dist),
                key="selected_dist_radio",
                label_visibility="collapsed",
            )
            st.session_state.selected_dist = sel

    with filter_col:
        st.subheader("Filters")
        names = wb.filter_names()
        if not names:
            st.info("No filters yet. Use **➕ New filter** below.")
            st.session_state.selected_filter = None
        else:
            if st.session_state.selected_filter not in names:
                st.session_state.selected_filter = names[0]
                st.session_state.pop("selected_filter_radio", None)
            sel = st.radio(
                "Select", names,
                format_func=lambda x: _format_filter(wb, x),
                index=names.index(st.session_state.selected_filter),
                key="selected_filter_radio",
                label_visibility="collapsed",
            )
            st.session_state.selected_filter = sel

    # Selected-filter details panel: full config of the currently-selected
    # filter, for verification before applying.
    if st.session_state.selected_filter is not None:
        f = wb.filters[st.session_state.selected_filter]
        q_names = [c.name for c in wb.world.quality_configs]
        with st.expander(
            f"📄 Selected filter details: {st.session_state.selected_filter}",
            expanded=False,
        ):
            ac = f.application
            st.markdown(
                f"- **Cutoff** (number passing): **{ac.cutoff:,}**  "
                f"·  **Resource** awarded to passers: **{ac.resource:g}**\n"
                f"- **Reads** per scientist: **{ac.reads}**  "
                f"·  Target **assessments per reviewer**: **{ac.assessments_per_reviewer}**"
            )
            ss_line = (
                f"- **Self-selection**: enabled  "
                f"·  cushion factor = **{f.cushion_factor:.2f}**  "
                f"·  equilibrate = **{f.equilibrate_selectivity}**"
                if f.self_selection_enabled
                else "- **Self-selection**: disabled"
            )
            st.markdown(ss_line)
            # Per-quality weights row
            qw = (f.quality_weights if f.quality_weights is not None
                  else [c.weight for c in wb.world.quality_configs])
            weight_lines = [
                f"  - **{qn}**: weight = **{w:.2f}**"
                for qn, w in zip(q_names, qw)
            ]
            st.markdown("- **Score weights** (1-5 weighted average):\n"
                        + "\n".join(weight_lines)
                        + f"\n  - **Productivity**: weight = **{f.productivity_weight:.2f}**"
                        f"  ·  opportunity credit β = **{f.opportunity_credit:.2f}**")

    st.divider()

    st.subheader("Operations")
    op1, op2, op3, op4 = st.columns(4)
    can_apply = (st.session_state.selected_dist is not None
                 and st.session_state.selected_filter is not None)
    do_apply = op1.button("▶ Apply selected filter to selected distribution",
                           disabled=not can_apply, width="stretch")
    if do_apply:
        st.session_state.show_apply = True
    do_mix = op2.button("✚ Mix distributions",
                         disabled=len(wb.distributions) < 2, width="stretch")
    if do_mix:
        st.session_state.show_mix = True
    can_del_d = (st.session_state.selected_dist is not None
                 and wb.can_remove_distribution(st.session_state.selected_dist))
    do_del_d = op3.button("🗑 Delete distribution",
                          disabled=not can_del_d, width="stretch")
    can_del_f = (st.session_state.selected_filter is not None
                 and wb.can_remove_filter(st.session_state.selected_filter))
    do_del_f = op4.button("🗑 Delete filter",
                          disabled=not can_del_f, width="stretch")

    if do_del_d:
        try:
            wb.remove_distribution(st.session_state.selected_dist)
            st.session_state.selected_dist = None
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if do_del_f:
        try:
            wb.remove_filter(st.session_state.selected_filter)
            st.session_state.selected_filter = None
            st.rerun()
        except Exception as e:
            st.error(str(e))

    # ---------------- Apply form ----------------
    if st.session_state.show_apply and can_apply:
        base_default = (f"{st.session_state.selected_dist} → "
                        f"{st.session_state.selected_filter}")
        default_name = base_default
        i = 2
        while default_name in wb.distributions:
            default_name = f"{base_default} ({i})"
            i += 1
        with st.form("apply_form", border=True):
            st.write(f"**Apply** filter `{st.session_state.selected_filter}` "
                     f"to distribution `{st.session_state.selected_dist}`")
            new_name = st.text_input("Output name", value=default_name)
            seed = int(st.number_input("Seed", value=0, step=1, key="apply_seed"))
            create = st.form_submit_button("Create", type="primary")
            cancel = st.form_submit_button("Cancel")
        if create:
            try:
                clean_name = new_name.strip()
                if not clean_name:
                    raise ValueError("Output name cannot be empty.")
                if clean_name in wb.distributions:
                    raise ValueError(f"'{clean_name}' already exists.")
                with st.spinner(f"Applying {st.session_state.selected_filter}…"):
                    new_d = apply_filter(
                        wb,
                        wb.distributions[st.session_state.selected_dist],
                        wb.filters[st.session_state.selected_filter],
                        clean_name, seed,
                    )
                    wb.add_distribution(new_d)
                st.session_state.show_apply = False
                st.session_state.selected_dist = new_d.name
                st.session_state.pop("selected_dist_radio", None)
                st.success(f"Created '{new_d.name}' (n={new_d.n:,})")
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Apply failed: {e}")
        if cancel:
            st.session_state.show_apply = False
            st.rerun()

    # ---------------- Mix form ----------------
    if st.session_state.show_mix and len(wb.distributions) >= 2:
        with st.form("mix_form", border=True):
            st.write("**Mix distributions** (weighted bootstrap)")
            chosen = st.multiselect(
                "Inputs", wb.distribution_names(),
                default=wb.distribution_names()[:2],
            )
            w_str = st.text_input(
                "Weights (comma-separated, in chosen order; auto-normalize)",
                value=", ".join("1.0" for _ in chosen) if chosen else "1.0",
            )
            new_name = st.text_input("Output name", value="Mix")
            target_n = int(st.number_input(
                "Output size", min_value=10, max_value=2_000_000,
                value=10_000, step=1000,
            ))
            seed = int(st.number_input("Seed", value=0, step=1, key="mix_seed"))
            create = st.form_submit_button("Create", type="primary")
            cancel = st.form_submit_button("Cancel")
        if create:
            try:
                clean = new_name.strip()
                if not clean:
                    raise ValueError("Output name cannot be empty.")
                if clean in wb.distributions:
                    raise ValueError(f"'{clean}' already exists.")
                weights = [float(w.strip()) for w in w_str.split(",")]
                inputs = [wb.distributions[n] for n in chosen]
                with st.spinner("Mixing…"):
                    new_d = mix_distributions(
                        inputs, weights, clean, target_n, seed,
                    )
                    wb.add_distribution(new_d)
                st.session_state.show_mix = False
                st.session_state.selected_dist = new_d.name
                st.session_state.pop("selected_dist_radio", None)
                st.success(f"Created '{new_d.name}' (n={new_d.n:,})")
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Mix failed: {e}")
        if cancel:
            st.session_state.show_mix = False
            st.rerun()

    # ---------------- New distribution form ----------------
    with st.expander("➕ New distribution from world", expanded=False):
        with st.form("new_dist_form"):
            d_name = st.text_input("Name", value=f"Pool {len(wb.distributions) + 1}")
            d_n = int(st.number_input(
                "N (population size)", min_value=10, max_value=10_000_000,
                value=100_000, step=10_000,
            ))
            d_seed = int(st.number_input("Seed", value=0, step=1, key="newd_seed"))
            create = st.form_submit_button("Create distribution", type="primary")
        if create:
            try:
                with st.spinner("Drawing…"):
                    new_d = fresh_draw(wb, d_name.strip(), d_n, d_seed)
                wb.add_distribution(new_d)
                st.session_state.selected_dist = new_d.name
                st.session_state.pop("selected_dist_radio", None)
                st.success(f"Created '{new_d.name}' (n={new_d.n:,})")
                st.rerun()
            except Exception as e:
                st.error(f"Create failed: {e}")

    # ---------------- New filter form ----------------
    with st.expander("➕ New filter", expanded=False):
        Nq = len(wb.world.quality_configs)
        with st.form("new_filter_form"):
            f_name = st.text_input("Name", value=f"Filter {len(wb.filters) + 1}",
                                    key="newf_name")
            cols = st.columns(2)
            f_cutoff = int(cols[0].number_input(
                "Number passing (cutoff)", min_value=1, max_value=10_000_000,
                value=10_000, step=100, key="newf_cutoff",
            ))
            f_resource = float(cols[1].number_input(
                "Resource awarded to passers", min_value=0.0, max_value=1000.0,
                value=1.0, step=0.1, key="newf_resource",
            ))
            cols = st.columns(2)
            f_reads = int(cols[0].number_input(
                "Reads per scientist", min_value=1, max_value=20, value=3,
                key="newf_reads",
            ))
            f_assess = int(cols[1].number_input(
                "Target assessments per reviewer", min_value=1, max_value=500,
                value=15, key="newf_assess",
            ))

            st.markdown("**Self-selection**")
            ss_on = st.checkbox("Enable self-selection", value=True, key="newf_ss")
            cushion = st.slider(
                "Optimism cushion factor",
                1.0, 3.0, 1.5, 0.05, key="newf_cushion", disabled=not ss_on,
                help=(
                    "Determines how generous a scientist's 'I might as well "
                    "try' threshold is. Concretely: a scientist applies if "
                    "they perceive themselves in the top "
                    "(cushion × actual-acceptance-rate) of the pool. "
                    "Cushion = 1.0 means apply only if you think you're "
                    "exactly at the bar; 1.5 (default) means apply if you "
                    "think you're in the top 1.5× the slots; 3.0 means "
                    "apply pretty much regardless. There is no canonical "
                    "empirical value for this — it's a behavioral knob."
                ),
            )
            equilibrate = st.checkbox(
                "Equilibrate selectivity (raise cushion to fill cutoff if needed)",
                value=True, key="newf_eq", disabled=not ss_on,
                help=(
                    "When the SS step under-subscribes the cutoff, iterate "
                    "the cushion upward (capped at X=1) so the competition "
                    "fills. Models 'people calibrate to actual selectivity "
                    "over time'. Disable to study under-subscription."
                ),
            )

            st.markdown("**Score weights** (0 to 1; weighted average of 1-5 scores)")
            st.caption("Each reviewer scores each dimension 1-5; the final "
                       "score is a weighted average. Weight 1.0 = full "
                       "importance, 0.5 = half, 0 = ignored. Defaults are "
                       "inherited from the world.")
            cols = st.columns(min(Nq + 1, 5))
            quality_weight_inputs: list[float] = []
            for q in range(Nq):
                qcfg = wb.world.quality_configs[q]
                init_val = min(1.0, max(0.0, float(qcfg.weight)))
                w = cols[q % len(cols)].slider(
                    f"{qcfg.name}",
                    0.0, 1.0, init_val, 0.05,
                    key=f"newf_qw_{q}",
                )
                quality_weight_inputs.append(w)
            prod_w = cols[Nq % len(cols)].slider(
                "Productivity",
                0.0, 1.0, 1.0, 0.05, key="newf_pw",
                help=(
                    "Weight on the productivity-per-opportunity dimension "
                    "(treated as just another quality). Default 1.0 — same "
                    "as full-importance quality. Set to 0 to ignore prior "
                    "track records (the productivity dimension is then fully "
                    "off — not computed and not shown in the per-quality "
                    "drill-down).\n\n"
                    "Applicants with opportunity = 0 (no prior track record) "
                    "are assigned the **neutral pile-mean score** for "
                    "productivity (rounds to 4 by default), so they aren't "
                    "penalized for never having had the chance — reviewers "
                    "default to a middle score when there's nothing to "
                    "assess. Calibration of the productivity dimension runs "
                    "only on the subset of the reviewer's pile that has "
                    "prior opportunity."
                ),
            )

            beta = st.slider(
                "Opportunity credit (β)",
                0.0, 1.0, 0.0, 0.05, key="newf_pb",
                help=(
                    "How much reviewers reward raw past resources, "
                    "regardless of what was made of them. β = 0: pure "
                    "conversion-rate (productivity / opportunity). "
                    "β = 1: pure log(opportunity) — credit for having had "
                    "the opportunity at all. Real reviewers sit between. "
                    "Currently inert when all applicants share the same "
                    "opportunity history (no path divergence)."
                ),
            )

            create = st.form_submit_button("Create filter", type="primary")
        if create:
            try:
                f = FilterConfig(
                    name=f_name.strip(),
                    application=ApplicationConfig(
                        name=f_name.strip(),
                        reads=f_reads,
                        assessments_per_reviewer=f_assess,
                        cutoff=f_cutoff,
                        resource=f_resource,
                    ),
                    self_selection_enabled=ss_on,
                    cushion_factor=cushion,
                    equilibrate_selectivity=equilibrate,
                    quality_weights=quality_weight_inputs,
                    productivity_weight=prod_w,
                    opportunity_credit=beta,
                )
                wb.add_filter(f)
                st.session_state.selected_filter = f.name
                st.session_state.pop("selected_filter_radio", None)
                st.success(f"Created filter '{f.name}'")
                st.rerun()
            except Exception as e:
                st.error(f"Create failed: {e}")


# ---------------------------------------------------------------------------
# EXPLORER TAB
# ---------------------------------------------------------------------------

with tab_explorer:
    if st.session_state.selected_dist is None or st.session_state.selected_dist not in wb.distributions:
        st.info("Select a distribution from the Library tab.")
    else:
        d = wb.distributions[st.session_state.selected_dist]
        st.header(d.name)

        cols = st.columns(5)
        cols[0].metric("n", f"{d.n:,}")
        cols[1].metric("Mean Potential", f"{d.potential.mean():.3f}")
        cols[2].metric("σ Potential", f"{d.potential.std():.3f}")
        cols[3].metric("Mean Productivity", f"{d.productivity.mean():.3f}")
        cols[4].metric("Mean Opportunity", f"{d.opportunity.mean():.3f}")

        with st.expander("⬇ Export this population (CSV)", expanded=False):
            st.caption(
                "One row per scientist, for digging in with pandas / R / MATLAB. "
                "**Filter result:** every parent-pool scientist — their *true* "
                "traits, whether they `applied` / `survived`, and their observed "
                "1–5 scores (blank if they didn't apply). That's the "
                "traits→outcome table, so \"chances for people like me\" is a "
                "one-line group-by. **Fresh draw / mix:** the population's true "
                "traits. `true_*` columns are the hidden ground truth; `obs_*` "
                "are reviewer averages; `root_id` (single-root chains) joins "
                "across stages."
            )
            gen_key = f"csv_text::{d.name}::{d.n}"
            if st.button("Prepare CSV", key=f"prep_csv_{d.name}"):
                with st.spinner("Building CSV…"):
                    st.session_state[gen_key] = export_distribution_csv(wb, d.name)
            csv_text = st.session_state.get(gen_key)
            if csv_text:
                safe_file = "".join(
                    c if (c.isalnum() or c in "-_.") else "_" for c in d.name
                ) or "population"
                st.download_button(
                    "Download CSV", data=csv_text,
                    file_name=f"{safe_file}.csv", mime="text/csv",
                    key=f"dl_csv_{d.name}",
                )
                st.caption(f"{max(csv_text.count(chr(10)) - 1, 0):,} rows ready.")

        with st.expander("Lineage", expanded=True):
            st.code(lineage_text(wb, d.name))

        # Path-color toggle: available for any distribution that has at
        # least 2 distinct path labels (typically a Mix or its descendants).
        multi_path = (
            d.path_label is not None
            and len(set(d.path_label.tolist())) >= 2
        )
        toggle_cols = st.columns(2)
        shared_axis = toggle_cols[0].toggle(
            "Shared x-axis on Potential charts", value=True,
            help=("On: True Potential and Filter Mistakes share the parent "
                  "distribution's full Potential range, with cutoff lines "
                  "drawn on both. Off: each chart auto-fits its own data."),
        )
        color_by_path = toggle_cols[1].toggle(
            "Color by path through filters",
            value=False, disabled=not multi_path,
            help=("Available when a distribution has scientists from "
                  "multiple paths through the DAG (typically after a Mix). "
                  "When on, plots show per-path overlays instead of "
                  "survived/rejected coloring."),
        )

        if d.filter_artifacts is not None:
            fa = d.filter_artifacts
            st.subheader("Filter artifacts")
            facts = []
            facts.append(f"Pool entered with **{fa.pool_size_before:,}**.")
            if fa.self_selection_enabled:
                facts.append(
                    f"Self-selection: intended X = **{fa.self_selection_x_intended:.3f}**, "
                    f"realized X = **{fa.self_selection_x_realized:.3f}**, "
                    f"threshold = **{fa.self_selection_threshold:.3f}**, "
                    f"applied = **{fa.pool_size_after_self_selection:,}**."
                )
                if fa.equilibration_iterations > 0:
                    facts.append(
                        f"Equilibrated cushion ({fa.equilibration_iterations} bisect steps) "
                        f"to {'fill cutoff exactly.' if not fa.underfilled else 'maximum X=1; still under-filled.'}"
                    )
            facts.append(
                f"Reviewer filter: **{fa.n_reviewers}** reviewers, "
                f"**{fa.actual_assessments_per_reviewer}** assessments/reviewer, "
                f"survived **{fa.pool_size_after:,}**."
            )
            facts.append(
                f"Productivity dimension assessed: "
                f"**{'yes' if fa.productivity_dimension_used else 'no'}**.  "
                f"Resource awarded: **{fa.resource_awarded:g}**."
            )
            for f in facts:
                st.markdown("- " + f)

            parent_name = (
                d.origin.input_distribution_name
                if isinstance(d.origin, ApplyFilterOrigin) else None
            )
            parent = wb.distributions.get(parent_name) if parent_name else None

            x_range = None
            ideal_cutoff = None
            if shared_axis and parent is not None:
                pp = parent.potential
                x_range = (float(pp.min()), float(pp.max()))
                k = fa.pool_size_after
                n = len(pp)
                if 0 < k < n:
                    order = np.argsort(-pp)
                    ideal_cutoff = float(pp[order[k - 1]])
            ss_threshold = (
                fa.self_selection_threshold
                if shared_axis and fa.self_selection_enabled else None
            )

            st.subheader("Reviewer perception vs reality (overall)")
            fig = scatter_observed_vs_potential(d, parent, color_by_path=color_by_path)
            if fig is not None:
                st.pyplot(fig); plt.close("all")

            # Per-quality scatter — pick a quality to drill into.
            q_names = [c.name for c in wb.world.quality_configs]
            # Pull the filter used to produce this distribution, so we can
            # show its per-quality weights in the dropdown labels.
            used_filter = (
                wb.filters.get(d.origin.filter_name)
                if isinstance(d.origin, ApplyFilterOrigin) else None
            )
            if used_filter is not None and used_filter.quality_weights is not None:
                qw_for_label = list(used_filter.quality_weights)
            else:
                qw_for_label = [c.weight for c in wb.world.quality_configs]
            prod_w_for_label = (
                used_filter.productivity_weight if used_filter is not None else 1.0
            )
            options = list(q_names)
            if fa.productivity_dimension_used:
                options.append("Productivity")

            def _drilldown_label(name: str) -> str:
                if name == "Productivity":
                    return f"{name}  (weight {prod_w_for_label:.2f})"
                idx = q_names.index(name)
                return f"{name}  (weight {qw_for_label[idx]:.2f})"

            picked = st.selectbox(
                "Per-quality drill-down",
                options,
                index=0,
                format_func=_drilldown_label,
                help=("Plot true value vs reviewer-averaged 1–5 score for a "
                      "single quality. No cutoff lines (those are at the "
                      "overall-score level); quartile guides on both axes. "
                      "Weight in parentheses is from the filter that produced "
                      "this distribution."),
            )
            if picked == "Productivity":
                fig = scatter_productivity(d, parent, color_by_path=color_by_path)
            else:
                q_idx = q_names.index(picked)
                fig = scatter_per_quality(d, parent, q_idx, picked,
                                           color_by_path=color_by_path)
            if fig is not None:
                st.pyplot(fig); plt.close("all")

            st.subheader("True Potential")
            intermediate = None
            parent_for_overlay = None
            if parent is not None and fa.applied_mask is not None and fa.self_selection_enabled:
                intermediate = parent.potential[fa.applied_mask]
            if shared_axis and parent is not None:
                parent_for_overlay = parent.potential
            tp_stacked = st.toggle(
                "Stack True-Potential bars",
                value=False, key="tp_stacked",
                disabled=not (color_by_path and multi_path),
                help=("Only meaningful when path-coloring is on (paths are "
                      "disjoint). Stacking the pre-SS / post-SS / final "
                      "overlays would double-count nested subsets, so those "
                      "stay layered with transparency regardless."),
            )
            fig = hist_potential(
                d, intermediate=intermediate,
                parent_potential=parent_for_overlay,
                x_range=x_range,
                ideal_cutoff=ideal_cutoff,
                ss_threshold=ss_threshold,
                color_by_path=color_by_path,
                stacked=tp_stacked,
            )
            st.pyplot(fig); plt.close("all")

            st.subheader("Filter mistakes")
            fm_stacked = st.toggle(
                "Stack Filter-Mistakes bars", value=True, key="fm_stacked",
                help=("On (default): FP and FN sub-categories stack within "
                      "each side of the cutoff. Off: bars overlap with "
                      "transparency."),
            )
            fig = hist_filter_mistakes(d, parent, x_range=x_range,
                                        color_by_path=color_by_path,
                                        stacked=fm_stacked)
            if fig is not None:
                st.pyplot(fig); plt.close("all")
        else:
            tp_stacked = st.toggle(
                "Stack True-Potential bars",
                value=False, key="tp_stacked_basic",
                disabled=not (color_by_path and multi_path),
            )
            st.subheader("True Potential")
            x_range = (float(d.potential.min()), float(d.potential.max())) if shared_axis else None
            st.pyplot(hist_potential(d, x_range=x_range,
                                       color_by_path=color_by_path,
                                       stacked=tp_stacked))
            plt.close("all")

        with st.expander("Optional: Lotka log-log productivity", expanded=False):
            result = root_relative_productivity(wb, d.name)
            if result is None:
                st.info("Lotka view unavailable: this distribution descends "
                        "from a Mix (multiple roots), so root-relative "
                        "productivity isn't defined.")
            else:
                full_p, root = result
                st.pyplot(lotka_loglog(full_p, len(full_p), root))
                plt.close("all")
