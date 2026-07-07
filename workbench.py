"""Workbench: a session-level world (qualities, reviewer characteristics,
sigmoid, SS) plus libraries of named distributions and named filters,
connected by a DAG of operations. The world is shared by every artifact
in the workbench — that's what makes 'mix two distributions' coherent."""
from __future__ import annotations

import io
import pickle
from dataclasses import dataclass, field, replace
from typing import Optional, Union

import numpy as np

from sim import (
    ApplicationConfig,
    FilterConfig,
    QualityConfig,
    ReviewerAccuracyConfig,
    SelfSelectionConfig,
    SigmoidConfig,
    WorldConfig,
    compute_potential,
    default_world,
    derive_rng,
    make_qualities,
    make_self_assessment_accuracies,
    raw_potential_params,
    run_application,
)


WORKBENCH_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Origin variants — a distribution's provenance.
# ---------------------------------------------------------------------------


@dataclass
class FreshDrawOrigin:
    target_n: int
    actual_n: int
    seed: int


@dataclass
class ApplyFilterOrigin:
    input_distribution_name: str
    filter_name: str
    seed: int


@dataclass
class MixOrigin:
    distribution_names: list[str]
    weights: list[float]
    target_n: int
    seed: int


Origin = Union[FreshDrawOrigin, ApplyFilterOrigin, MixOrigin]


# ---------------------------------------------------------------------------
# Filter artifacts: byproducts of applying a filter, kept for visualization.
# ---------------------------------------------------------------------------


@dataclass
class FilterArtifacts:
    """Indexed against the *input* distribution (the parent), not the output."""
    pool_size_before: int
    pool_size_after_self_selection: int
    pool_size_after: int
    n_reviewers: int
    actual_assessments_per_reviewer: int
    observed_scores: np.ndarray            # length = pool_size_after_ss
    cutoff_score: float
    survived_mask: np.ndarray              # length = pool_size_before
    self_selection_enabled: bool
    self_selection_x_intended: float
    self_selection_x_realized: float
    self_selection_threshold: float
    applied_mask: np.ndarray               # length = pool_size_before
    productivity_dimension_used: bool
    resource_awarded: float
    equilibration_iterations: int
    underfilled: bool
    # Per-quality 1-5 scores averaged across the scientist's reviewers.
    # Shape: (n_applied, Nq). Indexed in applicant-only order, so to align
    # with the parent pool you'd use `parent[applied_mask][i] ↔ row i`.
    per_quality_observed: Optional[np.ndarray] = None
    # Same idea for productivity (when active in this filter).
    productivity_observed: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# NamedDistribution: per-scientist data + provenance. Configs come from
# workbench.world (no per-distribution copy).
# ---------------------------------------------------------------------------


@dataclass
class NamedDistribution:
    name: str
    qualities: np.ndarray              # (n, Nq)
    potential: np.ndarray              # (n,) in [0, 1]
    productivity: np.ndarray           # (n,)
    opportunity: np.ndarray            # (n,)
    self_assessment_accuracies: np.ndarray  # (n,)
    origin: Origin
    # Single-root tracking for the Lotka view; None when descended from a Mix.
    root_distribution_name: Optional[str]
    indices_in_root: Optional[np.ndarray]
    filter_artifacts: Optional[FilterArtifacts] = None
    # Per-scientist path label — for FreshDraw, it's just the dist's name
    # everywhere. After a Mix, each scientist's label inherits from the source
    # dist they were drawn from. Propagated through ApplyFilter via survived
    # mask. Used to color Explorer plots by ancestral path.
    path_label: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return self.potential.shape[0]


# ---------------------------------------------------------------------------
# Workbench: world + libraries + DAG (implicit in origin pointers).
# ---------------------------------------------------------------------------


@dataclass
class Workbench:
    format_version: int = WORKBENCH_FORMAT_VERSION
    world: WorldConfig = field(default_factory=default_world)
    distributions: dict[str, NamedDistribution] = field(default_factory=dict)
    filters: dict[str, FilterConfig] = field(default_factory=dict)

    def distribution_names(self) -> list[str]:
        return list(self.distributions.keys())

    def filter_names(self) -> list[str]:
        return list(self.filters.keys())

    def distribution_descendants(self, name: str) -> list[str]:
        out: set[str] = set()
        frontier = list(self._direct_distribution_children(name))
        while frontier:
            child = frontier.pop()
            if child in out:
                continue
            out.add(child)
            frontier.extend(self._direct_distribution_children(child))
        return sorted(out)

    def _direct_distribution_children(self, name: str) -> list[str]:
        out = []
        for dn, d in self.distributions.items():
            if dn == name:
                continue
            o = d.origin
            if isinstance(o, ApplyFilterOrigin) and o.input_distribution_name == name:
                out.append(dn)
            elif isinstance(o, MixOrigin) and name in o.distribution_names:
                out.append(dn)
        return out

    def filter_descendants(self, filter_name: str) -> list[str]:
        return sorted(
            dn for dn, d in self.distributions.items()
            if isinstance(d.origin, ApplyFilterOrigin)
            and d.origin.filter_name == filter_name
        )

    def can_remove_distribution(self, name: str) -> bool:
        return name in self.distributions and not self.distribution_descendants(name)

    def can_remove_filter(self, filter_name: str) -> bool:
        return filter_name in self.filters and not self.filter_descendants(filter_name)

    def add_distribution(self, dist: NamedDistribution) -> None:
        if dist.name in self.distributions:
            raise ValueError(f"Distribution '{dist.name}' already exists.")
        self.distributions[dist.name] = dist

    def add_filter(self, f: FilterConfig) -> None:
        if f.name in self.filters:
            raise ValueError(f"Filter '{f.name}' already exists.")
        self.filters[f.name] = f

    def remove_distribution(self, name: str) -> None:
        if not self.can_remove_distribution(name):
            raise ValueError(
                f"Cannot remove '{name}' — has descendants: "
                f"{self.distribution_descendants(name)}"
            )
        del self.distributions[name]

    def remove_filter(self, filter_name: str) -> None:
        if not self.can_remove_filter(filter_name):
            raise ValueError(
                f"Cannot remove '{filter_name}' — has descendants: "
                f"{self.filter_descendants(filter_name)}"
            )
        del self.filters[filter_name]

    def world_locked(self) -> bool:
        """True if there are any distributions or filters; we forbid editing
        the world's quality structure once anything depends on it."""
        return bool(self.distributions) or bool(self.filters)


# ---------------------------------------------------------------------------
# Operations.
# ---------------------------------------------------------------------------


def fresh_draw(
    workbench: Workbench,
    name: str,
    target_n: int,
    seed: int,
) -> NamedDistribution:
    """Draw a fresh population using the workbench's world config."""
    world = workbench.world
    rng = derive_rng(seed, "fresh_draw")
    qualities = make_qualities(target_n, world.quality_configs, rng)
    weights = np.array([c.weight for c in world.quality_configs])
    raw_mu, raw_sigma = raw_potential_params(world.quality_configs)
    potential = compute_potential(
        qualities, weights, raw_mu, raw_sigma, world.sigmoid_config.k,
    )
    sa_acc = make_self_assessment_accuracies(
        target_n,
        SelfSelectionConfig(
            enabled=True,
            accuracy_mean=world.ss_accuracy_mean,
            accuracy_concentration=world.ss_accuracy_concentration,
        ),
        rng,
    )
    return NamedDistribution(
        name=name,
        qualities=qualities,
        potential=potential,
        productivity=np.zeros(target_n),
        opportunity=np.zeros(target_n),
        self_assessment_accuracies=sa_acc,
        origin=FreshDrawOrigin(target_n=target_n, actual_n=target_n, seed=seed),
        root_distribution_name=name,
        indices_in_root=np.arange(target_n),
        filter_artifacts=None,
        path_label=np.array([name] * target_n, dtype=object),
    )


def _ss_apply_count(cur_p, sa_acc, cutoff, pot_params, rng_seed, cushion):
    n = cur_p.size
    pass_rate = cutoff / max(1, n)
    x = min(1.0, cushion * pass_rate)
    rng = derive_rng(rng_seed, "ss_probe")
    if x >= 1.0:
        return n, np.ones(n, dtype=bool), float("nan")
    threshold = float(np.quantile(cur_p, 1.0 - x))
    pot_mu, pot_sigma = pot_params
    noise = rng.normal(pot_mu, pot_sigma, size=n)
    perceived = sa_acc * cur_p + (1.0 - sa_acc) * noise
    applied = perceived >= threshold
    return int(applied.sum()), applied, threshold


def _equilibrate_cushion(cur_p, sa_acc, cutoff, pot_params, rng_seed, intended_cushion):
    """Bisect cushion upward (loosening SS threshold) until enough scientists
    apply to fill the cutoff. Caps at the cushion that pushes X to 1.0."""
    n = cur_p.size
    pass_rate = cutoff / max(1, n)
    n_intended, _, _ = _ss_apply_count(
        cur_p, sa_acc, cutoff, pot_params, rng_seed, intended_cushion,
    )
    if n_intended >= cutoff:
        return intended_cushion, 0, False
    cushion_for_x_one = (1.0 / pass_rate) if pass_rate > 0 else 1.0
    n_max, _, _ = _ss_apply_count(
        cur_p, sa_acc, cutoff, pot_params, rng_seed, cushion_for_x_one,
    )
    if n_max < cutoff:
        return cushion_for_x_one, 0, True
    lo, hi = intended_cushion, cushion_for_x_one
    iters = 0
    best = cushion_for_x_one
    for _ in range(20):
        iters += 1
        mid = (lo + hi) / 2.0
        n_mid, _, _ = _ss_apply_count(cur_p, sa_acc, cutoff, pot_params, rng_seed, mid)
        if n_mid >= cutoff:
            hi = mid
            best = mid
        else:
            lo = mid
    return best, iters, False


def apply_filter(
    workbench: Workbench,
    input_dist: NamedDistribution,
    filter_cfg: FilterConfig,
    output_name: str,
    seed: int,
) -> NamedDistribution:
    """Apply a filter (using workbench.world) to a distribution. Returns the
    post-filter survivor distribution with productivity/opportunity awards
    applied to survivors."""
    world = workbench.world
    sa_for_call = (
        input_dist.self_assessment_accuracies
        if filter_cfg.self_selection_enabled else None
    )
    cur_p = input_dist.potential
    pot_params = (
        (float(cur_p.mean()), float(cur_p.std()))
        if cur_p.size >= 2
        else (float(cur_p.mean()) if cur_p.size else 0.0, 1.0)
    )

    pass_rate = filter_cfg.application.cutoff / max(1, cur_p.size)
    intended_cushion = filter_cfg.cushion_factor
    intended_x = min(1.0, intended_cushion * pass_rate)

    realized_cushion = intended_cushion
    equilibration_iters = 0
    underfilled = False
    if (
        filter_cfg.self_selection_enabled
        and filter_cfg.equilibrate_selectivity
        and intended_x < 1.0
    ):
        realized_cushion, equilibration_iters, underfilled = _equilibrate_cushion(
            cur_p, input_dist.self_assessment_accuracies,
            filter_cfg.application.cutoff, pot_params, seed,
            intended_cushion,
        )

    rng = derive_rng(seed, "apply_filter")
    out = run_application(
        input_dist.qualities,
        input_dist.potential,
        input_dist.productivity,
        input_dist.opportunity,
        world,
        filter_cfg,
        sa_for_call,
        pot_params,
        rng,
        cushion_factor_override=realized_cushion,
    )
    survived = out["survived_mask"]
    survivor_count = int(survived.sum())

    new_productivity = input_dist.productivity[survived].copy()
    new_opportunity = input_dist.opportunity[survived].copy()
    new_productivity += filter_cfg.application.resource * input_dist.potential[survived]
    new_opportunity += filter_cfg.application.resource

    if input_dist.root_distribution_name is not None and input_dist.indices_in_root is not None:
        new_root = input_dist.root_distribution_name
        new_indices = input_dist.indices_in_root[survived].copy()
    else:
        new_root, new_indices = None, None

    artifacts = FilterArtifacts(
        pool_size_before=input_dist.n,
        pool_size_after_self_selection=int(out["applied_mask"].sum()),
        pool_size_after=survivor_count,
        n_reviewers=out["n_reviewers"],
        actual_assessments_per_reviewer=out["actual_per_reviewer"],
        observed_scores=out["observed_scores"],
        cutoff_score=out["cutoff_score"],
        survived_mask=survived,
        self_selection_enabled=out["self_selection_enabled"],
        self_selection_x_intended=intended_x,
        self_selection_x_realized=out["self_selection_x"],
        self_selection_threshold=out["self_selection_threshold"],
        applied_mask=out["applied_mask"],
        productivity_dimension_used=out["productivity_dimension_used"],
        resource_awarded=float(filter_cfg.application.resource),
        equilibration_iterations=equilibration_iters,
        underfilled=underfilled,
        per_quality_observed=out.get("per_quality_observed"),
        productivity_observed=out.get("productivity_observed"),
    )

    # Extend each scientist's path with this filter's name so distinct paths
    # remain distinguishable across subsequent Mix operations.
    if input_dist.path_label is not None:
        parent_paths = input_dist.path_label[survived]
        new_path_label = np.array(
            [f"{p} → {filter_cfg.name}" for p in parent_paths],
            dtype=object,
        )
    else:
        new_path_label = None
    return NamedDistribution(
        name=output_name,
        qualities=input_dist.qualities[survived].copy(),
        potential=input_dist.potential[survived].copy(),
        productivity=new_productivity,
        opportunity=new_opportunity,
        self_assessment_accuracies=input_dist.self_assessment_accuracies[survived].copy(),
        origin=ApplyFilterOrigin(
            input_distribution_name=input_dist.name,
            filter_name=filter_cfg.name,
            seed=seed,
        ),
        root_distribution_name=new_root,
        indices_in_root=new_indices,
        filter_artifacts=artifacts,
        path_label=new_path_label,
    )


def mix_distributions(
    dists: list[NamedDistribution],
    weights: list[float],
    output_name: str,
    target_n: int,
    seed: int,
) -> NamedDistribution:
    """Bootstrap-sample target_n agents from the inputs at the given weights."""
    if not dists:
        raise ValueError("Need at least one distribution to mix.")
    if len(dists) != len(weights):
        raise ValueError("Number of weights must match number of distributions.")
    weights_arr = np.asarray(weights, dtype=float)
    if (weights_arr < 0).any():
        raise ValueError("Weights must be non-negative.")
    s = weights_arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive value.")
    weights_arr = weights_arr / s
    Nq = dists[0].qualities.shape[1]
    for d in dists[1:]:
        if d.qualities.shape[1] != Nq:
            raise ValueError("All distributions must have the same Nq.")

    rng = derive_rng(seed, "mix")
    sources = rng.choice(len(dists), size=target_n, p=weights_arr)
    out_qualities = np.zeros((target_n, Nq))
    out_potential = np.zeros(target_n)
    out_productivity = np.zeros(target_n)
    out_opportunity = np.zeros(target_n)
    out_sa_acc = np.zeros(target_n)
    out_path_label = np.empty(target_n, dtype=object)
    for i, d in enumerate(dists):
        mask = sources == i
        if not mask.any():
            continue
        idx = rng.integers(0, d.n, size=int(mask.sum()))
        out_qualities[mask] = d.qualities[idx]
        out_potential[mask] = d.potential[idx]
        out_productivity[mask] = d.productivity[idx]
        out_opportunity[mask] = d.opportunity[idx]
        out_sa_acc[mask] = d.self_assessment_accuracies[idx]
        # Inherit path_label from source dist; falls back to source's name
        # if source had none (shouldn't happen for fresh-drawn dists but
        # belt-and-suspenders).
        if d.path_label is not None:
            out_path_label[mask] = d.path_label[idx]
        else:
            out_path_label[mask] = d.name

    return NamedDistribution(
        name=output_name,
        qualities=out_qualities,
        potential=out_potential,
        productivity=out_productivity,
        opportunity=out_opportunity,
        self_assessment_accuracies=out_sa_acc,
        origin=MixOrigin(
            distribution_names=[d.name for d in dists],
            weights=list(weights),
            target_n=target_n,
            seed=seed,
        ),
        root_distribution_name=None,
        indices_in_root=None,
        filter_artifacts=None,
        path_label=out_path_label,
    )


# ---------------------------------------------------------------------------
# Save / load / import.
# ---------------------------------------------------------------------------


def _restamp_workbench(wb) -> None:
    """Walk the workbench and reassign each dataclass instance's `__class__`
    to this module's currently-loaded class. Streamlit's hot-reload can leave
    in-memory objects pointing to a previous module-version's classes, which
    pickle then refuses to serialize ('not the same object'). Re-stamping
    fixes that without forcing the user to lose session state. The data is
    structurally identical between versions; only the class identity differs."""
    wb.__class__ = Workbench
    if wb.world is not None:
        wb.world.__class__ = WorldConfig
        for q in wb.world.quality_configs:
            q.__class__ = QualityConfig
        for ra in wb.world.reviewer_accuracy_configs:
            ra.__class__ = ReviewerAccuracyConfig
        wb.world.sigmoid_config.__class__ = SigmoidConfig
    for d in wb.distributions.values():
        d.__class__ = NamedDistribution
        if d.filter_artifacts is not None:
            d.filter_artifacts.__class__ = FilterArtifacts
        oname = type(d.origin).__name__
        if oname == "FreshDrawOrigin":
            d.origin.__class__ = FreshDrawOrigin
        elif oname == "ApplyFilterOrigin":
            d.origin.__class__ = ApplyFilterOrigin
        elif oname == "MixOrigin":
            d.origin.__class__ = MixOrigin
    for f in wb.filters.values():
        f.__class__ = FilterConfig
        f.application.__class__ = ApplicationConfig


def save_workbench(wb: Workbench, file_obj) -> None:
    try:
        pickle.dump(wb, file_obj)
    except (pickle.PicklingError, TypeError) as e:
        if "not the same object" in str(e):
            if hasattr(file_obj, "seek") and hasattr(file_obj, "truncate"):
                file_obj.seek(0)
                file_obj.truncate()
            _restamp_workbench(wb)
            pickle.dump(wb, file_obj)
        else:
            raise


def load_workbench(file_obj) -> Workbench:
    obj = pickle.load(file_obj)
    if not isinstance(obj, Workbench):
        raise ValueError(
            "Not a Workbench. (Pre-v2 .pkl files are no longer supported — "
            "the world-level config refactor changed the schema.)"
        )
    if obj.format_version != WORKBENCH_FORMAT_VERSION:
        raise ValueError(
            f"Workbench format v{obj.format_version} not supported "
            f"(expected v{WORKBENCH_FORMAT_VERSION})."
        )
    # Older pickles propagated path_label unchanged through apply_filter,
    # which left Mix descendants with all-identical labels. Re-derive labels
    # using the new "append filter name on apply" semantics. Idempotent.
    rebuild_path_labels(obj)
    return obj


def rebuild_path_labels(wb: Workbench) -> None:
    """Walk the DAG in topological order and recompute each distribution's
    path_label using the current semantics:
      - FreshDraw: every scientist labeled with the dist's own name
      - ApplyFilter: parent's label + ' → ' + filter name (per scientist)
      - Mix: each scientist inherits the source dist's label they came from
    Re-runs Mix's bootstrap sampling deterministically (same seed) so the
    source assignments match the original."""
    name_order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited or name not in wb.distributions:
            return
        d = wb.distributions[name]
        o = d.origin
        if isinstance(o, ApplyFilterOrigin):
            visit(o.input_distribution_name)
        elif isinstance(o, MixOrigin):
            for parent_name in o.distribution_names:
                visit(parent_name)
        visited.add(name)
        name_order.append(name)

    for n in wb.distribution_names():
        visit(n)

    for name in name_order:
        d = wb.distributions[name]
        o = d.origin
        if isinstance(o, FreshDrawOrigin):
            d.path_label = np.array([name] * d.n, dtype=object)
        elif isinstance(o, ApplyFilterOrigin):
            parent = wb.distributions.get(o.input_distribution_name)
            if parent is None or parent.path_label is None or d.filter_artifacts is None:
                d.path_label = np.array([name] * d.n, dtype=object)
                continue
            survived = d.filter_artifacts.survived_mask
            parent_paths = parent.path_label[survived]
            d.path_label = np.array(
                [f"{p} → {o.filter_name}" for p in parent_paths],
                dtype=object,
            )
        elif isinstance(o, MixOrigin):
            sources_dists = [wb.distributions[nn] for nn in o.distribution_names]
            weights = np.asarray(o.weights, dtype=float)
            s = weights.sum()
            if s <= 0:
                d.path_label = np.array([name] * d.n, dtype=object)
                continue
            weights = weights / s
            # Replicate mix sampling (same RNG state as the original mix call).
            rng = derive_rng(o.seed, "mix")
            sources = rng.choice(len(sources_dists), size=d.n, p=weights)
            new_label = np.empty(d.n, dtype=object)
            for i, src_d in enumerate(sources_dists):
                mask = sources == i
                if not mask.any():
                    continue
                idx = rng.integers(0, src_d.n, size=int(mask.sum()))
                if src_d.path_label is not None:
                    new_label[mask] = src_d.path_label[idx]
                else:
                    new_label[mask] = src_d.name
            d.path_label = new_label


def _suffix_to_avoid_collision(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    i = 2
    while f"{name} ({i})" in existing:
        i += 1
    return f"{name} ({i})"


def import_workbench_into(target: Workbench, source: Workbench) -> dict[str, str]:
    """Merge `source` into `target`. Worlds must match (same Nq and quality
    names) — otherwise raises. Returns rename map for collisions."""
    if not _worlds_compatible(target.world, source.world):
        raise ValueError(
            "Cannot import: world configs differ. The imported workbench has "
            f"qualities {[c.name for c in source.world.quality_configs]}; "
            f"this workbench has {[c.name for c in target.world.quality_configs]}."
        )
    rename_dist: dict[str, str] = {}
    rename_filter: dict[str, str] = {}
    existing_dists = set(target.distribution_names())
    existing_filters = set(target.filter_names())
    for name in source.distribution_names():
        new_name = _suffix_to_avoid_collision(name, existing_dists)
        rename_dist[name] = new_name
        existing_dists.add(new_name)
    for name in source.filter_names():
        new_name = _suffix_to_avoid_collision(name, existing_filters)
        rename_filter[name] = new_name
        existing_filters.add(new_name)
    for old_name, new_name in rename_filter.items():
        f = source.filters[old_name]
        target.filters[new_name] = replace(f, name=new_name)
    for old_name, new_name in rename_dist.items():
        d = source.distributions[old_name]
        new_origin = d.origin
        if isinstance(new_origin, ApplyFilterOrigin):
            new_origin = ApplyFilterOrigin(
                input_distribution_name=rename_dist.get(
                    new_origin.input_distribution_name,
                    new_origin.input_distribution_name,
                ),
                filter_name=rename_filter.get(
                    new_origin.filter_name, new_origin.filter_name
                ),
                seed=new_origin.seed,
            )
        elif isinstance(new_origin, MixOrigin):
            new_origin = MixOrigin(
                distribution_names=[
                    rename_dist.get(n, n) for n in new_origin.distribution_names
                ],
                weights=list(new_origin.weights),
                target_n=new_origin.target_n,
                seed=new_origin.seed,
            )
        new_root = (
            rename_dist.get(d.root_distribution_name, d.root_distribution_name)
            if d.root_distribution_name is not None else None
        )
        target.distributions[new_name] = replace(
            d, name=new_name, origin=new_origin,
            root_distribution_name=new_root,
        )
    out = {**{k: v for k, v in rename_dist.items() if k != v},
           **{k: v for k, v in rename_filter.items() if k != v}}
    return out


def _worlds_compatible(a: WorldConfig, b: WorldConfig) -> bool:
    """Worlds are compatible for import if they have the same number of
    qualities AND the same names. Other params can differ — but Nq must match
    so that distribution arrays line up."""
    if len(a.quality_configs) != len(b.quality_configs):
        return False
    return all(qa.name == qb.name
               for qa, qb in zip(a.quality_configs, b.quality_configs))


# ---------------------------------------------------------------------------
# Lineage helpers.
# ---------------------------------------------------------------------------


def lineage_text(wb: Workbench, dist_name: str) -> str:
    lines: list[str] = []
    visited: set[str] = set()

    def recurse(name: str, depth: int) -> None:
        if name in visited or depth > 20:
            lines.append("  " * depth + f"{name} (cycle/limit)")
            return
        visited.add(name)
        d = wb.distributions.get(name)
        if d is None:
            lines.append("  " * depth + f"{name} (missing)")
            return
        prefix = "  " * depth
        if isinstance(d.origin, FreshDrawOrigin):
            lines.append(f"{prefix}{name}  ←  fresh draw (n={d.origin.actual_n}, seed={d.origin.seed})")
        elif isinstance(d.origin, ApplyFilterOrigin):
            lines.append(
                f"{prefix}{name}  ←  apply '{d.origin.filter_name}' "
                f"to '{d.origin.input_distribution_name}' (seed={d.origin.seed})"
            )
            recurse(d.origin.input_distribution_name, depth + 1)
        elif isinstance(d.origin, MixOrigin):
            wts = ", ".join(f"{w:.2f}" for w in d.origin.weights)
            lines.append(
                f"{prefix}{name}  ←  mix [{wts}] of "
                f"{d.origin.distribution_names} (seed={d.origin.seed})"
            )
            for n in d.origin.distribution_names:
                recurse(n, depth + 1)
    recurse(dist_name, 0)
    return "\n".join(lines)


def root_relative_productivity(
    wb: Workbench, dist_name: str
) -> Optional[tuple[np.ndarray, str]]:
    """Returns (productivity_per_root_index, root_name) or None for Mix-rooted
    distributions. Supports the Lotka view."""
    d = wb.distributions.get(dist_name)
    if d is None or d.root_distribution_name is None or d.indices_in_root is None:
        return None
    root = wb.distributions.get(d.root_distribution_name)
    if root is None:
        return None
    full = np.zeros(root.n)
    full[d.indices_in_root] = d.productivity
    return full, d.root_distribution_name
