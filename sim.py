"""Meritocracy simulator core.

Scientists carry a vector of true hidden qualities (Normal-distributed).
Reviewers assess scientists with per-quality accuracy a in [0, 1]:
    observed_q = a_q * true_q + (1 - a_q) * noise_q,  noise_q ~ N(mu_q, sigma_q)
Applications: Nr reviewers each do `reads` assessments; per scientist, the
observed per-quality vector is summed over qualities to a per-reviewer raw
score, then averaged across that scientist's reviewers. Top-k by score pass.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


def derive_rng(seed: int, label: str) -> np.random.Generator:
    """Derive a deterministic RNG from (seed, label). Two calls with the
    same `seed` but different `label` produce uncorrelated streams. This
    prevents the bug where `fresh_draw(seed=0)` and `apply_filter(seed=0)`
    both call `default_rng(0)` and produce the same value sequence —
    causing the SS noise to mirror the potential draws."""
    h = hashlib.sha256(f"{label}:{int(seed)}".encode()).digest()
    derived = int.from_bytes(h[:8], "big")
    return np.random.default_rng(derived)


# ---------------------------------------------------------------------------
# Logit-Normal sampling for bounded accuracies in [0, 1].
#
# Replaces Beta because Beta with extreme means and high variance becomes
# edge-peaked or U-shaped — its mode migrates away from the configured mean.
# Logit-Normal is always unimodal at the configured mean, regardless of σ.
# ---------------------------------------------------------------------------

_sigma_logit_cache: dict[tuple[float, float], float] = {}


def _solve_sigma_logit(mean: float, target_std: float) -> float:
    """Find σ_logit such that std(sigmoid(N(logit(mean), σ_logit))) ≈ target_std.
    Uses bisection with a fixed-seed Monte Carlo eval (~8k samples). Cached
    so repeated calls with the same params are essentially free."""
    if target_std <= 1e-6 or mean <= 1e-6 or mean >= 1 - 1e-6:
        return 0.0
    key = (round(float(mean), 4), round(float(target_std), 4))
    cached = _sigma_logit_cache.get(key)
    if cached is not None:
        return cached
    mu_logit = float(np.log(mean / (1 - mean)))
    max_std = float(np.sqrt(mean * (1 - mean)) * 0.99)
    target = min(float(target_std), max_std)
    z = np.random.default_rng(42).standard_normal(8000)
    def std_at(s):
        return float(np.std(1.0 / (1.0 + np.exp(-(mu_logit + s * z)))))
    lo, hi = 1e-4, 8.0
    for _ in range(35):
        mid = (lo + hi) / 2
        if std_at(mid) < target:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2
    _sigma_logit_cache[key] = result
    return result


def _logit_normal_samples(rng, mean, target_std: float, n: int) -> np.ndarray:
    """n samples from a logit-Normal with the configured mean (= mode) and
    approximate target std. Always unimodal at `mean`, bounded in (0, 1)."""
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    if target_std <= 1e-6 or n <= 0:
        return np.full(n, mean)
    sigma_logit = _solve_sigma_logit(mean, target_std)
    if sigma_logit <= 1e-6:
        return np.full(n, mean)
    mu_logit = float(np.log(mean / (1 - mean)))
    z = rng.standard_normal(n)
    return 1.0 / (1.0 + np.exp(-(mu_logit + sigma_logit * z)))


def _logit_normal_per_reviewer(rng, base_mean: float, target_std: float,
                                shifted_means: np.ndarray) -> np.ndarray:
    """Logit-Normal samples where each reviewer has a different shifted mean
    (used for the judgment-correlation case). Same σ_logit for everyone — it's
    derived from the base mean's target_std and applied uniformly. Each
    reviewer's logit_mean is set to logit(their shifted_mean)."""
    if target_std <= 1e-6 or shifted_means.size == 0:
        return shifted_means
    sigma_logit = _solve_sigma_logit(base_mean, target_std)
    if sigma_logit <= 1e-6:
        return shifted_means
    clipped = np.clip(shifted_means, 1e-6, 1 - 1e-6)
    logit_means = np.log(clipped / (1 - clipped))
    z = rng.standard_normal(shifted_means.size)
    return 1.0 / (1.0 + np.exp(-(logit_means + sigma_logit * z)))


# ---------------------------------------------------------------------------
# Pile-based 1–5 scoring. Each reviewer calibrates their pile of applications
# so the scores follow Normal(pile_mean, pile_sigma) clipped to [1, 5]. This
# captures the empirical reality that reviewers spread their pile across the
# scale rather than reading raw perceived quality directly.
# ---------------------------------------------------------------------------

_NORMAL_LOOKUP_SIZE = 20000
_NORMAL_LOOKUP: Optional[np.ndarray] = None


def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard-Normal CDF via precomputed sorted sample.
    Pure-numpy (no scipy). Accuracy ≈ 0.001 over (0, 1)."""
    global _NORMAL_LOOKUP
    if _NORMAL_LOOKUP is None:
        _NORMAL_LOOKUP = np.sort(
            np.random.default_rng(7).standard_normal(_NORMAL_LOOKUP_SIZE)
        )
    p_clipped = np.clip(p, 0.5 / _NORMAL_LOOKUP_SIZE, 1 - 0.5 / _NORMAL_LOOKUP_SIZE)
    idx = np.clip((p_clipped * _NORMAL_LOOKUP_SIZE).astype(int),
                   0, _NORMAL_LOOKUP_SIZE - 1)
    return _NORMAL_LOOKUP[idx]


def calibrate_pile_to_int(
    perceived: np.ndarray,
    pile_mean: float = 3.8,
    pile_sigma: float = 1.0,
    max_score: int = 5,
) -> np.ndarray:
    """Map a vector of continuous perceived scores (one reviewer's pile on
    one quality) to integer 1..max_score via rank-based calibration to a
    target Normal(pile_mean, pile_sigma) distribution, clipped and rounded.
    Order-preserving by construction."""
    n = perceived.size
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([int(np.clip(round(pile_mean), 1, max_score))])
    ranks = np.argsort(np.argsort(perceived))
    quantiles = (ranks + 0.5) / n
    z = _inv_norm_cdf(quantiles)
    target = pile_mean + pile_sigma * z
    return np.clip(np.round(target), 1, max_score).astype(int)


# Backward-compat alias (older callers may import this name)
calibrate_pile_to_1_5 = calibrate_pile_to_int


@dataclass
class QualityConfig:
    name: str
    mu: float = 0.0
    sigma: float = 1.0
    weight: float = 1.0  # contribution to raw Potential
    biphasic: bool = False
    p_yes: float = 0.4
    delta: float = 1.0
    # When is_communication is True, this quality also modifies how reviewers
    # perceive the OTHER qualities. For a scientist with z_g = standardized
    # value of this quality, every other-quality noise term gets:
    #   noise ~ max(N(polish·z_g, σ_q), -2σ_q + protection·z_g)
    # So high-communication scientists get an upward bias and a floor against
    # unlucky low scores; low-communication get the opposite.
    is_communication: bool = False
    polish_strength: float = 0.0
    protection_strength: float = 0.0

    def marginal_mean(self) -> float:
        if self.biphasic:
            return (self.p_yes - 0.5) * self.delta
        return self.mu

    def marginal_std(self) -> float:
        if self.biphasic:
            return self.delta * float(np.sqrt(self.p_yes * (1.0 - self.p_yes)))
        return self.sigma


@dataclass
class ReviewerAccuracyConfig:
    """Configuration for how reviewers perceive a single quality.
    `mean` and `concentration` parameterize the per-reviewer accuracy
    distribution (logit-Normal, mode at `mean`). `pile_mean` and `pile_sigma`
    parameterize the target distribution reviewers project their pile onto
    when converting continuous perceptions to integer 1-5 scores."""
    mean: float = 0.7
    concentration: float = 8.0
    pile_mean: float = 3.8
    pile_sigma: float = 1.0


@dataclass
class ApplicationConfig:
    name: str
    reads: int = 3
    assessments_per_reviewer: int = 15  # target; actual may differ slightly
    cutoff: int = 100  # number of scientists passing
    qualities_assessed: Optional[np.ndarray] = None  # bool mask len Nq, None=all
    pool_size_for_x: Optional[int] = None  # denominator for X = 1.5*cutoff/n;
    # uses current pool size if None. Set on App1 when the initial pool was
    # oversampled to hit a target post-self-selection size.
    resource: float = 1.0  # amount of "resource" awarded to scientists who pass


@dataclass
class SigmoidConfig:
    """Steepness of the sigmoid that maps raw weighted-sum potential to [0,1].
    Larger k = sharper transition. Presets: gentle≈0.6, medium≈1.5, sharp≈3.5."""
    k: float = 1.5


@dataclass
class ProductivityConfig:
    """Productivity tracking + reviewer assessment of past performance.
    Each Application awards `resource_k` to passers; productivity[i] grows by
    resource_k * potential[i], opportunity[i] grows by resource_k.
    From App2 onward, reviewers assess perceived productivity, contributing
    score = scale * [(1-β) * (perceived/opportunity) + β * log(opportunity+1)].
    pedigree_weight β controls how much reviewers reward raw past resources
    vs the conversion rate; only matters when paths diverge across scientists
    (currently always uniform-by-cohort in sequential mode)."""
    enabled: bool = True
    reviewer_accuracy_mean: float = 0.7
    reviewer_accuracy_concentration: float = 8.0
    pedigree_weight: float = 0.0
    productivity_score_scale: float = 5.0


@dataclass
class SelfSelectionConfig:
    """Per-scientist self-assessment accuracy is drawn once from
    Beta(mean*concentration, (1-mean)*concentration) and stays fixed across
    Applications. Before each Application a noisy perceived Potential is
    computed, and a scientist applies only if perceived >= the (1-X) quantile
    of the actual pool's true Potential, where X = min(1, 1.5*pass_fraction)."""
    enabled: bool = False
    accuracy_mean: float = 0.30          # poor by default
    accuracy_concentration: float = 8.0


@dataclass
class WorldConfig:
    """Session-level definition of the quality space and how reviewers
    perceive it. Shared by every distribution and filter in the workbench;
    this is what makes 'mix two distributions' a coherent operation."""
    quality_configs: list[QualityConfig]
    sigmoid_config: SigmoidConfig
    # one entry per quality (same length as quality_configs):
    reviewer_accuracy_configs: list[ReviewerAccuracyConfig]
    judgment_correlations: list[float]   # corr(reviewer's q value, accuracy on q)
    weight_correlations: list[float]     # corr(reviewer's q value, weight on q)
    # Self-assessment by scientists themselves
    ss_accuracy_mean: float = 0.30
    ss_accuracy_concentration: float = 8.0
    # How well reviewers see prior productivity (one global pair)
    productivity_accuracy_mean: float = 0.65
    productivity_accuracy_concentration: float = 8.0
    productivity_pile_mean: float = 3.8
    productivity_pile_sigma: float = 1.0
    # Reviewers emit integer scores from 1 to max_score for each quality and
    # for the productivity dimension. Default 5 (NSF-like); 9 ≈ NIH-like.
    max_score: int = 5


@dataclass
class FilterConfig:
    """Per-filter knobs. quality_weights overrides the world's per-quality
    weights for THIS filter only — different competitions value different
    things. productivity_weight is the productivity dimension's slot in the
    same per-quality weighting (so productivity stops feeling 'special')."""
    name: str
    application: "ApplicationConfig"
    self_selection_enabled: bool = True
    cushion_factor: float = 1.5
    equilibrate_selectivity: bool = True
    quality_weights: Optional[list[float]] = None  # length Nq; None → world defaults
    productivity_weight: float = 1.0
    # β: 0 = reviewers value pure conversion-rate (productivity / opportunity);
    # 1 = reviewers value pure log(opportunity) — credit for having had the
    # opportunity at all, regardless of what was made of it.
    opportunity_credit: float = 0.0


def default_world() -> WorldConfig:
    """Literature-informed defaults: 4 qualities (Intelligence, Drive,
    Creativity, Communication). Reviewer accuracies in line with empirical
    IRR (~0.2-0.7). Communication has polish=0.3 and protection=0.5."""
    qcs = [
        QualityConfig(name="Intelligence", mu=0.0, sigma=1.0, weight=1.0),
        QualityConfig(name="Drive",        mu=0.0, sigma=1.0, weight=1.0),
        QualityConfig(name="Creativity",   mu=0.0, sigma=1.0, weight=1.0),
        QualityConfig(name="Communication", mu=0.0, sigma=1.0, weight=1.0,
                      is_communication=True,
                      polish_strength=0.3, protection_strength=0.5),
    ]
    racs = [
        ReviewerAccuracyConfig(mean=0.70, concentration=8.0),
        ReviewerAccuracyConfig(mean=0.55, concentration=8.0),
        ReviewerAccuracyConfig(mean=0.45, concentration=8.0),
        ReviewerAccuracyConfig(mean=0.80, concentration=8.0),
    ]
    return WorldConfig(
        quality_configs=qcs,
        sigmoid_config=SigmoidConfig(k=0.6),
        reviewer_accuracy_configs=racs,
        judgment_correlations=[0.0, 0.0, 0.0, 0.0],
        weight_correlations=[0.0, 0.0, 0.0, 0.0],
        ss_accuracy_mean=0.30,
        ss_accuracy_concentration=8.0,
        productivity_accuracy_mean=0.65,
        productivity_accuracy_concentration=8.0,
    )


@dataclass
class ApplicationResult:
    name: str
    pool_size_before: int                  # pre-self-selection
    pool_size_after_self_selection: int    # post-self-selection, pre-reviewer
    pool_size_after: int                   # post-everything
    n_reviewers: int
    actual_assessments_per_reviewer: int
    observed_scores: np.ndarray            # len pool_size_after_self_selection
    cutoff_score: float
    survived_mask: np.ndarray              # bool, len pool_size_before (full filter)
    self_selection_enabled: bool
    self_selection_x: float                # the fraction-of-top scientists self-target
    self_selection_threshold: float        # Potential threshold; nan if disabled
    applied_mask: np.ndarray               # bool, len pool_size_before
    true_potential_before: np.ndarray      # pre-self-selection
    true_potential_after: np.ndarray       # post-everything
    true_qualities_before: np.ndarray      # (pool_size_before, Nq)
    true_qualities_after: np.ndarray       # (pool_size_after, Nq)
    # productivity tracking
    productivity_before: np.ndarray        # (pool_size_before,) — pre this Application
    productivity_after: np.ndarray         # (pool_size_after,)
    opportunity_before: np.ndarray         # (pool_size_before,)
    opportunity_after: np.ndarray          # (pool_size_after,)
    productivity_dimension_used: bool      # True for App2+ (when opportunity > 0)
    resource_awarded: float                # this Application's resource (for reference)


def make_qualities(
    n: int,
    configs: list[QualityConfig],
    rng,
    source: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw n quality vectors. If `source` is given, sample n rows with
    replacement. Otherwise each quality is drawn either from N(μ, σ) or — if
    biphasic — from a two-point ±delta/2 distribution with prob p_yes."""
    if source is not None and source.size > 0:
        if source.ndim != 2 or source.shape[1] != len(configs):
            raise ValueError(
                f"Source qualities have shape {source.shape}; "
                f"expected (*, {len(configs)})."
            )
        idx = rng.integers(0, source.shape[0], size=n)
        return source[idx].copy()
    out = np.zeros((n, len(configs)))
    for i, c in enumerate(configs):
        if c.biphasic:
            yes = rng.random(n) < c.p_yes
            out[:, i] = np.where(yes, c.delta / 2.0, -c.delta / 2.0)
        else:
            out[:, i] = rng.normal(c.mu, c.sigma, size=n)
    return out


def make_reviewer_accuracies(
    n: int, configs: list[ReviewerAccuracyConfig], rng
) -> np.ndarray:
    """Uncorrelated per-quality accuracies. Logit-Normal: always unimodal
    at the configured mean."""
    Nq = len(configs)
    out = np.zeros((n, Nq))
    for q, c in enumerate(configs):
        target_std = float(np.sqrt(c.mean * (1 - c.mean) / max(c.concentration + 1, 1e-9)))
        out[:, q] = _logit_normal_samples(rng, c.mean, target_std, n)
    return out


# How much one σ of a reviewer's own quality value shifts their accuracy
# (when judgment_correlation = 1.0). Tunable later; 0.2 means a +1σ reviewer
# gets a +0.2 shift in their accuracy mean, modulo the correlation strength.
_JUDGMENT_SHIFT_SCALE = 0.2


def make_correlated_reviewer_accuracies(
    reviewer_qualities: np.ndarray,
    quality_configs: list[QualityConfig],
    accuracy_configs: list[ReviewerAccuracyConfig],
    judgment_correlations: list[float],
    rng,
) -> np.ndarray:
    """For each reviewer × quality, sample from logit-Normal whose mode is
    shifted by (corr · z_reviewer_q · 0.2). Default corrs = 0 → uncorrelated."""
    n_reviewers, Nq = reviewer_qualities.shape
    out = np.zeros((n_reviewers, Nq))
    for q in range(Nq):
        qcfg = quality_configs[q]
        acfg = accuracy_configs[q]
        corr = judgment_correlations[q]
        z = (reviewer_qualities[:, q] - qcfg.marginal_mean()) / max(qcfg.marginal_std(), 1e-12)
        shifted = np.clip(acfg.mean + corr * z * _JUDGMENT_SHIFT_SCALE, 0.001, 0.999)
        target_std = float(np.sqrt(acfg.mean * (1 - acfg.mean) / max(acfg.concentration + 1, 1e-9)))
        out[:, q] = _logit_normal_per_reviewer(rng, acfg.mean, target_std, shifted)
    return out


def compute_reviewer_weights(
    reviewer_qualities: np.ndarray,
    quality_configs: list[QualityConfig],
    base_weights: list[float],
    weight_correlations: list[float],
) -> np.ndarray:
    """Per-reviewer per-quality weights. weight = base + corr · z_reviewer_q,
    then normalized so each reviewer's total weight matches sum(base_weights).
    The correlation expresses *relative* preference among qualities — a
    high-Drive reviewer values Drive more *relative* to other qualities,
    not absolute score magnitude. Without normalization, weight_corr would
    just rescale every score by a per-reviewer factor (especially bad in
    the single-quality case where there's nothing to redistribute among)."""
    Nq = len(quality_configs)
    Nr = reviewer_qualities.shape[0]
    out = np.zeros((Nr, Nq))
    for q in range(Nq):
        qcfg = quality_configs[q]
        z = (reviewer_qualities[:, q] - qcfg.marginal_mean()) / max(qcfg.marginal_std(), 1e-12)
        out[:, q] = base_weights[q] + weight_correlations[q] * z

    if any(c != 0 for c in weight_correlations):
        target = float(np.sum(base_weights))
        if abs(target) > 1e-9:
            row_sums = out.sum(axis=1, keepdims=True)
            small = np.abs(row_sums) < 1e-9
            safe = np.where(small, 1.0, row_sums)
            out = out * (target / safe)
            if small.any():
                out[small.flatten()] = np.array(base_weights)
    return out


def raw_potential_params(quality_configs: list[QualityConfig]) -> tuple[float, float]:
    """Analytical mean and std of the raw weighted sum, before sigmoid.
    Uses each quality's marginal mean/std (handles biphasic + Normal alike)."""
    weights = np.array([c.weight for c in quality_configs])
    mus = np.array([c.marginal_mean() for c in quality_configs])
    sigmas = np.array([c.marginal_std() for c in quality_configs])
    raw_mu = float((weights * mus).sum())
    raw_sigma = float(np.sqrt(((weights * sigmas) ** 2).sum()))
    return raw_mu, raw_sigma


def compute_potential(
    qualities: np.ndarray,
    weights: np.ndarray,
    raw_mu: float,
    raw_sigma: float,
    sigmoid_k: float,
) -> np.ndarray:
    """Map raw weighted-sum potential to [0,1] via a sigmoid centered at the
    raw mean (raw_mu) and scaled by raw std (raw_sigma). Larger sigmoid_k =
    sharper transition. The resulting Potential is the scientist's "ability
    to make the most of opportunity" — it sets the conversion rate from
    resource into productivity."""
    raw = qualities @ weights
    z = sigmoid_k * (raw - raw_mu) / max(raw_sigma, 1e-12)
    # numerically stable sigmoid
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def make_self_assessment_accuracies(
    n: int,
    cfg: SelfSelectionConfig,
    rng,
    source: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw n self-assessment accuracies. Bootstraps from `source` when given.
    Otherwise samples from logit-Normal — always unimodal at `accuracy_mean`."""
    if source is not None and source.size > 0:
        idx = rng.integers(0, source.size, size=n)
        return source[idx].copy()
    target_std = float(np.sqrt(
        cfg.accuracy_mean * (1 - cfg.accuracy_mean)
        / max(cfg.accuracy_concentration + 1, 1e-9)
    ))
    return _logit_normal_samples(rng, cfg.accuracy_mean, target_std, n)


def predict_apply_fraction(
    ss_config: SelfSelectionConfig,
    x: float,
    sigmoid_k: float,
    n_samples: int = 20000,
) -> float:
    """Monte-Carlo estimate of the fraction of a sigmoid-mapped Normal pool
    that will self-select with X = `x`. Sigmoid-aware because the [0,1]
    transform changes the perceived-vs-threshold geometry. Stable seed."""
    if not ss_config.enabled or x >= 1.0:
        return 1.0
    rng_pred = np.random.default_rng(12345)
    target_std = float(np.sqrt(
        ss_config.accuracy_mean * (1 - ss_config.accuracy_mean)
        / max(ss_config.accuracy_concentration + 1, 1e-9)
    ))
    a = _logit_normal_samples(
        rng_pred, ss_config.accuracy_mean, target_std, n_samples,
    )
    raw = rng_pred.standard_normal(n_samples)
    true = 1.0 / (1.0 + np.exp(-sigmoid_k * raw))   # sigmoid in std units
    pool_mu = float(true.mean())
    pool_sigma = float(true.std())
    noise = rng_pred.normal(pool_mu, pool_sigma, size=n_samples)
    perceived = a * true + (1.0 - a) * noise
    threshold = np.quantile(true, 1.0 - x)
    return float((perceived >= threshold).mean())


def predict_initial_pop_for_target(
    target_post_ss: int,
    ss_config: SelfSelectionConfig,
    app1_cutoff: int,
    sigmoid_k: float,
) -> int:
    """Estimate initial pop so ~target_post_ss scientists apply to App1.
    No oversampling when SS is off or when X>=1 (everyone applies)."""
    if not ss_config.enabled:
        return target_post_ss
    x = min(1.0, 1.5 * app1_cutoff / max(1, target_post_ss))
    p = predict_apply_fraction(ss_config, x, sigmoid_k)
    if p <= 1e-3:
        return max(target_post_ss, app1_cutoff) * 1000  # extreme; user will notice
    return max(target_post_ss, int(round(target_post_ss / p)))


def _pick_nr(pool: int, reads: int, target_per_reviewer: int) -> int:
    """Pick number of reviewers so each does ~target_per_reviewer assessments.
    Nr >= reads (each scientist needs `reads` distinct reviewers). assign_reviews
    handles uneven loads when pool*reads is not a clean multiple of Nr."""
    if pool <= 0:
        return reads
    total = pool * reads
    return max(reads, min(total, round(total / max(1, target_per_reviewer))))


def assign_reviews(
    n_scientists: int, n_reviewers: int, reads: int, rng
) -> np.ndarray:
    """Each scientist gets `reads` distinct reviewers; reviewers' loads are
    balanced to within 1 (some get floor, some get ceil of the average).
    Fast vectorized shuffle-with-swap-fixup for typical sizes; greedy
    fallback for tiny / dense pools where shuffle dupes are too common."""
    total = n_scientists * reads
    if reads > n_reviewers:
        raise ValueError(f"reads ({reads}) > n_reviewers ({n_reviewers}); impossible")
    base = total // n_reviewers
    extra = total - base * n_reviewers   # this many reviewers carry one extra

    # Heuristic: when each reviewer has only a few slots relative to reads,
    # the shuffle approach has too many duplicates to fix. Use greedy then.
    avg_per = total / max(1, n_reviewers)
    if avg_per < 2.5 or n_scientists < 50:
        return _assign_greedy(n_scientists, n_reviewers, reads, base, extra, rng)
    try:
        return _assign_shuffle(n_scientists, n_reviewers, reads, base, extra, rng)
    except RuntimeError:
        return _assign_greedy(n_scientists, n_reviewers, reads, base, extra, rng)


def _build_slots(n_reviewers: int, base: int, extra: int, rng) -> np.ndarray:
    """Return a flat array containing reviewer ids: `extra` reviewers
    appear (base+1) times, the rest appear `base` times. Total length =
    n_reviewers*base + extra."""
    counts = np.full(n_reviewers, base, dtype=np.int64)
    if extra > 0:
        chosen = rng.choice(n_reviewers, size=extra, replace=False)
        counts[chosen] += 1
    return np.repeat(np.arange(n_reviewers), counts)


def _assign_shuffle(n_scientists, n_reviewers, reads, base, extra, rng):
    """Shuffle the slot array, reshape to (n_scientists, reads), then fix
    within-row duplicates by swapping cells with random other rows."""
    slots = _build_slots(n_reviewers, base, extra, rng)
    rng.shuffle(slots)
    assignment = slots.reshape(n_scientists, reads).copy()

    for outer in range(50):
        # Vectorized duplicate row detection
        sorted_rows = np.sort(assignment, axis=1)
        has_dupe = (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any(axis=1)
        dupe_rows = np.where(has_dupe)[0]
        if dupe_rows.size == 0:
            return assignment
        # Fix each dupe row by swapping a duplicated cell with a cell in
        # another (clean) row.
        for i in dupe_rows.tolist():
            row = assignment[i]
            seen: dict[int, int] = {}
            dup_pos = -1
            for j in range(reads):
                v = int(row[j])
                if v in seen:
                    dup_pos = j
                    break
                seen[v] = j
            if dup_pos < 0:
                continue
            dup_val = int(row[dup_pos])
            row_i_set = {int(x) for x in row}
            for _try in range(500):
                k = int(rng.integers(n_scientists))
                if k == i:
                    continue
                l = int(rng.integers(reads))
                v = int(assignment[k, l])
                if v == dup_val or v in row_i_set:
                    continue
                row_k_others = {int(x) for x in assignment[k]} - {v}
                if dup_val in row_k_others:
                    continue
                assignment[i, dup_pos], assignment[k, l] = (
                    assignment[k, l], assignment[i, dup_pos]
                )
                break
    raise RuntimeError("shuffle-fixup couldn't resolve duplicates")


def _assign_greedy(n_scientists, n_reviewers, reads, base, extra, rng):
    """Process scientists in random order, picking the `reads` reviewers
    with the most remaining capacity each time. Slow but always works."""
    capacity = np.full(n_reviewers, base, dtype=np.int64)
    if extra > 0:
        chosen = rng.choice(n_reviewers, size=extra, replace=False)
        capacity[chosen] += 1
    assignment = np.empty((n_scientists, reads), dtype=np.int64)
    order = np.arange(n_scientists)
    rng.shuffle(order)
    for s in order:
        tiebreak = rng.random(n_reviewers)
        keys = np.lexsort((tiebreak, -capacity))
        chosen = keys[:reads]
        if int(capacity[chosen].min()) <= 0:
            raise RuntimeError(
                f"Greedy exhausted capacity at scientist {s} — "
                f"shouldn't happen for valid inputs."
            )
        chosen_perm = chosen.copy()
        rng.shuffle(chosen_perm)
        assignment[s] = chosen_perm
        capacity[chosen] -= 1
    return assignment


def _self_selection_step(
    true_potential: np.ndarray,
    self_assessment_accuracies: np.ndarray,
    cutoff: int,
    pot_params: tuple[float, float],
    rng,
    pool_size_for_x: Optional[int] = None,
    cushion_factor: float = 1.5,
) -> tuple[np.ndarray, float, float]:
    """Returns (applied_mask, x_fraction, threshold_potential)."""
    n = len(true_potential)
    denom = pool_size_for_x if pool_size_for_x is not None else n
    x_frac = min(1.0, cushion_factor * cutoff / max(1, denom))
    if x_frac >= 1.0:
        # everyone applies
        return np.ones(n, dtype=bool), x_frac, float("nan")
    threshold = float(np.quantile(true_potential, 1.0 - x_frac))
    pot_mu, pot_sigma = pot_params
    noise = rng.normal(pot_mu, pot_sigma, size=n)
    perceived = (
        self_assessment_accuracies * true_potential
        + (1.0 - self_assessment_accuracies) * noise
    )
    applied_mask = perceived >= threshold
    if not applied_mask.any():
        # degenerate fallback: at least admit the highest perceived
        applied_mask = np.zeros(n, dtype=bool)
        applied_mask[int(np.argmax(perceived))] = True
    return applied_mask, x_frac, threshold


def run_application(
    qualities: np.ndarray,
    true_potential: np.ndarray,
    productivity: np.ndarray,
    opportunity: np.ndarray,
    world: WorldConfig,
    filter_cfg: FilterConfig,
    self_assessment_accuracies: Optional[np.ndarray],
    pot_params: tuple[float, float],
    rng,
    cushion_factor_override: Optional[float] = None,
) -> dict:
    """Run one Application: optional self-selection pre-filter, then reviewer
    filter on the applied pool. Reviewers have identities (own quality
    vectors); accuracy and weight per quality may be correlated with their
    own value of that quality. Communication-typed qualities, if any, modify
    perception of the other qualities (polish bias + protective floor)."""
    n_scientists, Nq = qualities.shape
    cfg_q = world.quality_configs
    if Nq != len(cfg_q):
        raise ValueError(
            f"Distribution Nq ({Nq}) doesn't match world Nq ({len(cfg_q)})."
        )
    cushion = cushion_factor_override if cushion_factor_override is not None else filter_cfg.cushion_factor
    app_config = filter_cfg.application

    # ---- self-selection pre-filter ----
    if self_assessment_accuracies is not None:
        applied_mask, x_frac, ss_threshold = _self_selection_step(
            true_potential, self_assessment_accuracies,
            app_config.cutoff, pot_params, rng,
            pool_size_for_x=app_config.pool_size_for_x,
            cushion_factor=cushion,
        )
        ss_enabled = True
    else:
        applied_mask = np.ones(n_scientists, dtype=bool)
        x_frac = 1.0
        ss_threshold = float("nan")
        ss_enabled = False

    applied_idx = np.where(applied_mask)[0]
    n_applied = applied_idx.size
    applied_qualities = qualities[applied_mask]
    applied_productivity = productivity[applied_mask]
    applied_opportunity = opportunity[applied_mask]

    # ---- reviewer pool: drawn fresh from the world's quality distribution ----
    Nr = _pick_nr(n_applied, app_config.reads, app_config.assessments_per_reviewer)
    actual_per_reviewer = (n_applied * app_config.reads) // max(1, Nr)

    reviewer_qualities = make_qualities(Nr, cfg_q, rng)
    reviewer_accuracies = make_correlated_reviewer_accuracies(
        reviewer_qualities, cfg_q,
        world.reviewer_accuracy_configs,
        world.judgment_correlations, rng,
    )
    base_weights = (
        list(filter_cfg.quality_weights)
        if filter_cfg.quality_weights is not None
        else [c.weight for c in cfg_q]
    )
    # NOTE: weight_correlations is currently disabled — every reviewer uses
    # the same `base_weights`. Kept on WorldConfig for future re-enablement.
    # reviewer_weights = compute_reviewer_weights(
    #     reviewer_qualities, cfg_q, base_weights, world.weight_correlations,
    # )
    assignment = assign_reviews(n_applied, Nr, app_config.reads, rng)

    quality_mus = np.array([c.marginal_mean() for c in cfg_q])
    quality_sigmas = np.array([c.marginal_std() for c in cfg_q])

    # ---- communication effect setup (per applicant) ----
    comm_idx = next((i for i, c in enumerate(cfg_q) if c.is_communication), None)
    if comm_idx is not None:
        comm_cfg = cfg_q[comm_idx]
        comm_mu = comm_cfg.marginal_mean()
        comm_sigma = max(comm_cfg.marginal_std(), 1e-12)
        z_comm = (applied_qualities[:, comm_idx] - comm_mu) / comm_sigma
        polish = comm_cfg.polish_strength
        protection = comm_cfg.protection_strength
    else:
        z_comm = np.zeros(n_applied)
        polish, protection = 0.0, 0.0

    # ---- productivity dimension setup ----
    # Require weight > 0 so a filter that explicitly sets productivity weight
    # to zero genuinely turns the dimension off (no scoring, no per-quality
    # drill-down in the UI), not just zeros out its contribution post-hoc.
    productivity_active = (
        n_applied > 0
        and float(applied_opportunity.max(initial=0.0)) > 0.0
        and float(filter_cfg.productivity_weight) > 0.0
    )
    if productivity_active:
        prod_target_std = float(np.sqrt(
            world.productivity_accuracy_mean
            * (1 - world.productivity_accuracy_mean)
            / max(world.productivity_accuracy_concentration + 1, 1e-9)
        ))
        reviewer_prod_acc = _logit_normal_samples(
            rng, world.productivity_accuracy_mean, prod_target_std, Nr,
        )
        prod_pool_mu = float(applied_productivity.mean())
        prod_pool_sigma = (
            max(float(applied_productivity.std()), 1e-6)
            if n_applied >= 2 else 1.0
        )
        beta_w = filter_cfg.opportunity_credit
        prod_w = filter_cfg.productivity_weight

    # ---- Step 1: continuous perceived observations per (scientist, read, q) ----
    continuous_obs = np.zeros((n_applied, app_config.reads, Nq))
    continuous_prod = np.zeros((n_applied, app_config.reads)) if productivity_active else None
    for r in range(app_config.reads):
        rev_idx = assignment[:, r]
        accs = reviewer_accuracies[rev_idx]
        noise = np.zeros((n_applied, Nq))
        for q in range(Nq):
            base_noise = rng.normal(0.0, quality_sigmas[q], size=n_applied)
            if q != comm_idx:
                bias = polish * z_comm
                floor = -2.0 * quality_sigmas[q] + protection * z_comm
                effective = np.maximum(base_noise, floor) + bias
                noise[:, q] = quality_mus[q] + effective
            else:
                noise[:, q] = quality_mus[q] + base_noise
        continuous_obs[:, r, :] = accs * applied_qualities + (1.0 - accs) * noise

        if productivity_active:
            a_prod = reviewer_prod_acc[rev_idx]
            prod_noise = rng.normal(prod_pool_mu, prod_pool_sigma, size=n_applied)
            perceived_prod = a_prod * applied_productivity + (1.0 - a_prod) * prod_noise
            opp_present = applied_opportunity > 0
            ratio = np.zeros_like(applied_opportunity)
            ratio[opp_present] = (
                perceived_prod[opp_present] / applied_opportunity[opp_present]
            )
            pedigree = np.log(applied_opportunity + 1.0)
            continuous_prod[:, r] = (1.0 - beta_w) * ratio + beta_w * pedigree

    # ---- Step 2: pile-based 1-5 calibration per reviewer per quality ----
    # Build reviewer → list of (scientist_idx, read_idx) entries
    pile_members: list[list[tuple[int, int]]] = [[] for _ in range(Nr)]
    for s in range(n_applied):
        for r in range(app_config.reads):
            pile_members[assignment[s, r]].append((s, r))

    score_1to5 = np.zeros((n_applied, app_config.reads, Nq), dtype=np.int64)
    # Float dtype so we can store the exact (un-rounded) pile_mean for opp=0
    # applicants — keeping reviewer-emitted scores as integers but allowing
    # the neutral default to be the actual mean (e.g., 3.8) rather than its
    # rounded version (4), which would otherwise introduce a small upward bias.
    score_1to5_prod = (
        np.zeros((n_applied, app_config.reads), dtype=np.float64)
        if productivity_active else None
    )
    for rev_id in range(Nr):
        pile = pile_members[rev_id]
        if not pile:
            continue
        s_idx = np.array([p[0] for p in pile])
        r_idx = np.array([p[1] for p in pile])
        # Per quality
        for q in range(Nq):
            pile_obs = continuous_obs[s_idx, r_idx, q]
            cal = calibrate_pile_to_int(
                pile_obs,
                pile_mean=world.reviewer_accuracy_configs[q].pile_mean,
                pile_sigma=world.reviewer_accuracy_configs[q].pile_sigma,
                max_score=world.max_score,
            )
            score_1to5[s_idx, r_idx, q] = cal
        # Productivity dimension. Scientists with opportunity = 0 have
        # nothing for the reviewer to assess — give them the neutral
        # rounded pile_mean instead of letting them tie at zero and rank
        # bottom-of-pile (which would penalize them for never having had a
        # chance, with the specific rank determined by arbitrary tie-break
        # order). Calibrate only the opp>0 sub-pile via rank-based mapping.
        if productivity_active:
            pile_opp = applied_opportunity[s_idx]
            opp_present = pile_opp > 0
            # Neutral default for opp=0 applicants: the EXACT pile_mean
            # (no rounding). This avoids the small upward bias that would
            # come from rounding 3.8 → 4. Clipped to the valid score range.
            neutral_exact = float(np.clip(
                world.productivity_pile_mean, 1.0, world.max_score
            ))
            if opp_present.all():
                pile_prod = continuous_prod[s_idx, r_idx]
                cal = calibrate_pile_to_int(
                    pile_prod,
                    pile_mean=world.productivity_pile_mean,
                    pile_sigma=world.productivity_pile_sigma,
                    max_score=world.max_score,
                )
                score_1to5_prod[s_idx, r_idx] = cal
            elif opp_present.any():
                # Mix of opp>0 and opp=0 in this pile — calibrate only the
                # opp>0 subset, assign exact neutral mean to the rest.
                sub_s = s_idx[opp_present]
                sub_r = r_idx[opp_present]
                sub_prod = continuous_prod[sub_s, sub_r]
                cal_sub = calibrate_pile_to_int(
                    sub_prod,
                    pile_mean=world.productivity_pile_mean,
                    pile_sigma=world.productivity_pile_sigma,
                    max_score=world.max_score,
                )
                score_1to5_prod[sub_s, sub_r] = cal_sub
                no_opp = ~opp_present
                score_1to5_prod[s_idx[no_opp], r_idx[no_opp]] = neutral_exact
            else:
                # Whole pile has no opportunity — everyone gets neutral.
                score_1to5_prod[s_idx, r_idx] = neutral_exact

    # ---- Step 3: weighted average per (scientist, read), then mean across reads ----
    # Productivity counts toward a scientist's score only when it was actually
    # assessable — i.e. they had prior opportunity. For a no-track-record
    # applicant (opportunity = 0) the productivity dimension is dropped from
    # BOTH the numerator and the denominator, so the dimensions reviewers *can*
    # judge carry full weight (a per-scientist denominator). Previously every
    # applicant shared one constant denominator and opp=0 applicants had a
    # neutral productivity score (pile_mean) folded in, which anchored their
    # composite toward that constant and pulled it toward the mean; dropping the
    # term removes that pull. Pools where everyone has opportunity are
    # unaffected (the denominator is constant there, as before).
    weights = np.array(base_weights, dtype=float)
    quality_weight_sum = float(weights.sum())
    if productivity_active:
        prod_assessed = (applied_opportunity > 0).astype(float)  # (n_applied,)
        denom = quality_weight_sum + float(prod_w) * prod_assessed
    else:
        denom = np.full(n_applied, quality_weight_sum, dtype=float)
    denom = np.where(denom <= 0.0, 1.0, denom)  # avoid div by zero

    per_reviewer_scores = np.zeros((n_applied, app_config.reads))
    for r in range(app_config.reads):
        weighted_q_sum = (score_1to5[:, r, :] * weights[None, :]).sum(axis=1)
        if productivity_active:
            weighted_q_sum = (weighted_q_sum
                              + prod_w * score_1to5_prod[:, r] * prod_assessed)
        per_reviewer_scores[:, r] = weighted_q_sum / denom

    final_scores = per_reviewer_scores.mean(axis=1)

    # Per-quality observed scores (averaged across the scientist's reviewers).
    per_quality_observed = score_1to5.mean(axis=1)  # (n_applied, Nq), 1-5 floats
    productivity_observed = (
        score_1to5_prod.astype(float).mean(axis=1) if productivity_active else None
    )

    k = min(app_config.cutoff, n_applied)
    if k >= n_applied:
        survived_among_applied = np.ones(n_applied, dtype=bool)
        cutoff_score = float(final_scores.min()) if n_applied > 0 else float("nan")
    else:
        order = np.lexsort((rng.random(n_applied), -final_scores))
        survived_among_applied = np.zeros(n_applied, dtype=bool)
        survived_among_applied[order[:k]] = True
        cutoff_score = float(final_scores[order[k - 1]])

    full_survived = np.zeros(n_scientists, dtype=bool)
    full_survived[applied_idx[survived_among_applied]] = True

    return {
        "applied_mask": applied_mask,
        "survived_mask": full_survived,
        "observed_scores": final_scores,
        "cutoff_score": cutoff_score,
        "self_selection_enabled": ss_enabled,
        "self_selection_x": x_frac,
        "self_selection_threshold": ss_threshold,
        "n_reviewers": Nr,
        "actual_per_reviewer": actual_per_reviewer,
        "productivity_dimension_used": productivity_active,
        "per_quality_observed": per_quality_observed,
        "productivity_observed": productivity_observed,
    }


def _legacy_run_simulation_DELETED_DO_NOT_USE(
    population_size: int,
    quality_configs: list[QualityConfig],
    reviewer_accuracy_configs: list[ReviewerAccuracyConfig],
    applications: list[ApplicationConfig],
    self_selection_config: Optional[SelfSelectionConfig] = None,
    productivity_config: Optional[ProductivityConfig] = None,
    sigmoid_config: Optional[SigmoidConfig] = None,
    seed: int = 0,
    qualities_source: Optional[np.ndarray] = None,
    accuracies_source: Optional[np.ndarray] = None,
) -> tuple[list[ApplicationResult], np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Returns (results, initial_qualities, initial_potential[in 0..1],
    self_assessment_accuracies, full_productivity, full_opportunity).
    `full_productivity` and `full_opportunity` are length actual_initial_pop
    arrays carrying the final productivity/opportunity for every initial
    scientist (zero for those who never passed any filter; cumulative for
    those who did, frozen at their dropout point). This lets us see the
    full Lotka-style output distribution."""
    rng = np.random.default_rng(seed)
    if productivity_config is None:
        productivity_config = ProductivityConfig()
    if sigmoid_config is None:
        sigmoid_config = SigmoidConfig()

    qualities = make_qualities(
        population_size, quality_configs, rng, source=qualities_source,
    )
    weights = np.array([c.weight for c in quality_configs])
    raw_mu, raw_sigma = raw_potential_params(quality_configs)
    potential = compute_potential(
        qualities, weights, raw_mu, raw_sigma, sigmoid_config.k,
    )

    if self_selection_config is not None and self_selection_config.enabled:
        sa_acc = make_self_assessment_accuracies(
            population_size, self_selection_config, rng,
            source=accuracies_source,
        )
    else:
        sa_acc = None

    # Track productivity / opportunity for the FULL initial population.
    # full_productivity[i] is scientist i's accumulated productivity at the
    # most recent time they were in the pool — frozen when they drop out.
    # cur_to_full maps current pool position k -> original index.
    actual_init_pop = qualities.shape[0]
    full_productivity = np.zeros(actual_init_pop)
    full_opportunity = np.zeros(actual_init_pop)
    cur_to_full = np.arange(actual_init_pop)

    results: list[ApplicationResult] = []
    cur_q = qualities
    cur_p = potential
    cur_sa = sa_acc

    for app in applications:
        if cur_p.size >= 2:
            cur_pot_params = (float(cur_p.mean()), float(cur_p.std()))
        else:
            cur_pot_params = (float(cur_p.mean()) if cur_p.size else 0.0, 1.0)

        # Read productivity/opportunity views into full arrays for run_application.
        cur_productivity = full_productivity[cur_to_full].copy()
        cur_opportunity = full_opportunity[cur_to_full].copy()

        out = run_application(
            cur_q, cur_p, cur_productivity, cur_opportunity,
            quality_configs, reviewer_accuracy_configs, productivity_config,
            app, cur_sa, cur_pot_params, rng,
        )
        survived = out["survived_mask"]

        # Snapshot before awarding this round's resource (= what reviewers saw).
        prod_before = cur_productivity.copy()
        opp_before = cur_opportunity.copy()

        # Award to survivors via the full arrays.
        survivor_full_idx = cur_to_full[survived]
        full_productivity[survivor_full_idx] += app.resource * cur_p[survived]
        full_opportunity[survivor_full_idx] += app.resource

        results.append(ApplicationResult(
            name=app.name,
            pool_size_before=cur_q.shape[0],
            pool_size_after_self_selection=int(out["applied_mask"].sum()),
            pool_size_after=int(survived.sum()),
            n_reviewers=out["n_reviewers"],
            actual_assessments_per_reviewer=out["actual_per_reviewer"],
            observed_scores=out["observed_scores"],
            cutoff_score=out["cutoff_score"],
            survived_mask=survived,
            self_selection_enabled=out["self_selection_enabled"],
            self_selection_x=out["self_selection_x"],
            self_selection_threshold=out["self_selection_threshold"],
            applied_mask=out["applied_mask"],
            true_potential_before=cur_p.copy(),
            true_potential_after=cur_p[survived].copy(),
            true_qualities_before=cur_q.copy(),
            true_qualities_after=cur_q[survived].copy(),
            qualities_assessed=out["qualities_mask"].copy(),
            productivity_before=prod_before,
            productivity_after=full_productivity[survivor_full_idx].copy(),
            opportunity_before=opp_before,
            opportunity_after=full_opportunity[survivor_full_idx].copy(),
            productivity_dimension_used=out["productivity_dimension_used"],
            resource_awarded=float(app.resource),
        ))
        cur_q = cur_q[survived]
        cur_p = cur_p[survived]
        cur_to_full = cur_to_full[survived]
        if cur_sa is not None:
            cur_sa = cur_sa[survived]

    return results, qualities, potential, sa_acc, full_productivity, full_opportunity


# --- save / load ---

import pickle

BUNDLE_FORMAT_VERSION = 3


@dataclass
class SimulationBundle:
    """Snapshot of a simulation run. v3 adds full-population productivity /
    opportunity arrays so the Lotka log-log panel can be rendered from a
    saved file."""
    format_version: int
    quality_configs: list[QualityConfig]
    reviewer_accuracy_configs: list[ReviewerAccuracyConfig]
    self_selection_config: Optional[SelfSelectionConfig]
    productivity_config: ProductivityConfig
    sigmoid_config: SigmoidConfig
    application_configs: list[ApplicationConfig]
    initial_qualities: np.ndarray            # (actual_initial_pop, Nq)
    initial_potential: np.ndarray            # (actual_initial_pop,) in [0,1]
    self_assessment_accuracies: Optional[np.ndarray]
    results: list[ApplicationResult]
    actual_initial_pop: int
    target_post_ss: int
    seed: int
    # v3 additions: final productivity / opportunity for *every* initial
    # scientist (zero for those who never passed a filter).
    full_productivity: np.ndarray
    full_opportunity: np.ndarray


def save_bundle(bundle: SimulationBundle, file_obj) -> None:
    pickle.dump(bundle, file_obj)


def load_bundle(file_obj) -> SimulationBundle:
    obj = pickle.load(file_obj)
    if not isinstance(obj, SimulationBundle):
        raise ValueError("File is not a SimulationBundle.")
    if obj.format_version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Bundle format v{obj.format_version} is not supported "
            f"(this build expects v{BUNDLE_FORMAT_VERSION}). "
            f"Re-run the simulation with the current code to produce a v"
            f"{BUNDLE_FORMAT_VERSION} file."
        )
    return obj
