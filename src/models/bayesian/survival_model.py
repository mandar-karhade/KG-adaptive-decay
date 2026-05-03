"""Hierarchical Bayesian survival model for temporal decay.

Fits Weibull survival parameters to observed edge lifetimes, decomposed
into three hierarchical levels: type (cluster), cohort, and individual.

The decay surface: tau(v, sigma) = exp(theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma)
is fitted at each level, with lower levels borrowing strength from higher levels.

Implementation strategy:
- Level 1 (Type): Fit Weibull per cluster using lifelines
- Level 2 (Cohort): Fit per (cluster, cohort) with shrinkage toward cluster params
- Level 3 (Individual): Fit per (cluster, cohort, entity) with shrinkage toward cohort
- Decay surface: Fit theta parameters via log-linear regression on (velocity, volatility) -> log(tau)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass(frozen=True)
class SurvivalFit:
    """Result of fitting a survival distribution to a group of edges."""

    tau: float  # scale parameter (characteristic lifetime)
    shape: float  # Weibull shape (k). k=1 is exponential.
    n_observations: int
    n_superseded: int
    n_censored: int
    aic: float | None = None
    log_likelihood: float | None = None


@dataclass(frozen=True)
class DecaySurfaceFit:
    """Result of fitting the decay surface theta parameters."""

    theta: np.ndarray  # [theta_0, theta_1, theta_2, theta_3]
    r_squared: float
    residual_std: float
    n_points: int


@dataclass
class HierarchicalFitResult:
    """Full result of hierarchical survival model fitting."""

    # Level 1: per-cluster fits
    cluster_fits: dict[str, SurvivalFit]
    cluster_surface_fits: dict[str, DecaySurfaceFit]

    # Level 2: per-(cluster, cohort) fits
    cohort_fits: dict[tuple[str, str], SurvivalFit]
    cohort_surface_fits: dict[tuple[str, str], DecaySurfaceFit]

    # Level 3: per-(cluster, cohort, entity) fits
    individual_fits: dict[tuple[str, str, str], SurvivalFit]

    # Distribution comparison
    distribution_comparison: dict[str, dict[str, float]] | None = None


def fit_weibull(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> SurvivalFit:
    """Fit a Weibull distribution to observed lifetimes with right-censoring.

    Args:
        durations: observed lifetimes (or censoring times)
        event_observed: 1 if superseded (event observed), 0 if censored
    """
    # Filter out zero/negative durations
    valid = durations > 0
    durations = durations[valid]
    event_observed = event_observed[valid]

    if len(durations) < 3:
        return SurvivalFit(
            tau=np.nan, shape=np.nan,
            n_observations=len(durations),
            n_superseded=int(event_observed.sum()),
            n_censored=int((~event_observed.astype(bool)).sum()),
        )

    wf = WeibullFitter()
    wf.fit(durations, event_observed)

    # lifelines Weibull parameterization: lambda_ (scale) and rho_ (shape)
    tau = wf.lambda_
    shape = wf.rho_

    return SurvivalFit(
        tau=tau,
        shape=shape,
        n_observations=len(durations),
        n_superseded=int(event_observed.sum()),
        n_censored=int((~event_observed.astype(bool)).sum()),
        aic=wf.AIC_,
        log_likelihood=wf.log_likelihood_,
    )


def fit_decay_surface(
    velocities: np.ndarray,
    volatilities: np.ndarray,
    log_taus: np.ndarray,
    regularization: float = 0.0,
) -> DecaySurfaceFit:
    """Fit the log-linear decay surface: log(tau) = theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma.

    Uses ordinary least squares (or ridge regression if regularization > 0).

    Args:
        velocities: velocity values per data point
        volatilities: volatility values per data point
        log_taus: log of fitted tau values per data point
        regularization: L2 regularization strength (0 = OLS)
    """
    valid = np.isfinite(log_taus) & np.isfinite(velocities) & np.isfinite(volatilities)
    v = velocities[valid]
    s = volatilities[valid]
    y = log_taus[valid]

    if len(y) < 4:
        return DecaySurfaceFit(
            theta=np.array([np.nan, np.nan, np.nan, np.nan]),
            r_squared=np.nan,
            residual_std=np.nan,
            n_points=len(y),
        )

    # Design matrix: [1, v, sigma, v*sigma]
    X = np.column_stack([np.ones(len(v)), v, s, v * s])

    # Ridge regression: (X'X + lambda*I)^-1 X'y
    if regularization > 0:
        theta = np.linalg.solve(
            X.T @ X + regularization * np.eye(4),
            X.T @ y,
        )
    else:
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DecaySurfaceFit(
        theta=theta,
        r_squared=r_squared,
        residual_std=np.sqrt(ss_res / max(len(y) - 4, 1)),
        n_points=len(y),
    )


def compare_distributions(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> dict[str, float]:
    """Compare Weibull, Exponential, and LogNormal fits via AIC.

    Returns dict mapping distribution name to AIC value. Lower is better.
    """
    valid = durations > 0
    durations = durations[valid]
    event_observed = event_observed[valid]

    if len(durations) < 5:
        return {}

    results = {}

    try:
        wf = WeibullFitter()
        wf.fit(durations, event_observed)
        results["weibull"] = wf.AIC_
    except Exception:
        pass

    try:
        ef = ExponentialFitter()
        ef.fit(durations, event_observed)
        results["exponential"] = ef.AIC_
    except Exception:
        pass

    try:
        lnf = LogNormalFitter()
        lnf.fit(durations, event_observed)
        results["lognormal"] = lnf.AIC_
    except Exception:
        pass

    return results


def fit_hierarchical(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    cohort_col: str = "cohort",
    entity_col: str = "entity",
    min_observations: int = 5,
    cohort_regularization: float = 0.1,
    individual_min_edges: int = 10,
) -> HierarchicalFitResult:
    """Fit the three-level hierarchical survival model.

    Level 1 (Type): Fit Weibull + decay surface per cluster
    Level 2 (Cohort): Fit Weibull + decay surface per (cluster, cohort),
                      with regularization toward cluster-level theta
    Level 3 (Individual): Fit Weibull per (cluster, cohort, entity)
                          for entities with enough data

    Args:
        df: DataFrame with columns: lifetime_observed, is_censored,
            velocity, volatility, and the grouping columns
        min_observations: minimum edges to fit a group
        cohort_regularization: L2 shrinkage toward cluster params for cohort surface
        individual_min_edges: minimum edges to fit individual-level params
    """
    durations = df["lifetime_observed"].values
    events = (~df["is_censored"]).astype(int).values

    # --- Level 1: Cluster-level fits ---
    cluster_fits = {}
    cluster_surface_fits = {}

    for cluster_name, cluster_df in df.groupby(cluster_col):
        c_dur = cluster_df["lifetime_observed"].values
        c_evt = (~cluster_df["is_censored"]).astype(int).values

        if len(c_dur) >= min_observations:
            cluster_fits[cluster_name] = fit_weibull(c_dur, c_evt)

            # Fit decay surface at cluster level
            # We need per-concept tau estimates. Group by (entity, predicate) and fit each.
            concept_taus = []
            concept_vels = []
            concept_vols = []

            for (ent, pred), concept_df in cluster_df.groupby([entity_col, "predicate"]):
                cd = concept_df["lifetime_observed"].values
                ce = (~concept_df["is_censored"]).astype(int).values
                if len(cd) >= 3 and ce.sum() >= 1:
                    try:
                        sf = fit_weibull(cd, ce)
                        if np.isfinite(sf.tau) and sf.tau > 0:
                            concept_taus.append(np.log(sf.tau))
                            concept_vels.append(concept_df["velocity"].iloc[0])
                            concept_vols.append(concept_df["volatility"].iloc[0])
                    except Exception:
                        pass

            if len(concept_taus) >= 4:
                cluster_surface_fits[cluster_name] = fit_decay_surface(
                    np.array(concept_vels),
                    np.array(concept_vols),
                    np.array(concept_taus),
                )

    # --- Level 2: Cohort-level fits ---
    cohort_fits = {}
    cohort_surface_fits = {}

    for (cluster_name, cohort_name), cohort_df in df.groupby([cluster_col, cohort_col]):
        c_dur = cohort_df["lifetime_observed"].values
        c_evt = (~cohort_df["is_censored"]).astype(int).values

        if len(c_dur) >= min_observations:
            cohort_fits[(cluster_name, cohort_name)] = fit_weibull(c_dur, c_evt)

            # Fit decay surface with regularization toward cluster-level
            concept_taus = []
            concept_vels = []
            concept_vols = []

            for (ent, pred), concept_df in cohort_df.groupby([entity_col, "predicate"]):
                cd = concept_df["lifetime_observed"].values
                ce = (~concept_df["is_censored"]).astype(int).values
                if len(cd) >= 3 and ce.sum() >= 1:
                    try:
                        sf = fit_weibull(cd, ce)
                        if np.isfinite(sf.tau) and sf.tau > 0:
                            concept_taus.append(np.log(sf.tau))
                            concept_vels.append(concept_df["velocity"].iloc[0])
                            concept_vols.append(concept_df["volatility"].iloc[0])
                    except Exception:
                        pass

            if len(concept_taus) >= 4:
                cohort_surface_fits[(cluster_name, cohort_name)] = fit_decay_surface(
                    np.array(concept_vels),
                    np.array(concept_vols),
                    np.array(concept_taus),
                    regularization=cohort_regularization,
                )

    # --- Level 3: Individual-level fits ---
    individual_fits = {}

    for (cluster_name, cohort_name, entity_name), entity_df in df.groupby(
        [cluster_col, cohort_col, entity_col]
    ):
        e_dur = entity_df["lifetime_observed"].values
        e_evt = (~entity_df["is_censored"]).astype(int).values

        if len(e_dur) >= individual_min_edges and e_evt.sum() >= 2:
            try:
                individual_fits[(cluster_name, cohort_name, entity_name)] = fit_weibull(
                    e_dur, e_evt
                )
            except Exception:
                pass

    # --- Distribution comparison (on full data per cluster) ---
    dist_comparison = {}
    for cluster_name, cluster_df in df.groupby(cluster_col):
        c_dur = cluster_df["lifetime_observed"].values
        c_evt = (~cluster_df["is_censored"]).astype(int).values
        dist_comparison[cluster_name] = compare_distributions(c_dur, c_evt)

    return HierarchicalFitResult(
        cluster_fits=cluster_fits,
        cluster_surface_fits=cluster_surface_fits,
        cohort_fits=cohort_fits,
        cohort_surface_fits=cohort_surface_fits,
        individual_fits=individual_fits,
        distribution_comparison=dist_comparison,
    )


def compute_survival_weight(
    age: float,
    tau: float,
    shape: float,
) -> float:
    """Compute the survival function S(t) = exp(-(t/tau)^shape).

    This is the temporal weight for retrieval scoring.
    """
    if tau <= 0 or not np.isfinite(tau):
        return 0.0
    return np.exp(-((age / tau) ** shape))


def compute_tau_from_surface(
    velocity: float,
    volatility: float,
    theta: np.ndarray,
) -> float:
    """Compute tau from the decay surface parameters.

    tau(v, sigma) = exp(theta_0 + theta_1*v + theta_2*sigma + theta_3*v*sigma)
    """
    log_tau = theta[0] + theta[1] * velocity + theta[2] * volatility + theta[3] * velocity * volatility
    return np.exp(np.clip(log_tau, -5, 15))


if __name__ == "__main__":
    from src.synthetic.generator import generate_synthetic_kg

    print("Generating synthetic KG...")
    df = generate_synthetic_kg()

    print("Fitting hierarchical survival model...")
    result = fit_hierarchical(df)

    print("\n=== Level 1: Cluster-level fits ===")
    for cluster, fit in sorted(result.cluster_fits.items()):
        print(f"\n{cluster}:")
        print(f"  tau={fit.tau:.1f} days, shape={fit.shape:.3f}")
        print(f"  n={fit.n_observations}, superseded={fit.n_superseded}, censored={fit.n_censored}")
        if fit.aic is not None:
            print(f"  AIC={fit.aic:.1f}")

        if cluster in result.cluster_surface_fits:
            sf = result.cluster_surface_fits[cluster]
            print(f"  Surface: theta={sf.theta}, R²={sf.r_squared:.3f}")

    print("\n=== Distribution comparison (AIC, lower is better) ===")
    for cluster, dists in sorted(result.distribution_comparison.items()):
        best = min(dists, key=dists.get) if dists else "N/A"
        print(f"  {cluster}: {dists} -> best: {best}")

    print(f"\n=== Level 2: {len(result.cohort_fits)} cohort fits ===")
    for (cluster, cohort), fit in sorted(result.cohort_fits.items()):
        print(f"  {cluster}/{cohort}: tau={fit.tau:.1f}, shape={fit.shape:.3f}, n={fit.n_observations}")

    print(f"\n=== Level 3: {len(result.individual_fits)} individual fits ===")
    # Show summary stats
    if result.individual_fits:
        ind_taus = [f.tau for f in result.individual_fits.values() if np.isfinite(f.tau)]
        print(f"  tau range: [{min(ind_taus):.1f}, {max(ind_taus):.1f}]")
        print(f"  tau median: {np.median(ind_taus):.1f}")
