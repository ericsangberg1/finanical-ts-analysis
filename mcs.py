"""
Model Confidence Set (Hansen, Lunde & Nason 2011) infrastructure for task3.

Used by `task3.ipynb` to:
  - compute per-horizon losses for six candidate families (RW, ARMA, three
    GARCH variants, ARMA-GARCH),
  - run the MCS bootstrap on QLIKE and squared-error losses,
  - simulate 200-day forward paths under the MCS-selected model,
  - run the PIT calibration check (Method 5.4.4 of the lecture notes).

Forecast and simulation routines follow §2.4 / §3.2.5 / §5.4.1 of the lecture
notes: closed-form best linear predictor for ARMA, parametric bootstrap
(Monte Carlo) for the nonlinear variance families.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.bootstrap import MCS
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


VAR_FLOOR = 1e-12

FAMILIES = [
    "RW", "ARMA", "GARCH(1,1)-norm", "GARCH(KC')", "EGARCH(KC')", "ARMA-GARCH",
]
PARSIMONY_ORDER = FAMILIES   # same order: simpler models first


# ============================================================================
# Result container
# ============================================================================

@dataclass(frozen=True)
class MCSResult:
    asset: str
    loss_name: str
    pvalues: pd.Series
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    note: str = ""


# ============================================================================
# Per-step losses
# ============================================================================

def per_step_losses(
    realized: np.ndarray, mean_hat: np.ndarray, var_hat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Squared-error (mean) and QLIKE (variance) losses at each holdout step."""
    var_safe = np.maximum(var_hat, VAR_FLOOR)
    eps = realized - mean_hat
    return {
        "se_mean": eps ** 2,
        "qlike": np.log(var_safe) + (realized ** 2) / var_safe,
    }


# ============================================================================
# Forecast generators — used to compute the per-horizon losses on the holdout
# ============================================================================

def forecast_rw(train: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """RW benchmark: constant sample mean + constant sample variance."""
    mu = float(train.mean())
    sigma2 = float(train.var(ddof=1))
    return np.full(horizon, mu), np.full(horizon, sigma2)


def forecast_arma(
    train: np.ndarray, horizon: int, order: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ARMA best linear predictor (§3.2.5 of the lecture notes)."""
    p, q = order
    if p == 0 and q == 0:
        return forecast_rw(train, horizon)
    fit = ARIMA(train, order=(p, 0, q), trend="c").fit()
    fc = fit.get_forecast(steps=horizon)
    return np.asarray(fc.predicted_mean, dtype=float), np.asarray(fc.var_pred_mean, dtype=float)


def forecast_garch(
    train: np.ndarray, horizon: int, *,
    p: int, q: int, dist: str = "normal", vol: str = "GARCH",
) -> tuple[np.ndarray, np.ndarray]:
    """GARCH / EGARCH multi-step forecast via the arch library.

    EGARCH lacks a closed-form multi-step variance and is handled by simulation
    (Method 5.4.1 of the lecture notes). Plain GARCH uses the analytic path.
    """
    scale = 100.0
    am = arch_model(
        train * scale, mean="Constant", vol=vol,
        p=p, q=q, dist=dist, rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    method = "simulation" if vol.upper() == "EGARCH" else "analytic"
    fc = res.forecast(
        horizon=horizon, reindex=False, method=method, simulations=2000,
    )
    mean_hat = np.asarray(fc.mean.iloc[-1].values, dtype=float) / scale
    var_hat = np.asarray(fc.variance.iloc[-1].values, dtype=float) / (scale ** 2)
    return mean_hat, var_hat


def forecast_arma_garch(
    train: np.ndarray, horizon: int, *,
    arma_order: tuple[int, int],
    p: int, q: int, dist: str = "normal",
    n_paths: int = 1000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage ARMA(p,q)-GARCH(p',q') (Def. 4.3.1 of the lecture notes).

    Method 4.2.2 (two-pass estimator): ARMA on the returns first, GARCH on the
    residuals. Aggregate per-horizon mean and variance are returned by Monte
    Carlo over the joint process.
    """
    paths = _simulate_arma_garch_paths(
        train, arma_order=arma_order, p=p, q=q, dist=dist,
        horizon=horizon, n_paths=n_paths, seed=seed,
    )
    return paths.mean(axis=0), paths.var(axis=0, ddof=1)


# ============================================================================
# Path-level simulators — used for the forward forecast and PIT calibration
# ============================================================================

def _simulate_arma_paths(
    train: np.ndarray, arma_order: tuple[int, int],
    horizon: int, n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    pa, qa = arma_order
    if pa == 0 and qa == 0:
        mu = float(train.mean())
        sigma = float(train.std(ddof=1))
        return rng.normal(mu, sigma, size=(n_paths, horizon))

    fit = ARIMA(train, order=(pa, 0, qa), trend="c").fit()
    sigma_e = float(np.std(fit.resid, ddof=1))
    phi = np.asarray(fit.arparams, dtype=float) if pa > 0 else np.array([])
    theta = np.asarray(fit.maparams, dtype=float) if qa > 0 else np.array([])
    intercept = (
        float(np.asarray(fit.params)[fit.param_names.index("const")])
        if "const" in fit.param_names else 0.0
    )
    e_paths = rng.normal(0.0, sigma_e, size=(n_paths, horizon))
    x_hist0 = np.asarray(train[-pa:], dtype=float) if pa > 0 else np.zeros(0)
    e_hist0 = np.asarray(fit.resid[-qa:], dtype=float) if qa > 0 else np.zeros(0)

    out = np.empty((n_paths, horizon), dtype=float)
    for k in range(n_paths):
        x_buf = x_hist0.tolist()
        e_buf = e_hist0.tolist()
        for h in range(horizon):
            e_new = float(e_paths[k, h])
            x_new = intercept
            for j, ph_j in enumerate(phi):
                x_new += ph_j * x_buf[-(j + 1)]
            for j, th_j in enumerate(theta):
                x_new += th_j * e_buf[-(j + 1)]
            x_new += e_new
            out[k, h] = x_new
            if pa > 0:
                x_buf.append(x_new); x_buf = x_buf[-pa:]
            if qa > 0:
                e_buf.append(e_new); e_buf = e_buf[-qa:]
    return out


def _simulate_garch_paths(
    train: np.ndarray, *,
    p: int, q: int, dist: str, vol: str,
    horizon: int, n_paths: int,
) -> np.ndarray:
    scale = 100.0
    am = arch_model(
        train * scale, mean="Constant", vol=vol,
        p=p, q=q, dist=dist, rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    sim = res.forecast(
        horizon=horizon, reindex=False,
        method="simulation", simulations=n_paths,
    )
    return np.asarray(sim.simulations.values[0], dtype=float) / scale


def _simulate_arma_garch_paths(
    train: np.ndarray, *,
    arma_order: tuple[int, int],
    p: int, q: int, dist: str,
    horizon: int, n_paths: int, seed: int = 42,
) -> np.ndarray:
    pa, qa = arma_order
    arma_fit = ARIMA(train, order=(pa, 0, qa), trend="c").fit()
    resid = np.asarray(arma_fit.resid, dtype=float)

    scale = 100.0
    am = arch_model(
        resid * scale, mean="Zero", vol="GARCH",
        p=p, q=q, dist=dist, rescale=False,
    )
    gf = am.fit(disp="off", show_warning=False)
    sim = gf.forecast(
        horizon=horizon, reindex=False,
        method="simulation", simulations=n_paths,
    )
    e_paths = np.asarray(sim.simulations.values[0], dtype=float) / scale

    phi = np.asarray(arma_fit.arparams, dtype=float) if pa > 0 else np.array([])
    theta = np.asarray(arma_fit.maparams, dtype=float) if qa > 0 else np.array([])
    intercept = (
        float(np.asarray(arma_fit.params)[arma_fit.param_names.index("const")])
        if "const" in arma_fit.param_names else 0.0
    )
    x_hist0 = np.asarray(train[-pa:], dtype=float) if pa > 0 else np.zeros(0)
    e_hist0 = resid[-qa:] if qa > 0 else np.zeros(0)

    out = np.empty((n_paths, horizon), dtype=float)
    for k in range(n_paths):
        x_buf = x_hist0.tolist()
        e_buf = e_hist0.tolist()
        for h in range(horizon):
            e_new = float(e_paths[k, h])
            x_new = intercept
            for j, ph_j in enumerate(phi):
                x_new += ph_j * x_buf[-(j + 1)]
            for j, th_j in enumerate(theta):
                x_new += th_j * e_buf[-(j + 1)]
            x_new += e_new
            out[k, h] = x_new
            if pa > 0:
                x_buf.append(x_new); x_buf = x_buf[-pa:]
            if qa > 0:
                e_buf.append(e_new); e_buf = e_buf[-qa:]
    return out


def simulate_forward(
    family: str, train: np.ndarray, *,
    arma_order: tuple[int, int] = (0, 0),
    garch_spec: dict | None = None,
    horizon: int = 200, n_paths: int = 1000, seed: int = 42,
) -> np.ndarray:
    """Refit `family` on `train` and return `(n_paths, horizon)` return paths."""
    rng = np.random.default_rng(seed)
    spec = garch_spec or {"p": 1, "q": 1, "dist": "normal"}
    p, q, dist = spec["p"], spec["q"], spec["dist"]

    if family == "RW":
        mu = float(train.mean())
        sigma = float(train.std(ddof=1))
        return rng.normal(mu, sigma, size=(n_paths, horizon))
    if family == "ARMA":
        return _simulate_arma_paths(train, arma_order, horizon, n_paths, rng)
    if family == "GARCH(1,1)-norm":
        return _simulate_garch_paths(
            train, p=1, q=1, dist="normal", vol="GARCH",
            horizon=horizon, n_paths=n_paths,
        )
    if family == "GARCH(KC')":
        return _simulate_garch_paths(
            train, p=p, q=q, dist=dist, vol="GARCH",
            horizon=horizon, n_paths=n_paths,
        )
    if family == "EGARCH(KC')":
        return _simulate_garch_paths(
            train, p=p, q=q, dist=dist, vol="EGARCH",
            horizon=horizon, n_paths=n_paths,
        )
    if family == "ARMA-GARCH":
        return _simulate_arma_garch_paths(
            train, arma_order=arma_order, p=p, q=q, dist=dist,
            horizon=horizon, n_paths=n_paths, seed=seed,
        )
    raise ValueError(f"Unknown family: {family}")


# ============================================================================
# MCS bootstrap + selection
# ============================================================================

def run_mcs(
    loss_df: pd.DataFrame, *,
    asset: str, loss_name: str,
    alpha: float = 0.10, n_boot: int = 1000, seed: int = 42,
) -> MCSResult:
    """Run the Hansen, Lunde & Nason (2011) MCS on a `T × K` per-step loss matrix.

    Uses the stationary block bootstrap with block length √T. When the loss
    differentials are nearly indistinguishable across all models, the arch
    library raises an IndexError during elimination — in that case we return the
    full candidate set with a note, which is the statistically honest answer.
    """
    block_size = max(2, int(np.sqrt(len(loss_df))))
    try:
        mcs = MCS(
            loss_df, size=alpha, reps=n_boot,
            block_size=block_size, seed=seed,
        )
        mcs.compute()
        pvals = mcs.pvalues.iloc[:, 0].sort_values(ascending=False)
        return MCSResult(
            asset=asset, loss_name=loss_name, pvalues=pvals,
            included=tuple(mcs.included), excluded=tuple(mcs.excluded),
        )
    except IndexError:
        models = tuple(loss_df.columns)
        pvals = pd.Series({m: 1.0 for m in models}, name="Pvalue")
        return MCSResult(
            asset=asset, loss_name=loss_name, pvalues=pvals,
            included=models, excluded=(),
            note="degenerate: losses indistinguishable, full set retained",
        )


def select_from_mcs(
    results: list[MCSResult],
    parsimony_order: list[str] | None = None,
) -> dict[str, str]:
    """For each series, return the most parsimonious survivor of the MCS."""
    order = parsimony_order or PARSIMONY_ORDER
    chosen: dict[str, str] = {}
    for r in results:
        survivors = set(r.included)
        for family in order:
            if family in survivors:
                chosen[r.asset] = family
                break
    return chosen


# ============================================================================
# PIT calibration (Method 5.4.4)
# ============================================================================

def pit_check(
    realized: np.ndarray, holdout_paths: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Per-step Probability Integral Transform + Kolmogorov-Smirnov test.

    `holdout_paths` is shape `(n_paths, horizon)`. For each horizon h, `u_h` is
    the empirical CDF of the simulated paths evaluated at the realised value.
    Under correct model specification `{u_h}` is uniform on `[0, 1]`; deviations
    diagnose miscalibration (mountain shape = fan too wide, U = fan too narrow).
    """
    u = np.array([
        (holdout_paths[:, h] <= realized[h]).mean()
        for h in range(len(realized))
    ])
    ks_stat, ks_p = stats.kstest(u, "uniform")
    return u, float(ks_stat), float(ks_p)


def price_paths(returns_paths: np.ndarray, p0: float) -> np.ndarray:
    """Convert log-return paths to price paths anchored at p0."""
    return p0 * np.exp(np.cumsum(returns_paths, axis=1))
