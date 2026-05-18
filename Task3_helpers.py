"""
Self-contained helper module for task3.ipynb.

Public surface used by task3.ipynb:
    clean_dataframe
    GARCHSpec, fit
    select_all_series
    evaluate_holdout_all_series, holdout_summary

Cleaning functions (clean_spikes_rolling, clean_spikes_global,
get_valid_segments_fixed) are NOT defined here — they are loaded at
import time directly from Task1.ipynb so task3 always uses Task 1's
canonical implementation. Any change to the cleaning code in Task1.ipynb
takes effect on the next Python kernel restart with no edits here.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import (
    ARX,
    ConstantMean,
    ConstantVariance,
    Normal,
    SkewStudent,
    StudentsT,
)
from arch.univariate.base import ARCHModelResult


# ============================================================================
# Data cleaning — re-loaded from Task1.ipynb so the implementation is shared.
# ============================================================================

_TASK1_PATH = Path(__file__).with_name("Task1.ipynb")
_NEEDED_NAMES = ("clean_spikes_rolling", "clean_spikes_global", "get_valid_segments_fixed")


def _load_cleaning_functions_from_task1() -> dict:
    """Parse Task1.ipynb and exec the cell sources that define the cleaning
    functions used here. Returns a namespace dict containing them."""
    if not _TASK1_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find Task1.ipynb at {_TASK1_PATH}. "
            "task3_helpers.py loads its cleaning functions from there."
        )

    nb = json.loads(_TASK1_PATH.read_text())
    # collect every cell whose source defines at least one of the needed
    # cleaning functions — they live in different cells in Task1.ipynb.
    relevant_sources: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if defined & set(_NEEDED_NAMES):
            relevant_sources.append(src)

    # exec only the function definitions (skip top-level statements that would
    # require notebook-specific globals like `df1`, `val_cols`, etc.).
    namespace: dict = {"np": np, "pd": pd}
    for src in relevant_sources:
        tree = ast.parse(src)
        keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not keep:
            continue
        module = ast.Module(body=keep, type_ignores=[])
        exec(compile(module, str(_TASK1_PATH), "exec"), namespace)

    missing = [n for n in _NEEDED_NAMES if n not in namespace]
    if missing:
        raise ImportError(
            f"Task1.ipynb did not define expected cleaning function(s): {missing}"
        )
    return namespace


_t1 = _load_cleaning_functions_from_task1()
clean_spikes_rolling = _t1["clean_spikes_rolling"]
clean_spikes_global = _t1["clean_spikes_global"]
get_valid_segments_fixed = _t1["get_valid_segments_fixed"]


def clean_dataframe(
    df: pd.DataFrame,
    truncate_at: int = 5255,
    window: int = 150,
    tolerance: float = 8.0,
) -> pd.DataFrame:
    """Truncate trailing missing block and apply Task 1's spike cleaning
    column-by-column. Mirrors the per-column loop in Task1.ipynb cell 9."""
    df_trunc = df.loc[:truncate_at]
    df_clean = df_trunc.copy()
    for col in df_trunc.columns:
        s = df_trunc[col]
        col_idx = df_clean.columns.get_loc(col)
        for st, ed in get_valid_segments_fixed(s):
            segment = s.iloc[st:ed]
            if ed - st >= int(1.5 * window):
                cleaned, _ = clean_spikes_rolling(segment, window=window, tolerance=tolerance)
            else:
                cleaned, _ = clean_spikes_global(segment, tolerance=tolerance)
            df_clean.iloc[st:ed, col_idx] = cleaned.values
    return df_clean


# ============================================================================
# GARCH spec + fit
# ============================================================================

VolModel = Literal["ConstantVariance", "Garch", "EGARCH"]
DistModel = Literal["normal", "t", "skewt"]
TrainWindow = Literal["full", "rolling", "post_break"]


@dataclass
class GARCHSpec:
    vol: VolModel = "Garch"
    p: int = 1
    q: int = 1
    ar_order: int = 0
    ma_order: int = 0
    dist: DistModel = "t"
    train_window: TrainWindow = "full"
    window_size: int | None = None
    start_day: int | None = None

    def __str__(self) -> str:
        if self.ar_order > 0 and self.ma_order > 0:
            mean = f"ARMA({self.ar_order},{self.ma_order})"
        elif self.ar_order > 0:
            mean = f"AR({self.ar_order})"
        elif self.ma_order > 0:
            mean = f"MA({self.ma_order})"
        else:
            mean = "const"
        if self.vol == "ConstantVariance":
            base = f"{mean}-ConstVar-{self.dist}"
        else:
            base = f"{mean}-{self.vol}({self.p},{self.q})-{self.dist}"
        if self.train_window == "rolling" and self.window_size is not None:
            base += f"[roll{self.window_size}]"
        elif self.train_window == "post_break" and self.start_day is not None:
            base += f"[post{self.start_day}]"
        return base


def _distribution(spec: GARCHSpec):
    if spec.dist == "skewt":
        return SkewStudent()
    if spec.dist == "t":
        return StudentsT()
    return Normal()


def _apply_training_window(y_data: pd.Series, spec: GARCHSpec) -> pd.Series:
    y = y_data.dropna()
    if spec.train_window == "rolling" and spec.window_size is not None:
        y = y.iloc[-spec.window_size:]
    elif spec.train_window == "post_break" and spec.start_day is not None:
        y = y.loc[y.index >= spec.start_day]
    return y


def fit(series: pd.Series, spec: GARCHSpec, rescale: bool = True) -> ARCHModelResult:
    """Fit a GARCH spec to a return series. Returns the fitted ARCHModelResult."""
    y_source = _apply_training_window(series, spec)
    y = y_source * 100 if rescale else y_source

    if spec.ma_order > 0:
        from statsmodels.tsa.arima.model import ARIMA
        arma = ARIMA(y.values, order=(spec.ar_order, 0, spec.ma_order), trend="c").fit()
        residuals = pd.Series(arma.resid, index=y.index)
        model = arch_model(
            residuals, mean="Zero", vol=spec.vol,
            p=spec.p, q=spec.q, dist=spec.dist,
        )
        return model.fit(disp="off", show_warning=False)

    lags = list(range(1, spec.ar_order + 1)) if spec.ar_order > 0 else 0
    distribution = _distribution(spec)

    if spec.vol == "ConstantVariance":
        if spec.ar_order > 0:
            model = ARX(
                y, lags=lags, constant=True,
                volatility=ConstantVariance(),
                distribution=distribution,
                rescale=False,
            )
        else:
            model = ConstantMean(
                y, volatility=ConstantVariance(),
                distribution=distribution, rescale=False,
            )
        return model.fit(disp="off", show_warning=False)

    model = arch_model(
        y,
        mean="ARX" if spec.ar_order > 0 else "Constant",
        lags=lags, vol=spec.vol,
        p=spec.p, q=spec.q, dist=spec.dist,
    )
    return model.fit(disp="off", show_warning=False)


def simulate(
    result: ARCHModelResult,
    horizon: int,
    n_paths: int = 1000,
    rescale: bool = True,
) -> np.ndarray:
    """Simulate `horizon` steps forward. Returns array (n_paths, horizon)."""
    paths = np.empty((n_paths, horizon))
    for i in range(n_paths):
        sim = result.model.simulate(result.params, nobs=horizon)
        paths[i] = np.clip(sim["data"].values, -1000, 1000)
    scale = 100 if rescale else 1
    return paths / scale


def simulate_with_volatility(
    result: ARCHModelResult,
    horizon: int,
    n_paths: int = 1000,
    rescale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns + conditional volatility paths."""
    data_paths = np.empty((n_paths, horizon))
    vol_paths = np.empty((n_paths, horizon))
    for i in range(n_paths):
        sim = result.model.simulate(result.params, nobs=horizon)
        data_paths[i] = np.clip(sim["data"].values, -1000, 1000)
        vol_paths[i] = np.minimum(sim["volatility"].values, 1000)
    scale = 100 if rescale else 1
    return data_paths / scale, vol_paths / scale


# ============================================================================
# KC' order selection (Bardet, Karé & Kengne 2021)
# ============================================================================

def _kc_prime(result) -> float:
    n = result.nobs
    k = len(result.params)
    logL = result.loglikelihood
    try:
        cov = result.param_cov.values
        sign, logdet_cov = np.linalg.slogdet(cov)
        if sign <= 0:
            return result.bic
        return (
            -2.0 * logL
            + (np.log(n) - np.log(2.0 * np.pi)) * k
            + (-logdet_cov)
            + 2.0 * np.log(k)
        )
    except Exception:
        return result.bic


def _fit_kc(series: pd.Series, ar_order: int, p: int, q: int) -> float:
    spec = GARCHSpec(
        vol="Garch", p=p, q=q,
        ar_order=ar_order, ma_order=0,
        dist="normal",
    )
    try:
        result = fit(series, spec, rescale=True)
        result_classic = result.model.fit(
            disp="off", show_warning=False, cov_type="classic"
        )
        return _kc_prime(result_classic)
    except Exception:
        return np.inf


def select_all_series(
    df_clean: pd.DataFrame,
    val_cols: list[str],
    max_ar: int = 2,
    max_garch_p: int = 2,
    max_garch_q: int = 2,
) -> tuple[dict[str, GARCHSpec], pd.DataFrame]:
    """Joint KC' selection over all AR(r)-GARCH(p,q) combinations."""
    out: dict[str, GARCHSpec] = {}
    rows: list[dict] = []

    for col in val_cols:
        series = df_clean[col].dropna()
        best_kc = np.inf
        best_ar, best_p, best_q = 0, 1, 1

        for ar in range(max_ar + 1):
            for p in range(1, max_garch_p + 1):
                for q in range(1, max_garch_q + 1):
                    kc = _fit_kc(series, ar_order=ar, p=p, q=q)
                    rows.append({
                        "series":    col,
                        "candidate": f"AR({ar})-GARCH({p},{q})",
                        "ar":        ar,
                        "p":         p,
                        "q":         q,
                        "kc_prime":  round(kc, 4),
                        "selected":  False,
                    })
                    if kc < best_kc:
                        best_kc = kc
                        best_ar, best_p, best_q = ar, p, q

        spec = GARCHSpec(
            vol="Garch", p=best_p, q=best_q,
            ar_order=best_ar, ma_order=0,
            dist="normal",
        )
        print(f"  {col:35s}  AR({best_ar})-GARCH({best_p},{best_q})  [dist=TBD]")
        out[col] = spec

        for row in rows:
            if (row["series"] == col
                    and row["ar"] == best_ar
                    and row["p"] == best_p
                    and row["q"] == best_q):
                row["selected"] = True

    return out, pd.DataFrame(rows)


# ============================================================================
# Holdout evaluation
# ============================================================================

def _qlike(actual: np.ndarray, forecast_var: np.ndarray) -> float:
    sigma2 = np.maximum(forecast_var, 1e-12)
    return float(np.mean(np.log(sigma2) + actual**2 / sigma2))


def _rmse_vol(actual: np.ndarray, forecast_var: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual**2 - forecast_var) ** 2)))


def _rmse_mean(actual: np.ndarray, forecast_mean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast_mean) ** 2)))


def _forecast_multi_step(
    result, horizon: int, spec: GARCHSpec,
) -> tuple[np.ndarray, np.ndarray]:
    needs_sim = spec.vol == "EGARCH"
    kwargs = {"horizon": horizon, "reindex": False}
    if needs_sim:
        kwargs["method"] = "simulation"
        kwargs["simulations"] = 500
    fc = result.forecast(**kwargs)
    mean = fc.mean.values[-1] / 100
    var = fc.variance.values[-1] / (100**2)
    return mean, var


def evaluate_holdout_forecast(
    series: pd.Series,
    spec: GARCHSpec,
    horizon: int = 200,
    n_paths: int = 1000,
) -> dict[str, object]:
    """Fit on series[:-horizon] and forecast the held-out window."""
    clean = series.dropna()
    if len(clean) <= horizon:
        raise ValueError(
            f"Series length {len(clean)} too short for holdout horizon {horizon}."
        )

    train = clean.iloc[:-horizon]
    actual = clean.iloc[-horizon:]
    actual_index = actual.index
    last_train_return = train.iloc[-1]

    result = fit(train, spec)
    forecast_mean, forecast_var = _forecast_multi_step(
        result, horizon=len(actual), spec=spec
    )
    simulated_return_paths = simulate(result, horizon=len(actual), n_paths=n_paths)
    _, simulated_vol_paths = simulate_with_volatility(
        result, horizon=len(actual), n_paths=n_paths
    )

    actual_values = actual.to_numpy()
    return {
        "actual": actual_values,
        "actual_index": actual_index.to_numpy(),
        "forecast_mean": forecast_mean,
        "forecast_var": forecast_var,
        "simulated_return_paths": simulated_return_paths,
        "simulated_vol_paths": simulated_vol_paths,
        "last_train_return": float(last_train_return),
        "rmse_mean": _rmse_mean(actual_values, forecast_mean),
        "qlike": _qlike(actual_values, forecast_var),
        "rmse_vol": _rmse_vol(actual_values, forecast_var),
        "train_size": len(train),
        "horizon": len(actual),
    }


def evaluate_holdout_all_series(
    df_clean: pd.DataFrame,
    selected_specs: dict[str, GARCHSpec],
    horizon: int = 200,
    n_paths: int = 1000,
) -> dict[str, dict[str, object]]:
    """Run holdout evaluation for every selected series."""
    out: dict[str, dict[str, object]] = {}
    for col, spec in selected_specs.items():
        print(f"  Holdout {col} ({spec}) ...", end=" ", flush=True)
        res = evaluate_holdout_forecast(
            df_clean[col], spec, horizon=horizon, n_paths=n_paths,
        )
        print(
            f"RMSE-mean={res['rmse_mean']:.6f}  "
            f"QLIKE={res['qlike']:.4f}  "
            f"RMSE-vol={res['rmse_vol']:.6f}"
        )
        out[col] = res
    return out


def holdout_summary(holdout_results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """One row per series with the headline holdout metrics."""
    rows = []
    for col, res in holdout_results.items():
        rows.append({
            "series": col,
            "train_size": res["train_size"],
            "horizon": res["horizon"],
            "rmse_mean": res["rmse_mean"],
            "qlike": res["qlike"],
            "rmse_vol": res["rmse_vol"],
        })
    return pd.DataFrame(rows).set_index("series")
