#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:33:45 2026

@author: axelivarsson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import arch
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings('ignore', category=InterpolationWarning)


os.getcwd()
os.chdir("/Users/axelivarsson/Desktop/FTS")

# Läs in data
df = pd.read_csv("spiff_data-2.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

asset_cols = [col for col in df.columns if col != "day"]

# Viktigt: använd np.nan, inte pd.NA
df[asset_cols] = df[asset_cols].replace(1000, np.nan)

# Se till att kolumnerna verkligen är numeriska
df[asset_cols] = df[asset_cols].apply(pd.to_numeric, errors="coerce")

df = df.set_index("day")

# Use common continuous segment after all internal 50-day gaps
start_day = 1500

df = df.loc[start_day:].copy()

# Update asset columns just to be safe
asset_cols = list(df.columns)
# =========================
# Robust outlier cleaning (rolling MAD)
# =========================

df_clean = df.copy()

window = 150
threshold = 5

for col in asset_cols:

    series = np.log(df_clean[col] / df_clean[col].shift(1))

    # Rolling median
    rolling_median = series.rolling(window, center=True).median()

    # Median Absolute Deviation (MAD)
    mad = (series - rolling_median).abs().rolling(window, center=True).median()

    # Robust z-score
    robust_z = (series - rolling_median).abs() / mad

    # Flag extreme spikes
    outliers = robust_z > threshold

    # Replace corresponding PRICE values with NaN
    df_clean.loc[outliers, col] = np.nan

# Interpolate only small/local gaps
df_clean[asset_cols] = df_clean[asset_cols].interpolate(limit=5)

print("Cleaning complete.")

plt.figure(figsize=(8, 6))

for col in df_clean.columns:
    plt.plot(df_clean.index, df_clean[col], linewidth=1.8, label=col)

plt.xlabel("Day")
plt.ylabel("Price")
plt.ylim(0, 22)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show() # Prices appear non-stationary, with no constant mean
# tranquility upward trend, but volatile
# water and gurkor steady upward trend, not so volatile
# guitars varying seasonally
# sugar rather steady downward trend
# trends vary a lot, also volatility
# around day 4000 many stocks crash, but bounce back quite fast
# some stocks seems to correlate
#%%
returns = df_clean.pct_change()
log_returns = np.log(df_clean) - np.log(df_clean.shift(1))
plt.figure(figsize=(12, 6))

for col in log_returns.columns:
    plt.plot(log_returns.index, log_returns[col], alpha=0.5, linewidth=1, label=col)

plt.title("Log-Returns")
plt.xlabel("Day")
plt.ylabel("Log-Return")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
# volatily clusters around 2500, 4000, 4800
# some outliers across the plot
# roughly symmetric around 0
# extreme movements occur simultaneously
# returns similar to log-return due to non-drastic changes
#%%
for col in df_clean.columns:
    
    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    
    # Returns
    axes[0].plot(returns.index, returns[col])
    axes[0].set_title(f"{col} - Returns")
    axes[0].grid(True, alpha=0.3)
    
    # Log-returns
    axes[1].plot(log_returns.index, log_returns[col])
    axes[1].set_title(f"{col} - Log Returns")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#%%
# =========================
# 6. Histograms of log-returns
# =========================
fig, axes = plt.subplots(len(asset_cols), 1, figsize=(10, 18))

for i, col in enumerate(asset_cols):
    axes[i].hist(log_returns[col].dropna(), bins=50)
    axes[i].set_title(f"{col} - Histogram of Log Returns")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# all centered around 0
# pointy top, deviates from normality, long tails
# guitars and tranquility a bit skewed
# heavy/fat tails for most
# 
#%%
# =========================
# 7. Summary statistics
# =========================
summary = pd.DataFrame({
    "mean": log_returns.mean(),
    "std": log_returns.std(),
    "skew": log_returns.skew(),
    "kurtosis": log_returns.kurtosis(),
    "min": log_returns.min(),
    "max": log_returns.max()
})

print("\nSummary statistics for log-returns:\n")
print(summary.round(4))
# most of them have 4-5 the volatility of gurkor and water
# skewness slightly negative for all, suggesting heavier negative tails
# normal distribution has kurtosis 0, stocks almost 0, sugar definitely not 0, suggesting heavy tails
# and usage of GARCH instead of normal distribution

# usual kurtosis 3
# excess kurtosis = 0, this is
# normal distribution underestimate risk

#%%
# =========================
# 8. Correlation matrix
# =========================
corr_matrix = log_returns.corr()

print("\nCorrelation matrix of log-returns:\n")
print(corr_matrix.round(3))
# clear relationship between water and gurkor
# slingshots and guitars
# many negative correlations
# group 1: gurkor, water (correlate)
# group 2: guitars, slingshots, sugar, tranquility (correlate)
# group 3: stocks (independent)

# =========================
# 9. Optional: ACF of returns and squared returns
# =========================
for col in asset_cols:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    plot_acf(log_returns[col].dropna(), ax=axes[0], lags=40)
    axes[0].set_title(f"{col} - ACF of Log Returns")
    
    plot_acf((log_returns[col].dropna() ** 2), ax=axes[1], lags=40)
    axes[1].set_title(f"{col} - ACF of Squared Log Returns")
    
    plt.tight_layout()
    plt.show()
#%%
from statsmodels.tsa.stattools import acf

plt.figure(figsize=(10,6))

for col in asset_cols:
    acf_vals = acf(log_returns[col].dropna(), nlags=40)
    plt.plot(acf_vals, label=col)

plt.axhline(0, linestyle="--", color="black")
plt.title("ACF of Log Returns (all assets)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(10,6))

for col in asset_cols:
    acf_vals = acf((log_returns[col].dropna()**2), nlags=40)
    plt.plot(acf_vals, label=col)

plt.axhline(0, linestyle="--", color="black")
plt.title("ACF of Squared Log Returns (all assets)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# low ACF in returns but high ACF in squared returns suggests predictable volatility
# but not direction of trajectory
# volatility clustering for guitars, slingshots, sugar seen by big dependence for small lag
# variance not constant, GARCH behavior
#%%

# =========================

# ADF, KPSS and Ljung-Box tests

# =========================

test_results = []

for col in asset_cols:

    x = log_returns[col].dropna()

    # ADF test

    adf_stat, adf_p, *_ = adfuller(x)

    # KPSS test

    kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")

    # Ljung-Box on returns

    lb_returns = acorr_ljungbox(x, lags=[20], return_df=True)

    lb_p = lb_returns["lb_pvalue"].iloc[0]

    # Ljung-Box on squared returns

    lb_squared = acorr_ljungbox(x**2, lags=[20], return_df=True)

    lb2_p = lb_squared["lb_pvalue"].iloc[0]

    test_results.append({

        "series": col,

        "ADF p-value": adf_p,

        "KPSS p-value": kpss_p,

        "LB returns p-value": lb_p,

        "LB squared returns p-value": lb2_p

    })

test_results_df = pd.DataFrame(test_results)

print("\nADF, KPSS and Ljung-Box test results:\n")

print(test_results_df.round(4))
#%%
# =========================
# Rolling volatility
# =========================

rolling_vol = log_returns.rolling(window=50).std()

plt.figure(figsize=(12, 6))

for col in asset_cols:
    plt.plot(
        rolling_vol.index,
        rolling_vol[col],
        linewidth=1.5,
        label=col
    )

plt.title("Rolling Volatility (50-day)")
plt.xlabel("Day")
plt.ylabel("Volatility")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%
# =========================
# Rolling correlations between assets
# =========================

# Make sure log_returns is a DataFrame

window = 100

pairs = [
    ("gurkor", "water"),
    ("guitars", "slingshots"),
    ("gurkor", "guitars"),
    ("gurkor", "slingshots"),
    ("stocks", "gurkor"),
    ("stocks", "guitars"),
    ("sugar", "tranquillity"),
]

rolling_corrs = {}

for a, b in pairs:
    rolling_corrs[f"{a} vs {b}"] = (
        log_returns[a]
        .rolling(window)
        .corr(log_returns[b])
    )

rolling_corrs_df = pd.DataFrame(rolling_corrs)

plt.figure(figsize=(12, 6))

for col in rolling_corrs_df.columns:
    plt.plot(rolling_corrs_df.index, rolling_corrs_df[col], label=col, linewidth=1.5)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"{window}-day Rolling Correlations of Log-Returns")
plt.xlabel("Day")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%
# =========================
# Rolling correlations, separate panels
# =========================

fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
axes = axes.flatten()

for i, col in enumerate(rolling_corrs_df.columns):
    axes[i].plot(rolling_corrs_df.index, rolling_corrs_df[col], linewidth=1.5)
    axes[i].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[i].set_title(col)
    axes[i].set_ylabel("Correlation")
    axes[i].grid(True, alpha=0.3)

fig.delaxes(axes[-1])

plt.xlabel("Day")
plt.tight_layout()
plt.show()
#%%
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
from scipy.spatial.distance import squareform

# =========================
# Rolling covariance
# =========================

window = 100

pairs = [
    ("gurkor", "water"),
    ("guitars", "slingshots"),
    ("gurkor", "guitars"),
    ("gurkor", "slingshots"),
    ("stocks", "gurkor"),
    ("stocks", "guitars"),
    ("sugar", "tranquillity"),
]


plt.figure(figsize=(12, 6))

for a, b in pairs:
    cov_ab = log_returns[a].rolling(window).cov(log_returns[b])
    plt.plot(cov_ab.index, cov_ab, label=f"{a} vs {b}")

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"{window}-day Rolling Covariances")
plt.xlabel("Day")
plt.ylabel("Covariance")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%
# =========================
# Rolling correlation heatmaps at selected days
# =========================

selected_days = [2000, 3000, 4000, 5000]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, day in enumerate(selected_days):
    # use data up to selected day
    data_window = log_returns.loc[:day].iloc[-window:]
    corr_mat = data_window.corr()

    sns.heatmap(
        corr_mat,
        ax=axes[i],
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        cbar=i == 0
    )

    axes[i].set_title(f"Rolling correlation matrix, day {day}")

plt.tight_layout()
plt.show()
#%%
# =========================
# Correlation clustering
# =========================

corr_mat = log_returns.dropna().corr()

# correlation distance
dist_mat = np.sqrt((1 - corr_mat) / 2)

# convert distance matrix to condensed form
condensed_dist = squareform(dist_mat, checks=False)

# hierarchical clustering
Z = linkage(condensed_dist, method="average")

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=corr_mat.columns)
plt.title("Hierarchical Clustering Based on Correlation Distance")
plt.ylabel("Distance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

############################## Task 3 #########################################
#%%
# =========================

# Modell 1, random walk

# =========================

h = 200 # steg framåt

z = 1.96   # 95% confidence interval

rw_forecasts = {} # random walk forecast values

rw_lower = {} # lower limit of interval

rw_upper = {}

for col in asset_cols:
    series = df_clean[col].dropna() # dataset
    last_day = int(series.index[-1]) # sista dagen användbar
    last_price = series.iloc[-1] # sista prisen användbara
    
    # Log-returns from observed data
    log_returns = np.log(series).diff().dropna()
    
    # Estimate daily volatility
    sigma = log_returns.std()
    
    # Future days
    future_days = np.arange(last_day + 1, last_day + h + 1)

    # Forecast log-price
    last_log_price = np.log(last_price)
    forecast_log_price = np.repeat(last_log_price, h)
    # Uncertainty grows with sqrt(h)
    steps = np.arange(1, h + 1)
    forecast_sd = sigma * np.sqrt(steps)
    lower_log = forecast_log_price - z * forecast_sd
    upper_log = forecast_log_price + z * forecast_sd
    
    # Back to price scale
    forecast_price = np.exp(forecast_log_price)
    lower_price = np.exp(lower_log)
    upper_price = np.exp(upper_log)
    rw_forecasts[col] = pd.Series(forecast_price, index=future_days)
    rw_lower[col] = pd.Series(lower_price, index=future_days)
    rw_upper[col] = pd.Series(upper_price, index=future_days)

rw_forecast_df_clean = pd.DataFrame(rw_forecasts)
rw_lower_df_clean = pd.DataFrame(rw_lower)
rw_upper_df_clean = pd.DataFrame(rw_upper)

# P_t+1 = P_t + e_t+1
# E[P_t+1] = P_1

# var[P_t+1] = h*sigma^2

#%%
rw_validation = {}

test_size = 200  # antal sista observationer som används som test

for col in asset_cols:

    series = df_clean[col].dropna()

    train_size = len(series) - test_size

    actual_prices = []

    forecast_prices = []

    errors = []

    pct_errors = []

    test_days = []

    for i in range(test_size):

        # information available at time t

        last_known_price = series.iloc[train_size + i - 1]

        # actual next price

        actual_price = series.iloc[train_size + i]

        # random walk forecast

        forecast_price = last_known_price

        actual_prices.append(actual_price)

        forecast_prices.append(forecast_price)

        errors.append(actual_price - forecast_price)

        pct_errors.append((actual_price - forecast_price) / forecast_price)

        test_days.append(series.index[train_size + i])

    actual_prices = pd.Series(actual_prices, index=test_days)

    forecast_prices = pd.Series(forecast_prices, index=test_days)

    errors = pd.Series(errors, index=test_days)

    pct_errors = pd.Series(pct_errors, index=test_days)

    mse = np.mean(errors**2)

    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(errors))

    mse_return = np.mean(pct_errors**2)

    rmse_return = np.sqrt(mse_return)

    mae_return = np.mean(np.abs(pct_errors))

    rw_validation[col] = {

        "actual_prices": actual_prices,

        "forecast_prices": forecast_prices,

        "errors": errors,

        "pct_errors": pct_errors,

        "MSE_price": mse,

        "RMSE_price": rmse,

        "MAE_price": mae,

        "MSE_return": mse_return,

        "RMSE_return": rmse_return,

        "MAE_return": mae_return

    }

# Summary table
rw_validation_table = pd.DataFrame({
    col: {
        "MSPE_return": rw_validation[col]["MSE_return"],
        "RMSE_return": rw_validation[col]["RMSE_return"],
        "MAE_return": rw_validation[col]["MAE_return"]
    }
    for col in asset_cols
}).T

print("\nRandom Walk walk-forward validation:\n")

# Print all columns, no dots
with pd.option_context(
    "display.max_columns", None,
    "display.width", 200,
    "display.expand_frame_repr", False
):
    print(rw_validation_table.round(6).to_string())

print("\nRandom Walk walk-forward validation:\n")

print(rw_validation_table.round(6))
#%%
# =========================
# Random Walk backtest plots
# =========================

h = 200
z = 1.96
zoom = 250

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

axes = axes.flatten()

for i, col in enumerate(asset_cols):

    series = df_clean[col].dropna()

    # Split train/test
    train = series.iloc[:-h]
    test = series.iloc[-h:]

    # Last observed train point
    last_price = train.iloc[-1]
    last_day = train.index[-1]

    # Estimate volatility from train only
    log_ret = np.log(train).diff().dropna()
    sigma = log_ret.std()

    # Forecast horizon
    future_days = test.index

    # RW forecast
    forecast_price = np.repeat(last_price, h)

    # Growing uncertainty
    steps = np.arange(1, h + 1)
    forecast_sd = sigma * np.sqrt(steps)

    lower = np.exp(np.log(last_price) - z * forecast_sd)
    upper = np.exp(np.log(last_price) + z * forecast_sd)

    # Plot last observed training data
    axes[i].plot(
        series.iloc[-zoom:].index,
        series.iloc[-zoom:],
        color="black",
        linewidth=2,
        label="True price"
    )

    # Forecast
    axes[i].plot(
        future_days,
        forecast_price,
        color="tab:orange",
        linewidth=2,
        label="RW forecast"
    )

    # CI
    axes[i].fill_between(
        future_days,
        lower,
        upper,
        alpha=0.25,
        color="tab:blue",
        label="95% CI"
    )

    # Vertical split line
    axes[i].axvline(
        last_day,
        linestyle="--",
        color="red",
        alpha=0.7
    )

    axes[i].set_title(f"{col} - Random Walk Backtest")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
# Remove empty last subplot
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()
#%%
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings('ignore', category=InterpolationWarning)
h = 200 # steg framåt

z = 1.96 # ci

def select_arma_order(log_ret, max_p=4, max_q=4):

    best_aic = np.inf 

    best_order = None 

    best_model = None

    for p in range(max_p + 1): # 0 till 4

        for q in range(max_q + 1): # 0 till 4

            try:

                model = ARIMA(

                    log_ret,

                    order=(p, 0, q),

                    trend="c"

                ).fit(method_kwargs={"maxiter": 200})

                if model.aic < best_aic:

                    best_aic = model.aic

                    best_order = (p, q)

                    best_model = model

            except Exception:

                continue

    return best_order, best_model, best_aic

# Phi och Theta fås genom MLE, e_t = r_t - r_t_hat
# Validering av AIC =2k - 2log(L) och residualanalys, dvs. ACF och ACF^2 nära 0, samt normalfördelade
# Kan också använd Ljung - Box
# För ARMA(1,2) \hat r_{T+1} = c + \phi_1 r_T + \theta_1 \hat\varepsilon_T + \theta_2 \hat\varepsilon_{T-1}
# och vidare \hat r_{T+2} = c + \phi_1 \hat r_{T+1} eftersom E(\varepsilon_{T+1})=0
# ARMA forecast går ofta mot den långsiktiga medelreturnen

arma_forecasts = {}

arma_lower = {}

arma_upper = {}

arma_orders = {}

for col in asset_cols:

    series = df_clean[col].dropna()

    last_day = int(series.index[-1])

    last_price = series.iloc[-1]

    last_log_price = np.log(last_price)

    log_ret = np.log(series).diff().dropna()

    log_ret = log_ret.reset_index(drop=True)

    order, model, aic = select_arma_order(log_ret, max_p=4, max_q=4)

    print(f"{col}: ARMA{order}, AIC = {aic:.2f}")

    arma_orders[col] = {
    "order": order,
    "aic": aic,
    "model": model
    }

    fc = model.get_forecast(steps=h)

    mean_ret = np.asarray(fc.predicted_mean)

    se_ret = np.asarray(fc.se_mean)

    cum_mean_ret = np.cumsum(mean_ret)

    forecast_log_price = last_log_price + cum_mean_ret

    cum_var = np.cumsum(se_ret**2)

    cum_se = np.sqrt(cum_var)

    lower_log = forecast_log_price - z * cum_se

    upper_log = forecast_log_price + z * cum_se

    future_days = np.arange(last_day + 1, last_day + h + 1)

    arma_forecasts[col] = pd.Series(np.exp(forecast_log_price), index=future_days)

    arma_lower[col] = pd.Series(np.exp(lower_log), index=future_days)

    arma_upper[col] = pd.Series(np.exp(upper_log), index=future_days)

arma_forecast_df_clean = pd.DataFrame(arma_forecasts)

arma_lower_df_clean = pd.DataFrame(arma_lower)

arma_upper_df_clean = pd.DataFrame(arma_upper)
# fixa summan av alla log-return och exponenera osv. 
# för confidence interval: \widehat{\log(P_{T+h})} \pm 1.96 \cdot SE_{\log(P_{T+h})}
# sen \widehat{\log(P_{T+h})} \pm 1.96 \cdot SE_{\log(P_{T+h})} och exponentiera
# ingen trend ochs easonality efterom log-returns har mean 0 och ACF nära 0
# med Prices hade det vart bättre med ARIMA och SARIMA

# =========================
# ARMA model summaries
# =========================

# =========================
# Print ARMA orders, AIC and parameters
# =========================

arma_summary_rows = []

for col in asset_cols:
    model = arma_orders[col]["model"]

    row = {
        "Asset": col,
        "ARMA Order": arma_orders[col]["order"],
        "AIC": arma_orders[col]["aic"]
    }

    for name, value in model.params.items():
        row[name] = value

    arma_summary_rows.append(row)

arma_summary_df = pd.DataFrame(arma_summary_rows)

print("\nARMA summary:\n")
print(arma_summary_df.round(6))
#%%
# =========================
# ARMA backtest plots
# True future path shown
# 4x2 layout
# =========================

h = 200
z = 1.96
zoom = 250

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

axes = axes.flatten()

for i, col in enumerate(asset_cols):

    series = df_clean[col].dropna()

    # split train/test
    train = series.iloc[:-h]
    test = series.iloc[-h:]

    order = arma_orders[col]["order"]

    # log-returns from training only
    log_ret = np.log(train).diff().dropna()
    log_ret = log_ret.reset_index(drop=True)

    # fit ARMA
    model = ARIMA(
        log_ret,
        order=(order[0], 0, order[1]),
        trend="c"
    ).fit(method_kwargs={"maxiter": 200})

    # forecast returns
    fc = model.get_forecast(steps=h)

    mean_ret = np.asarray(fc.predicted_mean)
    se_ret = np.asarray(fc.se_mean)

    # convert back to prices
    last_price = train.iloc[-1]
    last_log_price = np.log(last_price)

    cum_mean_ret = np.cumsum(mean_ret)

    forecast_log_price = last_log_price + cum_mean_ret

    cum_var = np.cumsum(se_ret**2)
    cum_se = np.sqrt(cum_var)

    lower_log = forecast_log_price - z * cum_se
    upper_log = forecast_log_price + z * cum_se

    forecast_price = np.exp(forecast_log_price)
    lower_price = np.exp(lower_log)
    upper_price = np.exp(upper_log)

    # show last observations including true future
    observed_zoom = series.iloc[-zoom:]

    # true prices
    axes[i].plot(
        observed_zoom.index,
        observed_zoom,
        color="black",
        linewidth=2,
        label="True price"
    )

    # forecast
    axes[i].plot(
        test.index,
        forecast_price,
        color="tab:orange",
        linewidth=2,
        label=f"ARMA{order} forecast"
    )

    # confidence interval
    axes[i].fill_between(
        test.index,
        lower_price,
        upper_price,
        alpha=0.25,
        color="tab:blue",
        label="95% CI"
    )

    # split line train/test
    axes[i].axvline(
        train.index[-1],
        linestyle="--",
        color="red",
        alpha=0.7
    )

    axes[i].set_title(f"{col} - ARMA{order} Backtest")
    axes[i].set_xlabel("Day")
    axes[i].set_ylabel("Price")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

# remove unused subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
