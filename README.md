# ADF_Stationarity_Test — Augmented Dickey-Fuller Unit Root Test for IBM SPSS Statistics

## Overview
**ADF_Stationarity_Test** is a native Python 3 extension for IBM SPSS Statistics (v28+) that performs the Augmented Dickey-Fuller (ADF) test for a unit root in univariate time series.

**Menu Location:** `Analyze -> Forecasting -> Augmented Dickey-Fuller (ADF) Test`

---

## Key Econometric Features
1. **Deterministic Specifications:**
   * `None (n)`: Pure random walk without drift ($\Delta y_t = \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$)
   * `Constant only (c)`: Random walk with drift
   * `Constant + linear trend (ct)`: Trend-stationary alternative
   * `Constant + linear + quadratic trend (ctt)`: Parabolic trend alternative
2. **Lag Order Selection & Small-Sample Protection:**
   * Automatic lag selection via **AIC**, **BIC/SIC**, or **Sequential $t$-statistic** on a unified sample window ($T - p_{\max} - 1$), or user-specified **Fixed Lag**.
   * Automatic small-sample cap via **Schwert's (1989) rule** ($p_{\text{Schwert}} = \lfloor 12 (T/100)^{1/4} \rfloor$).
   * Reports **Ng & Perron (2001) Modified AIC (MAIC)** across all candidate lags to diagnose size-power tradeoffs under volatility clustering or negative MA errors.
3. **Auxiliary Regression & Heteroskedasticity Diagnostics:**
   * Reports ordinary OLS estimates alongside **MacKinnon-White HC3 heteroskedasticity-robust $t$-statistics**.
   * Reports **Ljung-Box $Q$** with exact autoregressive degrees-of-freedom adjustment ($\text{df} = h - p$), **Durbin-Watson**, **Jarque-Bera** normality test, and **Engle's ARCH(1) LM test** for conditional volatility clustering.
4. **Time-Series Integrity Guardrails:**
   * Blocks execution if `SPLIT FILE` is active to prevent concatenating panel groups into a single series.
   * Separately audits benign leading/trailing missing values versus internal time-series gaps.
5. **Four Diagnostic Plots:**
   * Tested Series Plot
   * Information Criterion (AIC & BIC/SIC) vs. Lag Order
   * Auxiliary Regression Residuals
   * Residual Autocorrelation Function (ACF Lags $1 \dots K$ with 95% Bartlett bands)

---

## Benchmark Verification (IBM SPSS Sample Dataset: `stocks.sav`)
Tested on `High` ($N = 251$, Deterministic terms = `Constant + linear trend (ct)`, Lag selection = `AIC`, Max lag = `13`)[cite: 20, 21]:
* **ADF Test Statistic:** `-1.987` ($p = 0.6087$, Lags Used = `13`, Effective $N = 237$) — *Fail to reject $H_0$ at $\alpha = 0.05$*[cite: 20]
* **MacKinnon Critical Values:** 1%: `-3.997` | 5%: `-3.429` | 10%: `-3.138`[cite: 20]
* **Lag Selection Comparison (Common Sample $N = 237$):** AIC minimum at Lag 13 (`1331.104`) | BIC minimum at Lag 1 (`1352.514`) | Ng-Perron MAIC minimum at Lag 6 (`3.297`)[cite: 20, 21]
* **Auxiliary Regression Diagnostics:** Lagged level $\hat{\gamma} = -0.031895$ ($\text{OLS } t = -1.987$, $\text{HC3 Robust } t = -1.877$) | Durbin-Watson = `2.023` | Ljung-Box $Q(14) = 2.867$ ($df = 1, p = 0.0904$) | Engle ARCH(1) LM = `1.772` ($p = 0.1831$)[cite: 22, 25]

---

## Requirements
* IBM SPSS Statistics 28 or later (tested on SPSS Statistics 32 with Python 3.13)
* Python 3 packages: `numpy`, `scipy`, `statsmodels`, `matplotlib`

## License
Apache License 2.0 — Author: Aneek Sarkar