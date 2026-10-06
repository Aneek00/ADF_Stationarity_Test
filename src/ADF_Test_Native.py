# -*- coding: utf-8 -*-
"""
ADF_Test_Native.py

IBM SPSS Statistics Extension Module: Augmented Dickey-Fuller (ADF) Unit Root Test
Author: Aneek Sarkar
License: Apache 2.0

Supports both:
  1. Custom Dialog Builder (CDB) direct invocation via run_adf_native(...)
  2. Native SPSS XML Command invocation (ADF_STATIONARITY_TEST) via Run(args)
"""

from __future__ import division

import math
import os
import re
import shutil
import site
import sys
import tempfile
import time
import warnings

import numpy as np

import spss
import spssdata

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf

import statsmodels.api as sm

from statsmodels.stats.stattools import (
    durbin_watson,
    jarque_bera
)

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def _clean(value):
    """
    Universally extracts strings from lists/bytes and strips all quote
    variants and whitespace passed by Extension Builder or XML syntax.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        value = " ".join(
            [
                str(v)
                for v in value
            ]
        )
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)

    return str(value).replace("'", "").replace('"', "").strip()


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return len(value) > 0

    s = _clean(value).lower()

    return s not in (
        "",
        "0",
        "false",
        "no",
        "none",
        "(none)",
        "off"
    )


def _safe_float(value, default=np.nan):

    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# GUI VALUE NORMALIZATION
# ============================================================

def _normalize_regression(value):
    """Normalize deterministic-term selection for ADF."""

    val = _clean(value).lower()

    if val == "c":
        return "c"
    if val == "ct":
        return "ct"
    if val == "ctt":
        return "ctt"
    if val == "n":
        return "n"

    if "quadratic" in val:
        return "ctt"
    if "linear" in val and "trend" in val:
        return "ct"
    if "constant" in val and "no" not in val:
        return "c"
    if "no constant" in val or val in ("none", "(none)"):
        return "n"

    raise ValueError(
        "Unrecognized deterministic-term selection: %r"
        % value
    )


def _normalize_lag_method(value):

    s = _clean(value).lower()

    if s in (
        "fixed",
        "fixed lag",
        "none",
        "no"
    ):
        return "fixed"

    if (
        "akaike" in s
        or s == "aic"
    ):
        return "AIC"

    if (
        "bayesian" in s
        or "bic" in s
        or "sic" in s
    ):
        return "BIC"

    if (
        "sequential" in s
        or "t-stat" in s
        or "t statistic" in s
        or "tstat" in s
    ):
        return "t-stat"

    raise ValueError(
        "Unrecognized lag-selection method: %s"
        % value
    )


def _normalize_missing(value):

    s = _clean(value).lower()

    if "stop" in s:
        return "stop"

    return "exclude"


def _normalize_alpha(value):

    s = _clean(value).lower()

    if (
        "1%" in s
        or s in ("1", "0.01", ".01")
    ):
        return 0.01

    if (
        "5%" in s
        or s in ("5", "0.05", ".05")
    ):
        return 0.05

    if (
        "10%" in s
        or s in ("10", "0.10", ".10")
    ):
        return 0.10

    try:

        x = float(value)

        if x > 1:
            x = x / 100.0

        if 0 < x < 1:
            return x

    except Exception:
        pass

    return 0.05


def _parse_custom_alpha(value):

    try:
        x = float(_clean(value))
    except Exception:
        raise ValueError(
            "Custom alpha must be numeric."
        )

    if x > 1:
        x = x / 100.0

    if not (
        0 < x < 1
    ):
        raise ValueError(
            "Custom alpha must be strictly between 0 and 1."
        )

    return x


# ============================================================
# CHECKBOX / ITEM GROUP PARSING
# ============================================================

def _contains_option(value, patterns):

    if isinstance(value, (list, tuple)):

        for item in value:

            if _contains_option(
                item,
                patterns
            ):
                return True

        return False

    s = _clean(value).lower()

    for pattern in patterns:

        if pattern.lower() in s:
            return True

    return False


def _requested_diagnostics(value):

    return {

        "candidate_lag":
            _contains_option(
                value,
                (
                    "candidate lag",
                    "lag table",
                    "item_135"
                )
            ),

        "full_regression":
            _contains_option(
                value,
                (
                    "full auxiliary",
                    "auxiliary regression",
                    "item_136"
                )
            ),

        "regression_diagnostics":
            _contains_option(
                value,
                (
                    "regression diagnostics",
                    "item_137"
                )
            ),

        "validation":
            _contains_option(
                value,
                (
                    "validation",
                    "debug",
                    "item_134"
                )
            )
    }


def _requested_plots(value):

    return {

        "series":
            _contains_option(
                value,
                (
                    "tested series",
                    "tested_series",
                    "item_138"
                )
            ),

        "ic":
            _contains_option(
                value,
                (
                    "information criterion",
                    "information_criterion",
                    "criterion vs lag",
                    "item_139",
                    "item_142"
                )
            ),

        "residuals":
            _contains_option(
                value,
                (
                    "residuals",
                    "item_140"
                )
            ),

        "acf":
            _contains_option(
                value,
                (
                    "residual acf",
                    "residual_acf",
                    "item_141"
                )
            )
    }


# ============================================================
# SPSS VARIABLE & DATASET STATE VALIDATION
# ============================================================

def _check_spss_dataset_state():
    """
    Validates active dataset state for time-series integrity:
      1. Rejects active SPLIT FILE (prevents concatenating panel groups).
      2. Detects active WEIGHT variable (warns that ADF is unweighted).
    """
    try:
        split_vars = spss.GetSplitVariableNames()
    except Exception:
        split_vars = []

    if split_vars:
        raise ValueError(
            "SPLIT FILE is currently active on variable(s): %s. "
            "The Augmented Dickey-Fuller test requires a single contiguous "
            "time series and cannot concatenate split groups. "
            "Please disable Split File (Data -> Split File -> Analyze all cases) "
            "or isolate a single group via Data -> Select Cases."
            % ", ".join(split_vars)
        )

    weight_var = ""
    try:
        w = spss.GetWeightVar()
        if w:
            weight_var = str(w).strip()
    except Exception:
        weight_var = ""

    return weight_var


def _find_variable_index(name):

    target = _clean(name).lower()
    available = []

    for i in range(
        spss.GetVariableCount()
    ):

        current = (
            spss.GetVariableName(i)
        )
        available.append(current)

        if current.lower() == target:
            return i

    preview = (
        ", ".join(available[:8])
        if available
        else "(dataset is empty)"
    )

    raise ValueError(
        "Variable '%s' was not found "
        "in the active dataset. "
        "Active dataset variables: %s"
        % (name, preview)
    )


def _assert_numeric_variable(
    name,
    role
):

    index = _find_variable_index(name)

    variable_type = (
        spss.GetVariableType(index)
    )

    if variable_type != 0:

        raise ValueError(
            "%s '%s' must be numeric. "
            "ADF cannot be calculated on "
            "a string variable."
            % (
                role,
                name
            )
        )


# ============================================================
# DATA EXTRACTION & TIME-SERIES GAP AUDIT
# ============================================================

def _is_missing(value):

    if value is None:
        return True

    try:
        return not np.isfinite(
            float(value)
        )

    except Exception:
        return True


def _read_data(
    target_var,
    time_var=None
):

    _assert_numeric_variable(
        target_var,
        "Series to Test"
    )

    if time_var:

        _assert_numeric_variable(
            time_var,
            "Time/Order Variable"
        )

        varlist = (
            target_var
            + " "
            + time_var
        )

    else:

        varlist = target_var

    cursor = spssdata.Spssdata(
        varlist,
        omitmissing=False
    )

    rows = []

    try:

        for row in cursor:

            if time_var:

                rows.append(
                    (
                        row[0],
                        row[1]
                    )
                )

            else:

                rows.append(
                    (
                        row[0],
                        None
                    )
                )

    finally:

        try:
            cursor.CClose()
        except Exception:
            pass

    return rows


def _prepare_series(
    target_var,
    time_var,
    missing_method
):
    """
    Extracts the series and distinguishes benign leading/trailing missing
    observations from internal time-series gaps (which bridge non-adjacent
    periods when differencing).
    """

    weight_var = _check_spss_dataset_state()

    rows = _read_data(
        target_var,
        time_var
    )

    n_raw = len(rows)

    if n_raw == 0:
        raise ValueError(
            "The active dataset contains 0 observations."
        )

    # If a time variable is provided, sort rows with valid time values first;
    # any row with a missing time index is counted as missing.
    valid_flags = []
    for y, t in rows:
        if _is_missing(y) or (time_var and _is_missing(t)):
            valid_flags.append(False)
        else:
            valid_flags.append(True)

    total_missing = sum(1 for flag in valid_flags if not flag)

    if (
        missing_method == "stop"
        and total_missing > 0
    ):

        raise ValueError(
            "Missing values were found "
            "in the selected series/time variable. "
            "The dialog is set to 'Stop if missing "
            "values are present'. Missing cases: %d."
            % total_missing
        )

    # Identify leading/trailing missing vs internal gaps in case order
    valid_indices = [
        idx
        for idx, flag in enumerate(valid_flags)
        if flag
    ]

    if not valid_indices:
        raise ValueError(
            "All observations in '%s' are missing."
            % target_var
        )

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    leading_trailing_missing = (
        first_valid
        + (n_raw - 1 - last_valid)
    )

    internal_missing = (
        total_missing
        - leading_trailing_missing
    )

    clean = []

    for y, t in rows:

        if _is_missing(y):
            continue

        if (
            time_var
            and _is_missing(t)
        ):
            continue

        if time_var:

            clean.append(
                (
                    float(y),
                    float(t)
                )
            )

        else:

            clean.append(
                (
                    float(y),
                    None
                )
            )

    duplicate_time_count = 0

    if time_var:

        clean.sort(
            key=lambda z: z[1]
        )

        times = [
            z[1]
            for z in clean
        ]

        duplicate_time_count = (
            len(times)
            - len(set(times))
        )

    y = np.asarray(
        [
            z[0]
            for z in clean
        ],
        dtype=float
    )

    if y.size < 10:

        raise ValueError(
            "Too few usable observations "
            "for a reliable ADF test. "
            "At least 10 nonmissing observations "
            "are required (found %d)."
            % int(y.size)
        )

    if not np.all(
        np.isfinite(y)
    ):

        raise ValueError(
            "The selected series contains "
            "non-finite values."
        )

    if (
        np.max(y)
        == np.min(y)
    ):

        raise ValueError(
            "The selected series is constant. "
            "The ADF test cannot be computed "
            "for a constant series."
        )

    return {

        "y": y,

        "n_raw":
            n_raw,

        "n_used":
            int(y.size),

        "missing_count":
            total_missing,

        "leading_trailing_missing":
            int(leading_trailing_missing),

        "internal_missing":
            int(internal_missing),

        "duplicate_time_count":
            duplicate_time_count,

        "sorted":
            bool(time_var),

        "weight_var":
            weight_var
    }


# ============================================================
# ADF AUXILIARY REGRESSION
# ============================================================

def _build_adf_design(
    y,
    lag,
    regression
):
    """
    Constructs the exact ADF design matrix where:
      Column 0         = Lagged level (y_{t-1})
      Columns 1..lag   = Lagged differences (Delta y_{t-1} .. Delta y_{t-lag})
      Trailing columns = Deterministic terms (Constant, Linear trend, Quadratic trend)
    """

    y = np.asarray(
        y,
        dtype=float
    )

    dy = np.diff(y)

    if lag < 0:

        raise ValueError(
            "Lag order cannot be negative."
        )

    if (
        y.size
        - lag
        - 1
        <= 0
    ):

        raise ValueError(
            "Not enough observations "
            "for lag %d."
            % lag
        )

    endog = dy[lag:]

    level = y[
        lag:-1
    ]

    cols = [
        level
    ]

    labels = [
        "Lagged level"
    ]

    for j in range(
        1,
        lag + 1
    ):

        cols.append(
            dy[
                lag - j:
                -j
            ]
        )

        labels.append(
            "Lagged difference %d"
            % j
        )

    nobs = len(endog)

    if regression in (
        "c",
        "ct",
        "ctt"
    ):

        cols.append(
            np.ones(nobs)
        )

        labels.append(
            "Constant"
        )

    if regression in (
        "ct",
        "ctt"
    ):

        trend = np.arange(
            1,
            nobs + 1,
            dtype=float
        )

        cols.append(
            trend
        )

        labels.append(
            "Linear trend"
        )

    if regression == "ctt":

        cols.append(
            trend ** 2
        )

        labels.append(
            "Quadratic trend"
        )

    exog = np.column_stack(
        cols
    )

    return (
        endog,
        exog,
        labels
    )


def _fit_auxiliary(
    y,
    lag,
    regression,
    common_max_lag=None
):
    """
    Fits the OLS auxiliary regression.
    When common_max_lag is supplied, trims initial observations so all
    candidate lags (0..max_lag) are evaluated on the exact same sample
    window (T - max_lag - 1) used by statsmodels autolag search, while
    guaranteeing Column 0 is Lagged level and Column `lag` is Last Lag.
    """

    endog, exog, labels = (
        _build_adf_design(
            y,
            lag,
            regression
        )
    )

    if (
        common_max_lag is not None
        and common_max_lag > lag
    ):

        trim = int(
            common_max_lag - lag
        )

        if trim < len(endog):

            endog = endog[trim:]
            exog = exog[trim:, :]
            nobs = len(endog)

            if regression in (
                "ct",
                "ctt"
            ):

                trend_col = (
                    lag + 2
                )

                trend = np.arange(
                    1,
                    nobs + 1,
                    dtype=float
                )

                exog[:, trend_col] = trend

                if regression == "ctt":

                    exog[:, trend_col + 1] = (
                        trend ** 2
                    )

    model = sm.OLS(
        endog,
        exog
    ).fit()

    return {

        "model":
            model,

        "endog":
            endog,

        "exog":
            exog,

        "labels":
            labels
    }


# ============================================================
# ADF ENGINE
# ============================================================

def _run_adf(
    y,
    regression,
    lag_method,
    max_lag,
    fixed_lag
):

    if max_lag < 0:

        raise ValueError(
            "Maximum Lag must be >= 0."
        )

    if fixed_lag < 0:

        raise ValueError(
            "Fixed Lag must be >= 0."
        )

    n = len(y)

    ntrend = (
        0
        if regression == "n"
        else len(regression)
    )

    admissible_max = (
        n // 2
        - ntrend
        - 1
    )

    if admissible_max < 0:

        raise ValueError(
            "The sample is too short "
            "for the selected deterministic terms."
        )

    if max_lag > admissible_max:

        schwert_lag = int(
            12.0
            * (
                (n / 100.0) ** 0.25
            )
        )

        max_lag = max(
            0,
            min(
                schwert_lag,
                admissible_max
            )
        )

    if lag_method == "fixed":

        if fixed_lag > admissible_max:

            raise ValueError(
                "Fixed Lag (%d) cannot exceed "
                "admissible Maximum Lag (%d)."
                % (
                    fixed_lag,
                    admissible_max
                )
            )

        result = adfuller(
            y,
            maxlag=fixed_lag,
            regression=regression,
            autolag=None,
            store=True,
            regresults=True
        )

    else:

        result = adfuller(
            y,
            maxlag=max_lag,
            regression=regression,
            autolag=lag_method,
            store=True,
            regresults=True
        )

    adf_stat = None
    pvalue = None
    usedlag = None
    nobs = None
    critical_values = None
    icbest = np.nan
    store = None

    if len(result) == 4:

        (
            adf_stat,
            pvalue,
            critical_values,
            store
        ) = result

        usedlag = int(
            store.usedlag
        )

        nobs = int(
            store.nobs
        )

        raw_ic = getattr(
            store,
            "icbest",
            np.nan
        )

        try:

            icbest = float(
                raw_ic
            )

        except Exception:

            icbest = np.nan

    else:

        adf_stat = result[0]
        pvalue = result[1]
        usedlag = result[2]
        nobs = result[3]
        critical_values = result[4]

        if len(result) > 5:

            try:
                icbest = float(
                    result[5]
                )
            except Exception:
                icbest = np.nan

        store = result[-1]

    try:

        if not np.isfinite(
            float(icbest)
        ):

            icbest = np.nan

    except Exception:

        icbest = np.nan

    return {

        "adf_stat":
            float(adf_stat),

        "pvalue":
            float(pvalue),

        "usedlag":
            int(usedlag),

        "nobs":
            int(nobs),

        "critical_values":
            critical_values,

        "icbest":
            icbest,

        "store":
            store,

        "resols":
            store.resols,

        "effective_max_lag":
            int(max_lag)
    }


# ============================================================
# CANDIDATE LAG TABLE (FIXED INDEXING + NG-PERRON MAIC)
# ============================================================

def _candidate_lag_table(
    y,
    max_lag,
    regression,
    result=None
):
    """
    Evaluates lags 0..max_lag using _fit_auxiliary on the common sample
    window (T - max_lag - 1).
    Fixes the statsmodels store.autolag_results column-shift bug and
    computes Ng & Perron (2001) Modified AIC (MAIC).
    """

    rows = []

    for lag in range(
        0,
        max_lag + 1
    ):

        try:

            fit = _fit_auxiliary(
                y,
                lag,
                regression,
                common_max_lag=max_lag
            )

            model = fit["model"]
            exog = fit["exog"]

            # Column 0 is strictly Lagged level (ADF t-statistic)
            adf_t = float(
                model.tvalues[0]
            )

            # Column `lag` is strictly the last lagged difference (Delta y_{t-lag})
            last_lag_t = np.nan

            if lag >= 1:

                try:

                    last_lag_t = float(
                        model.tvalues[lag]
                    )

                except Exception:

                    last_lag_t = np.nan

            # Ng & Perron (2001) Modified AIC (MAIC)
            try:
                t_eff = float(model.nobs)
                sigma2_k = float(model.ssr) / t_eff
                gamma_hat = float(model.params[0])
                sum_y_lag_sq = float(np.sum(exog[:, 0] ** 2))
                tau_t = (gamma_hat ** 2) * sum_y_lag_sq / sigma2_k
                maic = math.log(sigma2_k) + (2.0 * (tau_t + lag)) / t_eff
            except Exception:
                maic = np.nan

            rows.append({

                "lag":
                    lag,

                "aic":
                    float(model.aic),

                "bic":
                    float(model.bic),

                "maic":
                    float(maic),

                "adf_t":
                    adf_t,

                "last_lag_t":
                    last_lag_t,

                "nobs":
                    int(model.nobs),

                "selected":
                    (
                        result is not None
                        and lag
                        == int(
                            result["usedlag"]
                        )
                    )
            })

        except Exception:

            rows.append({

                "lag":
                    lag,

                "aic":
                    np.nan,

                "bic":
                    np.nan,

                "maic":
                    np.nan,

                "adf_t":
                    np.nan,

                "last_lag_t":
                    np.nan,

                "nobs":
                    np.nan,

                "selected":
                    False
            })

    return rows


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def _fmt(
    x,
    digits=6
):

    if x is None:
        return ""

    try:

        x = float(x)

        if not np.isfinite(x):
            return ""

        return (
            "%.*f"
            % (
                digits,
                x
            )
        ).rstrip(
            "0"
        ).rstrip(
            "."
        )

    except Exception:

        return str(x)


def _p_fmt(x):

    try:

        x = float(x)

        if not np.isfinite(x):
            return ""

        if x < 0.0001:
            return "<.0001"

        return "%.4f" % x

    except Exception:

        return ""


def _add_table(
    title,
    rowdim,
    collabels,
    rows
):

    table = spss.BasePivotTable(
        title,
        "ADF_STATIONARITY_TEST"
    )

    table.SimplePivotTable(
        rowdim=rowdim,
        rowlabels=[
            str(r[0])
            for r in rows
        ],
        collabels=collabels,
        cells=[
            v
            for r in rows
            for v in r[1:]
        ]
    )

    return table


def _add_text(
    title,
    content
):

    return spss.TextBlock(
        title,
        content,
        title
    )


# ============================================================
# CONCLUSION (CLEAN ASCII ALPHA)
# ============================================================

def _conclusion(
    adf_stat,
    pvalue,
    alpha,
    critical_values
):

    if pvalue <= alpha:

        return (
            "Reject H0 at alpha = %s. There is "
            "evidence against a unit root; "
            "the series is stationary under "
            "the specified ADF regression."
            % _fmt(
                alpha,
                4
            )
        )

    cv = None

    if abs(
        alpha - 0.01
    ) < 1e-12:

        cv = critical_values.get(
            "1%"
        )

    elif abs(
        alpha - 0.05
    ) < 1e-12:

        cv = critical_values.get(
            "5%"
        )

    elif abs(
        alpha - 0.10
    ) < 1e-12:

        cv = critical_values.get(
            "10%"
        )

    if (
        cv is not None
        and adf_stat < cv
    ):

        return (
            "Reject H0 at alpha = %s because "
            "the ADF statistic is below the "
            "critical value. The series is "
            "stationary under the specified "
            "ADF regression."
            % _fmt(
                alpha,
                4
            )
        )

    return (
        "Fail to reject H0 at alpha = %s. "
        "The test does not provide sufficient "
        "evidence against a unit root; the "
        "series is not established as stationary "
        "under the specified ADF regression."
        % _fmt(
            alpha,
            4
        )
    )


# ============================================================
# COEFFICIENT LABELS
# ============================================================

def _coefficient_labels(
    regression,
    lag
):

    labels = [
        "Lagged level"
    ]

    for j in range(
        1,
        lag + 1
    ):

        labels.append(
            "Lagged difference %d"
            % j
        )

    if regression in (
        "c",
        "ct",
        "ctt"
    ):

        labels.append(
            "Constant"
        )

    if regression in (
        "ct",
        "ctt"
    ):

        labels.append(
            "Linear trend"
        )

    if regression == "ctt":

        labels.append(
            "Quadratic trend"
        )

    return labels


# ============================================================
# SPSS OUTPUT
# ============================================================

def _make_output(
    target_var,
    regression,
    lag_method,
    max_lag,
    fixed_lag,
    alpha,
    custom_alpha_used,
    missing_method,
    time_var,
    data_info,
    result,
    candidate_rows,
    diagnostics,
    plots
):

    spss.StartProcedure(
        "Augmented Dickey-Fuller (ADF) Test",
        "ADF_STATIONARITY_TEST"
    )

    # --------------------------------------------------------
    # ADF TEST STATISTICS
    # --------------------------------------------------------

    _add_table(

        "ADF Test Statistics",

        "Metric",

        ["Value"],

        [

            (
                "Test Statistic",
                _fmt(
                    result["adf_stat"],
                    3
                )
            ),

            (
                "p-value",
                _p_fmt(
                    result["pvalue"]
                )
            ),

            (
                "Lags Used",
                result["usedlag"]
            ),

            (
                "Observations",
                result["nobs"]
            )
        ]
    )

    # --------------------------------------------------------
    # CRITICAL VALUES
    # --------------------------------------------------------

    _add_table(

        "Critical Values",

        "Significance Level",

        ["Threshold"],

        [

            (
                "1%",
                _fmt(
                    result[
                        "critical_values"
                    ]["1%"],
                    3
                )
            ),

            (
                "5%",
                _fmt(
                    result[
                        "critical_values"
                    ]["5%"],
                    3
                )
            ),

            (
                "10%",
                _fmt(
                    result[
                        "critical_values"
                    ]["10%"],
                    3
                )
            )
        ]
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    conclusion = _conclusion(

        result["adf_stat"],

        result["pvalue"],

        alpha,

        result[
            "critical_values"
        ]
    )

    _add_text(

        "ADF Interpretation",

        "Null hypothesis (H0): The series contains "
        "a unit root.\n"

        "Alternative hypothesis (H1): The series "
        "is stationary under the specified "
        "deterministic terms.\n\n"

        + conclusion
    )

    # --------------------------------------------------------
    # SPECIFICATION LABELS
    # --------------------------------------------------------

    reg_label = {

        "c":
            "Constant only (c)",

        "ct":
            "Constant + linear trend (ct)",

        "ctt":
            "Constant + linear + quadratic trend (ctt)",

        "n":
            "None (n)"
    }[regression]

    lag_label = {

        "fixed":
            "Fixed",

        "AIC":
            "Akaike Information Criterion (AIC)",

        "BIC":
            "Bayesian Information Criterion (BIC/SIC)",

        "t-stat":
            "Sequential t-statistic"
    }[lag_method]

    if custom_alpha_used:

        alpha_source = (
            "Custom alpha = %s"
            % _fmt(
                alpha,
                4
            )
        )

    else:

        alpha_source = (
            "Selected = %s"
            % _fmt(
                alpha,
                4
            )
        )

    # --------------------------------------------------------
    # TEST SPECIFICATION (DYNAMIC LAG ROW)
    # --------------------------------------------------------

    spec_rows = [

        (
            "Series",
            target_var
        ),

        (
            "Deterministic terms",
            reg_label
        ),

        (
            "Lag selection",
            lag_label
        )
    ]

    if lag_method == "fixed":

        spec_rows.append(
            (
                "Fixed lag",
                fixed_lag
            )
        )

    else:

        spec_rows.append(
            (
                "Maximum lag",
                max_lag
            )
        )

    spec_rows.extend([

        (
            "Significance",
            alpha_source
        ),

        (
            "Missing-value treatment",
            missing_method
        ),

        (
            "Time/order variable",
            (
                time_var
                if time_var
                else
                "(none; case order used)"
            )
        )
    ])

    if data_info.get("weight_var"):
        spec_rows.append(
            (
                "Case weights note",
                "Active weight (%s) ignored; ADF requires unweighted series"
                % data_info["weight_var"]
            )
        )

    _add_table(

        "Test Specification",

        "Setting",

        ["Value"],

        spec_rows
    )

    # --------------------------------------------------------
    # CANDIDATE LAG TABLE (WITH NG-PERRON MAIC)
    # --------------------------------------------------------

    if (
        diagnostics["candidate_lag"]
        or diagnostics["validation"]
        or plots["ic"]
    ):

        rows = []

        for r in candidate_rows:

            selected = (
                "Yes"
                if r["lag"]
                == result["usedlag"]
                else ""
            )

            rows.append(

                (
                    str(
                        r["lag"]
                    ),

                    _fmt(
                        r["aic"],
                        3
                    ),

                    _fmt(
                        r["bic"],
                        3
                    ),

                    _fmt(
                        r.get("maic", np.nan),
                        3
                    ),

                    _fmt(
                        r["adf_t"],
                        3
                    ),

                    _fmt(
                        r["last_lag_t"],
                        3
                    ),

                    selected
                )
            )

        _add_table(

            "Candidate Lag Selection",

            "Lag",

            [
                "AIC",
                "BIC/SIC",
                "MAIC (Ng-Perron)",
                "ADF t",
                "Last Lag t",
                "Selected"
            ],

            rows
        )

    # --------------------------------------------------------
    # FULL AUXILIARY REGRESSION (COMPACT 6-COLUMN FIT WITH HC3)
    # --------------------------------------------------------

    if diagnostics["full_regression"]:

        model = result["resols"]

        labels = _coefficient_labels(
            regression,
            result["usedlag"]
        )

        try:
            robust_res = model.get_robustcov_results(
                cov_type="HC3"
            )
            hc3_t = robust_res.tvalues
        except Exception:
            hc3_t = [np.nan] * len(labels)

        rows = []

        for i, label in enumerate(
            labels
        ):

            rows.append(

                (
                    label,

                    _fmt(
                        model.params[i],
                        6
                    ),

                    _fmt(
                        model.bse[i],
                        6
                    ),

                    _fmt(
                        model.tvalues[i],
                        3
                    ),

                    _p_fmt(
                        model.pvalues[i]
                    ),

                    _fmt(
                        hc3_t[i],
                        3
                    )
                )
            )

        _add_table(

            "ADF Auxiliary Regression Coefficients",

            "Term",

            [
                "Coefficient",
                "Std. Error",
                "t",
                "Sig.",
                "Robust t (HC3)"
            ],

            rows
        )

        _add_text(

            "ADF Regression Note",

            "The ADF test statistic is the OLS "
            "t-statistic for the coefficient "
            "on the lagged level term. Its "
            "unit-root p-value is not the ordinary OLS "
            "coefficient Sig. value; it uses the "
            "nonstandard Dickey-Fuller distribution. "
            "MacKinnon-White HC3 heteroskedasticity-robust "
            "t-statistics are reported to diagnose whether "
            "volatility clustering distorts lagged-difference inference."
        )

    # --------------------------------------------------------
    # REGRESSION DIAGNOSTICS (WITH LJUNG-BOX DF & ENGLE ARCH)
    # --------------------------------------------------------

    if diagnostics["regression_diagnostics"]:

        model = result["resols"]

        resid = np.asarray(
            model.resid,
            dtype=float
        )

        try:

            dw = float(
                durbin_watson(
                    resid
                )
            )

        except Exception:

            dw = np.nan

        try:

            (
                jb_stat,
                jb_p,
                skew,
                kurt
            ) = jarque_bera(
                resid
            )

        except Exception:

            jb_stat = np.nan
            jb_p = np.nan
            skew = np.nan
            kurt = np.nan

        used_p = int(
            result["usedlag"]
        )

        lb_lag = max(
            used_p + 1,
            min(
                10,
                max(
                    1,
                    len(resid) // 5
                )
            )
        )

        lb_df_val = (
            lb_lag - used_p
        )

        try:

            try:

                lb_df = acorr_ljungbox(
                    resid,
                    lags=[lb_lag],
                    model_df=used_p,
                    return_df=True
                )

                lb_stat = float(
                    lb_df["lb_stat"].iloc[0]
                )

                lb_p = float(
                    lb_df["lb_pvalue"].iloc[0]
                )

            except Exception:

                lb = acorr_ljungbox(
                    resid,
                    lags=[lb_lag],
                    model_df=used_p,
                    return_df=False
                )

                lb_stat = float(
                    lb[0][0]
                )

                lb_p = float(
                    lb[1][0]
                )

        except Exception:

            lb_stat = np.nan
            lb_p = np.nan

        # Engle's ARCH(1) LM Test for conditional heteroskedasticity
        try:
            arch_lm_stat, arch_lm_p, _, _ = het_arch(
                resid,
                nlags=1
            )
            arch_lm_stat = float(arch_lm_stat)
            arch_lm_p = float(arch_lm_p)
        except Exception:
            arch_lm_stat = np.nan
            arch_lm_p = np.nan

        _add_table(

            "ADF Regression Diagnostics",

            "Metric",

            ["Value"],

            [

                (
                    "R-squared",
                    _fmt(
                        model.rsquared,
                        6
                    )
                ),

                (
                    "Adjusted R-squared",
                    _fmt(
                        model.rsquared_adj,
                        6
                    )
                ),

                (
                    "AIC",
                    _fmt(
                        model.aic,
                        3
                    )
                ),

                (
                    "BIC/SIC",
                    _fmt(
                        model.bic,
                        3
                    )
                ),

                (
                    "Standard error of regression",
                    _fmt(
                        np.sqrt(
                            model.ssr
                            / model.df_resid
                        ),
                        6
                    )
                ),

                (
                    "Durbin-Watson",
                    _fmt(
                        dw,
                        3
                    )
                ),

                (
                    "Residual mean",
                    _fmt(
                        np.mean(resid),
                        6
                    )
                ),

                (
                    "Residual SD",
                    _fmt(
                        np.std(
                            resid,
                            ddof=1
                        ),
                        6
                    )
                ),

                (
                    "Jarque-Bera",
                    _fmt(
                        jb_stat,
                        3
                    )
                ),

                (
                    "JB p-value",
                    _p_fmt(
                        jb_p
                    )
                ),

                (
                    "Ljung-Box lag (h)",
                    lb_lag
                ),

                (
                    "Ljung-Box df (h - p)",
                    lb_df_val
                ),

                (
                    "Ljung-Box Q",
                    _fmt(
                        lb_stat,
                        3
                    )
                ),

                (
                    "Ljung-Box p-value",
                    _p_fmt(
                        lb_p
                    )
                ),

                (
                    "Engle ARCH(1) LM stat",
                    _fmt(
                        arch_lm_stat,
                        3
                    )
                ),

                (
                    "Engle ARCH(1) p-value",
                    _p_fmt(
                        arch_lm_p
                    )
                )
            ]
        )

    # --------------------------------------------------------
    # VALIDATION / DEBUG (WITH INTERNAL GAP AUDIT)
    # --------------------------------------------------------

    if diagnostics["validation"]:

        _add_table(

            "Validation / Debug Information",

            "Metric",

            ["Value"],

            [

                (
                    "Raw cases read",
                    data_info["n_raw"]
                ),

                (
                    "Total missing cases excluded",
                    data_info[
                        "missing_count"
                    ]
                ),

                (
                    "Leading/trailing missing trimmed",
                    data_info.get(
                        "leading_trailing_missing",
                        0
                    )
                ),

                (
                    "Internal time-series gaps bridged",
                    data_info.get(
                        "internal_missing",
                        0
                    )
                ),

                (
                    "Usable observations",
                    data_info["n_used"]
                ),

                (
                    "Cases sorted by time",
                    (
                        "Yes"
                        if data_info["sorted"]
                        else
                        "No"
                    )
                ),

                (
                    "Duplicate time values",
                    data_info[
                        "duplicate_time_count"
                    ]
                ),

                (
                    "Final ADF regression observations",
                    result["nobs"]
                ),

                (
                    "Final lag used",
                    result["usedlag"]
                ),

                (
                    "Final ADF statistic",
                    _fmt(
                        result[
                            "adf_stat"
                        ],
                        8
                    )
                ),

                (
                    "Final p-value",
                    _p_fmt(
                        result["pvalue"]
                    )
                ),

                (
                    "Information criterion at selected lag",
                    (
                        _fmt(
                            result["icbest"],
                            6
                        )
                        if np.isfinite(
                            result["icbest"]
                        )
                        else
                        "Not applicable"
                    )
                )
            ]
        )

    spss.EndProcedure()


# ============================================================
# PLOT GENERATION (600x340px NO-CLIP + LAG-0 SLICED ACF)
# ============================================================

def _cleanup_old_temp_dirs(max_age_seconds=3600):
    """Silently cleans stale ADF_* plot temp folders older than 1 hour."""
    try:
        temp_root = tempfile.gettempdir()
        now = time.time()
        for name in os.listdir(temp_root):
            if name.startswith("ADF_"):
                full_path = os.path.join(temp_root, name)
                if os.path.isdir(full_path):
                    if now - os.path.getmtime(full_path) > max_age_seconds:
                        shutil.rmtree(full_path, ignore_errors=True)
    except Exception:
        pass


def _save_plots(
    y,
    result,
    candidate_rows,
    plot_flags,
    target_var
):

    try:

        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)

        import matplotlib

        matplotlib.use(
            "Agg"
        )

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

    except Exception as exc:

        return [], (
            "Matplotlib could not be imported: %s"
            % exc
        )

    _cleanup_old_temp_dirs()

    paths = []

    base = tempfile.mkdtemp(
        prefix="ADF_"
    )

    # 6.0 x 3.4 inches at 100 DPI = 600x340px (fits SPSS Viewer & A4 PDF margins without clipping)
    FIG_SIZE = (6.0, 3.4)
    FIG_DPI = 100

    def save_plot(
        fig,
        filename
    ):

        path = os.path.join(
            base,
            filename
        )

        fig.tight_layout(
            pad=1.1
        )

        fig.savefig(
            path,
            dpi=FIG_DPI
        )

        plt.close(
            fig
        )

        paths.append(
            path
        )

    # --------------------------------------------------------
    # TESTED SERIES
    # --------------------------------------------------------

    if plot_flags["series"]:

        fig = plt.figure(
            figsize=FIG_SIZE
        )

        ax = fig.add_subplot(
            111
        )

        ax.plot(
            np.arange(
                1,
                len(y) + 1
            ),
            y,
            linewidth=1.3,
            color="#1f77b4"
        )

        ax.set_title(
            "ADF Tested Series: %s"
            % target_var,
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Observation",
            fontsize=9
        )

        ax.set_ylabel(
            target_var,
            fontsize=9
        )

        ax.tick_params(
            labelsize=8
        )

        ax.grid(
            True,
            alpha=0.25
        )

        save_plot(
            fig,
            "tested_series.png"
        )

    # --------------------------------------------------------
    # INFORMATION CRITERION VS LAG
    # --------------------------------------------------------

    if plot_flags["ic"]:

        valid = []

        for row in candidate_rows:

            try:

                aic_value = float(
                    row["aic"]
                )

                if np.isfinite(
                    aic_value
                ):

                    valid.append(
                        row
                    )

            except Exception:

                pass

        if valid:

            lags = [
                r["lag"]
                for r in valid
            ]

            aic_values = [
                r["aic"]
                for r in valid
            ]

            bic_values = [
                r["bic"]
                for r in valid
            ]

            fig = plt.figure(
                figsize=FIG_SIZE
            )

            ax = fig.add_subplot(
                111
            )

            ax.plot(
                lags,
                aic_values,
                marker="o",
                markersize=4,
                linewidth=1.3,
                label="AIC"
            )

            ax.plot(
                lags,
                bic_values,
                marker="s",
                markersize=4,
                linewidth=1.3,
                label="BIC/SIC"
            )

            ax.axvline(
                result["usedlag"],
                linestyle="--",
                color="#2ca02c",
                linewidth=1.2,
                label="Selected lag (%d)"
                % result["usedlag"]
            )

            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )

            ax.set_title(
                "Information Criterion by Lag",
                fontsize=11,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Lag Order (p)",
                fontsize=9
            )

            ax.set_ylabel(
                "Criterion (lower is better)",
                fontsize=9
            )

            ax.tick_params(
                labelsize=8
            )

            ax.legend(
                fontsize=8,
                loc="best"
            )

            ax.grid(
                True,
                alpha=0.25
            )

            save_plot(
                fig,
                "information_criterion_vs_lag.png"
            )

    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    resid = np.asarray(
        result["resols"].resid,
        dtype=float
    )

    if plot_flags["residuals"]:

        fig = plt.figure(
            figsize=FIG_SIZE
        )

        ax = fig.add_subplot(
            111
        )

        ax.plot(
            np.arange(
                1,
                len(resid) + 1
            ),
            resid,
            linewidth=1.0,
            color="#1f77b4"
        )

        ax.axhline(
            0,
            linestyle="--",
            color="#d62728",
            linewidth=1.0
        )

        ax.set_title(
            "ADF Auxiliary Regression Residuals",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Regression Observation",
            fontsize=9
        )

        ax.set_ylabel(
            "Residual",
            fontsize=9
        )

        ax.tick_params(
            labelsize=8
        )

        ax.grid(
            True,
            alpha=0.25
        )

        save_plot(
            fig,
            "residuals.png"
        )

    # --------------------------------------------------------
    # RESIDUAL ACF (LAG 0 SLICED FOR TRUE CORRELOGRAM SCALE)
    # --------------------------------------------------------

    if plot_flags["acf"]:

        if len(resid) >= 5:

            nlags = min(
                24,
                max(
                    1,
                    len(resid) // 2 - 1
                )
            )

            values = acf(
                resid,
                nlags=nlags,
                fft=True
            )

            confidence = (
                1.96
                / math.sqrt(
                    len(resid)
                )
            )

            # Slice off Lag 0 (r_0 = 1.0) so lags 1..nlags scale clearly
            lags_acf = np.arange(
                1,
                len(values)
            )

            acf_vals = values[1:]

            max_abs = max(
                float(np.max(np.abs(acf_vals)))
                if len(acf_vals) > 0
                else 0.2,
                confidence
            )

            y_bound = min(
                1.0,
                max(
                    0.25,
                    max_abs * 1.35
                )
            )

            fig = plt.figure(
                figsize=FIG_SIZE
            )

            ax = fig.add_subplot(
                111
            )

            ax.stem(
                lags_acf,
                acf_vals
            )

            ax.axhline(
                0,
                linestyle="-",
                color="black",
                linewidth=0.8
            )

            ax.axhline(
                confidence,
                linestyle="--",
                color="#d62728",
                linewidth=1.0,
                label="95% CI"
            )

            ax.axhline(
                -confidence,
                linestyle="--",
                color="#d62728",
                linewidth=1.0
            )

            ax.set_ylim(
                -y_bound,
                y_bound
            )

            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )

            ax.set_title(
                "Residual Autocorrelation Function (Lags 1-%d)"
                % nlags,
                fontsize=11,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Lag",
                fontsize=9
            )

            ax.set_ylabel(
                "ACF",
                fontsize=9
            )

            ax.tick_params(
                labelsize=8
            )

            ax.legend(
                fontsize=8,
                loc="upper right"
            )

            ax.grid(
                True,
                alpha=0.25
            )

            save_plot(
                fig,
                "residual_acf.png"
            )

    return paths, None


# ============================================================
# SPSS VIEWER IMAGE INSERTION
# ============================================================

def _insert_images_into_viewer(
    paths
):

    """
    Insert generated PNG images into the designated SPSS Output Viewer
    after allowing the asynchronous C++ procedure stream to flush.
    """

    if not paths:

        return (
            False,
            "No plot files were generated."
        )

    # Allow spss.EndProcedure() stream to flush all pivot tables to Viewer UI
    # so plots always attach strictly at the bottom of the output tree.
    time.sleep(0.45)

    import SpssClient

    client_started = False

    try:

        SpssClient.StartClient()

        client_started = True

        output_doc = (
            SpssClient.GetDesignatedOutputDoc()
        )

        if output_doc is None:

            output_doc = (
                SpssClient.NewOutputDoc()
            )

            output_doc.SetAsDesignatedOutputDoc()

        output_items = (
            output_doc.GetOutputItems()
        )

        if (
            output_items is None
            or output_items.Size() == 0
        ):

            output_doc = (
                SpssClient.NewOutputDoc()
            )

            output_doc.SetAsDesignatedOutputDoc()

            output_items = (
                output_doc.GetOutputItems()
            )

        if output_items.Size() == 0:

            raise RuntimeError(
                "SPSS Output Viewer does not "
                "contain a root output item."
            )

        root_output_item = (
            output_items.GetItemAt(0)
        )

        root_header = (
            root_output_item.GetSpecificType()
        )

        if not hasattr(
            root_header,
            "InsertChildItem"
        ):

            raise RuntimeError(
                "The first SPSS output item "
                "is not a valid header item."
            )

        plot_header_output_item = (
            output_doc.CreateHeaderItem(
                "ADF Diagnostic Plots"
            )
        )

        root_index = (
            root_header.GetChildCount()
        )

        root_header.InsertChildItem(
            plot_header_output_item,
            root_index
        )

        plot_header = (
            root_header.GetChildItem(
                root_header.GetChildCount() - 1
            ).GetSpecificType()
        )

        inserted = 0

        for path in paths:

            if not os.path.exists(path):

                continue

            spss_path = path.replace("\\", "/")

            filename = os.path.basename(
                path
            )

            label = filename

            image_item = (
                output_doc.CreateImageChartItem(
                    spss_path,
                    label
                )
            )

            plot_header.InsertChildItem(
                image_item,
                plot_header.GetChildCount()
            )

            try:
                image_item.SetVisible(True)
            except Exception:
                pass

            inserted += 1

        try:

            output_ui = (
                output_doc.GetOutputUI()
            )

            if output_ui is not None:

                output_ui.SetVisible(
                    True
                )

        except Exception:

            pass

        if inserted == 0:

            raise RuntimeError(
                "Plot files were generated, "
                "but none could be inserted "
                "into the SPSS Output Viewer."
            )

        return (
            True,
            "%d diagnostic plot(s) inserted."
            % inserted
        )

    except Exception as exc:

        return (
            False,
            "SPSS Viewer plot insertion failed: %s"
            % exc
        )

    finally:

        if client_started:

            try:

                SpssClient.StopClient()

            except Exception:

                pass


# ============================================================
# PLOT STATUS OUTPUT (SILENT ON SUCCESS, LOGS ONLY ON ERROR)
# ============================================================

def _write_plot_status(
    status_message,
    error=False
):

    if not error:
        return

    try:

        spss.StartProcedure(
            "ADF Diagnostic Plot Status",
            "ADF_PLOT_STATUS"
        )

        _add_text(
            "Plot Status",
            "Diagnostic plots could not be "
            "inserted into the SPSS Viewer.\n\n"
            + status_message
        )

        spss.EndProcedure()

    except Exception:

        pass


# ============================================================
# MAIN EXTENSION ENTRY POINT (CDB + DIRECT PYTHON)
# ============================================================

def run_adf_native(
    var_name,
    regression_type="c",
    time_var="",
    lag_method="AIC",
    max_lag=13,
    fixed_lag=0,
    alpha="5%",
    custom_alpha=0.05,
    use_custom_alpha="",
    missing_method="exclude",
    diagnostics="",
    plots=""
):

    """
    Main function called by Custom Dialog Builder (CDB) or Python scripts.
    """

    warnings.filterwarnings(
        "ignore"
    )

    target_var = _clean(
        var_name
    )

    time_var = _clean(
        time_var
    )

    if time_var.lower() in (
        "(none)",
        "none"
    ):

        time_var = ""

    if not target_var:

        raise ValueError(
            "A series must be selected."
        )

    regression = (
        _normalize_regression(
            regression_type
        )
    )

    lag_method = (
        _normalize_lag_method(
            lag_method
        )
    )

    missing_method = (
        _normalize_missing(
            missing_method
        )
    )

    try:

        max_lag = int(
            float(_clean(max_lag))
        )

    except Exception:

        raise ValueError(
            "Maximum Lag must be "
            "a nonnegative integer."
        )

    try:

        fixed_lag = int(
            float(_clean(fixed_lag))
        )

    except Exception:

        raise ValueError(
            "Fixed Lag must be "
            "a nonnegative integer."
        )

    custom_enabled = _truthy(
        use_custom_alpha
    )

    if custom_enabled:

        alpha_value = (
            _parse_custom_alpha(
                custom_alpha
            )
        )

    else:

        alpha_value = (
            _normalize_alpha(
                alpha
            )
        )

    diag_flags = (
        _requested_diagnostics(
            diagnostics
        )
    )

    plot_flags = (
        _requested_plots(
            plots
        )
    )

    data_info = (
        _prepare_series(
            target_var,
            time_var if time_var else None,
            missing_method
        )
    )

    y = data_info["y"]

    result = _run_adf(

        y=y,

        regression=regression,

        lag_method=lag_method,

        max_lag=max_lag,

        fixed_lag=fixed_lag
    )

    effective_max_lag = result.get(
        "effective_max_lag",
        max_lag
    )

    candidate_rows = (
        _candidate_lag_table(

            y,

            effective_max_lag,

            regression,

            result=result
        )
    )

    _make_output(

        target_var=target_var,

        regression=regression,

        lag_method=lag_method,

        max_lag=effective_max_lag,

        fixed_lag=fixed_lag,

        alpha=alpha_value,

        custom_alpha_used=custom_enabled,

        missing_method=missing_method,

        time_var=time_var,

        data_info=data_info,

        result=result,

        candidate_rows=candidate_rows,

        diagnostics=diag_flags,

        plots=plot_flags
    )

    if any(
        plot_flags.values()
    ):

        plot_paths, plot_error = (
            _save_plots(

                y,

                result,

                candidate_rows,

                plot_flags,

                target_var
            )
        )

        if plot_error:

            _write_plot_status(
                plot_error,
                error=True
            )

        else:

            inserted, status = (
                _insert_images_into_viewer(
                    plot_paths
                )
            )

            if not inserted:
                _write_plot_status(
                    status,
                    error=True
                )

    return result


# ============================================================
# NATIVE SPSS XML COMMAND DISPATCHER (IBM EXTENSION HUB STANDARD)
# ============================================================

def Run(args):
    """
    Standard IBM SPSS Extension Hub entry point for XML syntax execution:
    ADF_STATIONARITY_TEST SERIES=var ...
    """
    try:
        from extension import Template, Syntax, processcmd
    except ImportError:
        raise RuntimeError(
            "The IBM SPSS 'extension' Python module is required for XML command execution."
        )

    args = args[list(args.keys())[0]]

    oobj = Syntax([
        Template("SERIES", subc="", var="var_name", ktype="existingvarlist", islist=False),
        Template("TIMEVAR", subc="", var="time_var", ktype="existingvarlist", islist=False),
        Template("DETERMINISTIC", subc="", var="regression_type", ktype="str"),
        Template("LAGMETHOD", subc="", var="lag_method", ktype="str"),
        Template("MAXLAG", subc="", var="max_lag", ktype="int"),
        Template("FIXEDLAG", subc="", var="fixed_lag", ktype="int"),
        Template("ALPHA", subc="", var="alpha", ktype="str"),
        Template("USECUSTOMALPHA", subc="", var="use_custom_alpha", ktype="bool"),
        Template("CUSTOMALPHA", subc="", var="custom_alpha", ktype="float"),
        Template("MISSING", subc="", var="missing_method", ktype="str"),
        Template("DIAGNOSTICS", subc="", var="diagnostics", ktype="str", islist=True),
        Template("PLOTS", subc="", var="plots", ktype="str", islist=True),
    ])

    if "HELP" in args:
        print(run_adf_native.__doc__)
    else:
        processcmd(oobj, args, run_adf_native)