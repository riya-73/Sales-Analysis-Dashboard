"""Data-driven analytics for the Sales Analysis Dashboard.

All functions operate on supplied observations and return empty/None results when
required fields or sample sizes are unavailable; no business data is synthesized.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

BASE_FIELDS = {"Date", "Product", "Category", "Sales"}
OPTIONAL_FIELDS = ["Order ID", "Customer ID", "Quantity", "Unit Price", "Cost", "Profit", "Customer Age", "Customer Gender"]


def clean_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Sales", "Quantity", "Unit Price", "Cost", "Profit", "Customer Age"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def available_fields(df: pd.DataFrame) -> set[str]:
    return set(df.columns)


def kpis(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "Sales" not in df:
        return {"revenue": 0.0, "orders": None, "aov": None, "revenue_per_customer": None}
    revenue = float(df.Sales.sum())
    orders = int(df["Order ID"].nunique()) if "Order ID" in df and df["Order ID"].notna().any() else None
    customers = int(df["Customer ID"].nunique()) if "Customer ID" in df and df["Customer ID"].notna().any() else None
    aov = revenue / orders if orders else None
    rpc = revenue / customers if customers else None
    return {"revenue": revenue, "orders": orders, "aov": aov, "revenue_per_customer": rpc}


def period_revenue(df: pd.DataFrame, frequency: str = "ME") -> pd.DataFrame:
    if not {"Date", "Sales"}.issubset(df.columns):
        return pd.DataFrame(columns=["Period", "Revenue"])
    d = clean_frame(df).dropna(subset=["Date"])
    out = d.set_index("Date").resample(frequency)["Sales"].sum(min_count=1).rename("Revenue").reset_index()
    return out.rename(columns={"Date": "Period"})


def growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = period_revenue(df, "ME")
    if out.empty:
        return out
    out["MoM Growth %"] = out.Revenue.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    annual = period_revenue(df, "YE")
    annual["YoY Growth %"] = annual.Revenue.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    out["Year"] = pd.to_datetime(out.Period).dt.year
    annual["Year"] = pd.to_datetime(annual.Period).dt.year
    return out.merge(annual[["Year", "YoY Growth %"]], on="Year", how="left").drop(columns="Year")


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Product", "Sales"}.issubset(df.columns):
        return pd.DataFrame()
    out = df.groupby([c for c in ["Product", "Category"] if c in df], dropna=False)["Sales"].sum().reset_index(name="Revenue")
    total = out.Revenue.sum()
    out["Contribution %"] = out.Revenue / total * 100 if total else 0
    out = out.sort_values("Revenue", ascending=False)
    out["Cumulative Contribution %"] = out["Contribution %"].cumsum()
    return out


def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Customer ID", "Order ID", "Date", "Sales"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    d = clean_frame(df).dropna(subset=["Customer ID", "Order ID", "Date"])
    if d.empty:
        return pd.DataFrame()
    now = d.Date.max() + pd.Timedelta(days=1)
    grouped = d.groupby("Customer ID").agg(RecencyDays=("Date", lambda x: (now - x.max()).days), Frequency=("Order ID", "nunique"), Revenue=("Sales", "sum"), FirstPurchase=("Date", "min"), LastPurchase=("Date", "max")).reset_index()
    total_revenue = grouped.Revenue.sum()
    grouped["Revenue contribution %"] = grouped.Revenue / total_revenue * 100 if total_revenue else 0.0
    grouped["Customer type"] = np.where(grouped.Frequency > 1, "Returning", "New / single observed order")
    grouped["Segment"] = "At risk"
    if len(grouped) >= 3:
        r = grouped.RecencyDays.rank(pct=True, ascending=False)
        f = grouped.Frequency.rank(pct=True)
        m = grouped.Revenue.rank(pct=True)
        grouped.loc[(r >= .67) & (f >= .67) & (m >= .67), "Segment"] = "Champions"
        grouped.loc[(grouped.Frequency <= 1) & (grouped.RecencyDays <= grouped.RecencyDays.median()), "Segment"] = "New customers"
        grouped.loc[(grouped.Frequency >= grouped.Frequency.median()) & (grouped.RecencyDays <= grouped.RecencyDays.median()), "Segment"] = "Loyal"
    return grouped


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Customer ID", "Order ID", "Date"}.issubset(df.columns):
        return pd.DataFrame()
    d = clean_frame(df).dropna(subset=["Customer ID", "Order ID", "Date"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["OrderMonth"] = d.Date.dt.to_period("M")
    first = d.groupby("Customer ID").OrderMonth.min().rename("CohortMonth")
    d = d.join(first, on="Customer ID")
    d["CohortIndex"] = (d.OrderMonth.dt.year - d.CohortMonth.dt.year) * 12 + d.OrderMonth.dt.month - d.CohortMonth.dt.month
    cohort = d.groupby(["CohortMonth", "CohortIndex"])["Customer ID"].nunique().unstack(fill_value=0)
    retention = cohort.div(cohort.iloc[:, 0], axis=0) * 100
    retention.index = retention.index.astype(str)
    return retention


def basket_rules(df: pd.DataFrame, min_support: float = .02, min_confidence: float = .2) -> pd.DataFrame:
    if not {"Order ID", "Product"}.issubset(df.columns):
        return pd.DataFrame()
    baskets = df.dropna(subset=["Order ID", "Product"]).groupby("Order ID").Product.apply(lambda s: set(s.astype(str)))
    n = len(baskets)
    if n < 2:
        return pd.DataFrame()
    support: dict[frozenset, float] = {}
    products = sorted(set().union(*baskets))
    for size in (1, 2):
        for combo in itertools.combinations(products, size):
            count = sum(set(combo).issubset(b) for b in baskets)
            if count / n >= min_support:
                support[frozenset(combo)] = count / n
    rows = []
    for itemset, sup in support.items():
        if len(itemset) != 2:
            continue
        a, b = sorted(itemset)
        sa, sb = support.get(frozenset([a]), 0), support.get(frozenset([b]), 0)
        for lhs, rhs, sl, sr in ((a, b, sa, sb), (b, a, sb, sa)):
            confidence = sup / sl if sl else 0
            if confidence >= min_confidence:
                rows.append({"If purchased": lhs, "Also purchased": rhs, "Support": sup, "Confidence": confidence, "Lift": confidence / sr if sr else math.nan})
    return pd.DataFrame(rows).sort_values("Lift", ascending=False) if rows else pd.DataFrame(columns=["If purchased", "Also purchased", "Support", "Confidence", "Lift"])


def outlier_summary(df: pd.DataFrame, column: str = "Sales") -> dict[str, Any]:
    x = pd.to_numeric(df[column], errors="coerce").dropna()
    if x.empty:
        return {}
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    iqr_n = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    z = np.abs(stats.zscore(x, nan_policy="omit")) if len(x) > 1 else np.zeros(len(x))
    return {"count": len(x), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()), "IQR outliers": iqr_n, "Z-score outliers (|z|>3)": int((z > 3).sum())}


def category_anova(df: pd.DataFrame) -> dict[str, Any] | None:
    if not {"Category", "Sales"}.issubset(df.columns):
        return None
    groups = [g.Sales.dropna().values for _, g in df.groupby("Category") if len(g.Sales.dropna()) >= 2]
    if len(groups) < 2:
        return None
    normal = all(len(g) >= 3 and stats.shapiro(g).pvalue >= .05 for g in groups)
    equal_var = stats.levene(*groups, center="median").pvalue >= .05
    if normal and equal_var:
        result = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (g.mean() - np.concatenate(groups).mean()) ** 2 for g in groups)
        ss_total = sum(((g - np.concatenate(groups).mean()) ** 2).sum() for g in groups)
        effect, effect_name, test_name = (float(ss_between / ss_total) if ss_total else 0.0), "eta-squared", "One-way ANOVA"
    else:
        result = stats.kruskal(*groups)
        n = sum(map(len, groups)); k = len(groups)
        effect, effect_name, test_name = (max(0.0, float((result.statistic - k + 1) / (n - k))) if n > k else 0.0), "epsilon-squared", "Kruskal–Wallis"
    return {"test": test_name, "statistic": float(result.statistic), "p_value": float(result.pvalue), "groups": len(groups), "effect_size": effect, "effect_size_name": effect_name}


def forecast_baselines(df: pd.DataFrame, periods: int = 6) -> tuple[pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
    """Select by expanding-window rolling-origin WAPE; uncertainty is model-based or backtest-based."""
    monthly = period_revenue(df, "ME")
    if monthly.Revenue.isna().any():
        return pd.DataFrame(), {}, pd.DataFrame()
    series = monthly.set_index("Period").Revenue.astype(float)
    if len(series) < 14:
        return pd.DataFrame(), {}, pd.DataFrame()
    errors, detail = rolling_origin_backtest(df)
    if not errors:
        return pd.DataFrame(), {}, detail
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    future = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    best_model = min(errors, key=lambda name: errors[name]["WAPE %"])
    fitted = None
    try:
        if best_model == "Moving average":
            values = np.repeat(series.tail(3).mean(), periods)
            lower = upper = None
        elif best_model.startswith("ARIMA"):
            from statsmodels.tsa.arima.model import ARIMA
            fitted = ARIMA(series, order=(1, 1, 1)).fit()
            pred = fitted.get_forecast(periods); values = np.asarray(pred.predicted_mean)
            ci = pred.conf_int(); lower, upper = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
        elif best_model.startswith("SARIMA"):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            fitted = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred = fitted.get_forecast(periods); values = np.asarray(pred.predicted_mean)
            ci = pred.conf_int(); lower, upper = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
        else:
            fitted = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated").fit()
            values = np.asarray(fitted.forecast(periods)); lower = upper = None
        if lower is None or upper is None:
            residuals = detail.loc[detail.Model == best_model, "Residual"].to_numpy()
            spread = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
            lower, upper = values - 1.96 * spread, values + 1.96 * spread
            interval_type = "Approx. 95% interval from rolling-origin errors"
        else:
            interval_type = "95% model prediction interval"
        forecast = pd.DataFrame({"Period": future, "Forecast": np.maximum(np.asarray(values), 0), "Lower 95%": np.maximum(np.asarray(lower), 0), "Upper 95%": np.maximum(np.asarray(upper), 0), "Selected model": best_model, "Interval": interval_type})
    except Exception:
        forecast = pd.DataFrame()
    return forecast, errors, detail


# --- Second-stage methods: inferential summaries, BI comparisons, and diagnostics ---

def mean_confidence_interval(values, confidence: float = 0.95) -> dict[str, float] | None:
    """Two-sided Student-t interval for an independent sample mean."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return None
    mean = float(np.mean(x)); se = float(stats.sem(x))
    if not np.isfinite(se):
        return None
    lo, hi = stats.t.interval(confidence, len(x) - 1, loc=mean, scale=se)
    return {"mean": mean, "lower": float(lo), "upper": float(hi), "confidence": confidence, "n": len(x)}


def correlation_analysis(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Pairwise Pearson correlation, p-value, sample size, and Fisher-z 95% CI."""
    cols = columns or list(df.select_dtypes(include="number").columns)
    rows = []
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            pair = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(pair)
            if n < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            r, p = stats.pearsonr(pair[left], pair[right])
            if n > 3 and abs(r) < 1:
                z = np.arctanh(r); margin = 1.96 / np.sqrt(n - 3)
                lower, upper = float(np.tanh(z - margin)), float(np.tanh(z + margin))
            else:
                lower, upper = float("nan"), float("nan")
            rows.append({"Variable 1": left, "Variable 2": right, "Pearson r": float(r), "p-value": float(p), "95% CI lower": lower, "95% CI upper": upper, "n": n})
    return pd.DataFrame(rows)


def distribution_summary(values, column: str = "Sales") -> dict[str, Any] | None:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return None
    result: dict[str, Any] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()), "skewness": float(x.skew()), "kurtosis": float(x.kurt()), "q1": float(x.quantile(.25)), "q3": float(x.quantile(.75))}
    # Shapiro-Wilk's p-value is not reliable for very large samples; limit to a reproducible head sample.
    if 3 <= len(x) <= 5000:
        test = stats.shapiro(x.to_numpy())
        result.update({"Shapiro statistic": float(test.statistic), "Shapiro p-value": float(test.pvalue), "normality sample n": len(x)})
    else:
        result["Shapiro p-value"] = None
    return result


def monthly_anomalies(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """Flag robust median/MAD outliers in monthly revenue; no flag when MAD is zero."""
    series = period_revenue(df, "ME")
    if series.Revenue.notna().sum() < 6:
        return pd.DataFrame(columns=["Period", "Revenue", "Robust Z", "Anomaly"])
    median = series.Revenue.median()
    mad = np.median(np.abs(series.Revenue.dropna().to_numpy() - median))
    series["Robust Z"] = (0.6745 * (series.Revenue - median) / mad) if mad > 0 else np.nan
    series["Anomaly"] = series["Robust Z"].abs() >= threshold
    return series


def apply_filters(df: pd.DataFrame, start=None, end=None, categories: list[str] | None = None) -> pd.DataFrame:
    out = clean_frame(df)
    if start is not None and "Date" in out:
        out = out[out.Date.dt.date >= pd.Timestamp(start).date()]
    if end is not None and "Date" in out:
        out = out[out.Date.dt.date <= pd.Timestamp(end).date()]
    if categories is not None and "Category" in out:
        out = out[out.Category.astype(str).isin([str(x) for x in categories])]
    return out.copy()


def compare_period_revenue(df: pd.DataFrame, first_start, first_end, second_start, second_end) -> dict[str, Any] | None:
    if first_start > first_end or second_start > second_end:
        return None
    a = apply_filters(df, first_start, first_end)
    b = apply_filters(df, second_start, second_end)
    if "Sales" not in df:
        return None
    ra, rb = float(a.Sales.sum()), float(b.Sales.sum())
    delta = rb - ra
    pct = delta / abs(ra) * 100 if ra != 0 else None
    return {"first_revenue": ra, "second_revenue": rb, "change": delta, "change_pct": pct, "first_rows": len(a), "second_rows": len(b)}


def product_period_changes(df: pd.DataFrame, recent_months: int = 2) -> pd.DataFrame:
    if not {"Date", "Product", "Sales"}.issubset(df.columns):
        return pd.DataFrame()
    d = clean_frame(df).dropna(subset=["Date"])
    if d.empty:
        return pd.DataFrame()
    end = d.Date.max().to_period("M"); recent_start = end - (recent_months - 1)
    prior_end = recent_start - 1; prior_start = prior_end - (recent_months - 1)
    d["Month"] = d.Date.dt.to_period("M")
    recent = d[d.Month.between(recent_start, end)].groupby("Product").Sales.sum().rename("Recent revenue")
    prior = d[d.Month.between(prior_start, prior_end)].groupby("Product").Sales.sum().rename("Prior revenue")
    out = pd.concat([prior, recent], axis=1).fillna(0).reset_index()
    out["Change"] = out["Recent revenue"] - out["Prior revenue"]
    out["Change %"] = np.where(out["Prior revenue"] != 0, out.Change / out["Prior revenue"].abs() * 100, np.nan)
    out["Direction"] = np.select([out.Change > 0, out.Change < 0], ["Growing", "Declining"], default="Flat")
    return out.sort_values("Change", ascending=False)


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    monthly = period_revenue(df, "ME")
    if monthly.Revenue.notna().sum() < 12:
        return pd.DataFrame()
    periods = pd.to_datetime(monthly.Period)
    monthly["Year"] = periods.dt.year; monthly["Month"] = periods.dt.month
    return monthly.pivot(index="Year", columns="Month", values="Revenue")


def rolling_origin_backtest(df: pd.DataFrame, max_origins: int = 6) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Expanding-window one-step backtest on the latest origins; errors are out-of-sample."""
    monthly = period_revenue(df, "ME")
    if monthly.Revenue.isna().any():
        return {}, pd.DataFrame()
    series = monthly.set_index("Period").Revenue.astype(float)
    if len(series) < 14:
        return {}, pd.DataFrame()
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    first_origin = max(12, len(series) - max_origins)
    rows = []
    for origin in range(first_origin, len(series)):
        train, actual = series.iloc[:origin], float(series.iloc[origin])
        target = series.index[origin]
        predictions = {"Moving average": float(train.tail(3).mean())}
        for name, factory in [("Exponential smoothing", lambda: ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated").fit().forecast(1))]:
            try: predictions[name] = float(np.asarray(factory())[0])
            except Exception: pass
        if len(train) >= 18:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                fitted = ARIMA(train, order=(1, 1, 1)).fit()
                if not fitted.mle_retvals.get("converged", True):
                    raise RuntimeError("ARIMA did not converge")
                predictions["ARIMA(1,1,1)"] = float(fitted.forecast(1).iloc[0])
            except Exception: pass
        if len(train) >= 36:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                fitted = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if not fitted.mle_retvals.get("converged", True):
                    raise RuntimeError("SARIMA did not converge")
                predictions["SARIMA(1,1,1)(1,1,1,12)"] = float(fitted.forecast(1).iloc[0])
            except Exception: pass
        for model, predicted in predictions.items():
            rows.append({"Origin": target, "Model": model, "Actual": actual, "Predicted": predicted, "Residual": actual - predicted})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {}, detail
    metrics = {}
    for model, g in detail.groupby("Model"):
        residual = g.Residual.to_numpy()
        metrics[model] = {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2))), "WAPE %": float(np.sum(np.abs(residual)) / max(np.sum(np.abs(g.Actual)), 1e-9) * 100), "Mean residual": float(np.mean(residual)), "Backtest origins": int(len(g))}
    return metrics, detail
