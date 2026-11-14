import io
import numpy as np
import pandas as pd

from analytics import (
    apply_filters, category_anova, compare_period_revenue, correlation_analysis,
    cohort_retention, customer_summary, distribution_summary, forecast_baselines,
    mean_confidence_interval, monthly_anomalies, monthly_seasonality,
    product_period_changes, rolling_origin_backtest,
)
from storage import DataStore, validate_data


def complete_data(months=24):
    dates = pd.date_range("2023-01-01", periods=months, freq="MS")
    rows=[]
    for i, dt in enumerate(dates):
        rows.extend([
            {"Date":dt,"Product":"A","Category":"Clothing","Sales":100+i,"Order ID":f"o{i}a","Customer ID":f"c{i%4}","Quantity":1},
            {"Date":dt,"Product":"B","Category":"Electronics","Sales":50+i/2,"Order ID":f"o{i}b","Customer ID":f"c{(i+1)%4}","Quantity":1},
        ])
    return pd.DataFrame(rows)


def test_confidence_interval_and_category_effect_size():
    ci=mean_confidence_interval([2,4,6,8])
    assert ci and ci["n"]==4 and ci["lower"]<ci["mean"]<ci["upper"]
    data=pd.DataFrame({"Category":["A"]*5+["B"]*5,"Sales":[1,2,2,3,2,10,12,11,13,12]})
    result=category_anova(data)
    assert result and 0<=result["effect_size"]<=1
    assert 0<=result["p_value"]<=1


def test_correlation_has_pairwise_ci_and_missing_values_are_handled():
    df=pd.DataFrame({"x":[1,2,3,4,5],"y":[2,4,6,8,10],"z":[1,None,1,2,3]})
    result=correlation_analysis(df)
    xy=result[(result["Variable 1"]=="x")&(result["Variable 2"]=="y")].iloc[0]
    assert xy["Pearson r"]==1 and xy["n"]==5
    assert np.isnan(xy["95% CI lower"])  # Fisher interval needs n > 3 and |r| < 1.
    assert distribution_summary(df.y)["n"]==5


def test_customer_new_returning_share_and_cohort():
    df=pd.DataFrame({"Date":pd.to_datetime(["2024-01-03","2024-02-05","2024-01-15","2024-02-12"]),"Customer ID":["a","a","b","b"],"Order ID":["a1","a2","b1","b2"],"Sales":[10,20,30,40]})
    cs=customer_summary(df)
    assert set(cs["Customer type"])=={"Returning"}
    assert np.isclose(cs["Revenue contribution %"].sum(),100)
    retention=cohort_retention(df)
    assert retention.iloc[:,0].eq(100).all()


def test_filter_and_selected_period_comparison():
    df=complete_data(3)
    filtered=apply_filters(df,"2023-02-01","2023-03-31",["Clothing"])
    assert len(filtered)==2 and filtered.Category.eq("Clothing").all()
    comparison=compare_period_revenue(df,"2023-01-01","2023-01-31","2023-02-01","2023-02-28")
    assert comparison["first_rows"]==2 and comparison["second_rows"]==2
    assert comparison["change"]>0


def test_product_growth_seasonality_and_anomalies():
    df=complete_data(24)
    changes=product_period_changes(df)
    assert set(changes.Direction)<= {"Growing","Declining","Flat"}
    seasonal=monthly_seasonality(df)
    assert not seasonal.empty and seasonal.shape[1]==12
    anomaly=monthly_anomalies(df)
    assert len(anomaly)==24 and anomaly.Anomaly.dtype==bool


def test_rolling_forecast_and_insufficient_history_guard():
    short=complete_data(10)
    forecast,metrics,details=forecast_baselines(short)
    assert forecast.empty and not metrics and details.empty
    long=complete_data(24)
    metrics,details=rolling_origin_backtest(long,max_origins=3)
    assert metrics and len(details)>0 and {"MAE","RMSE","WAPE %","Mean residual"}<=set(next(iter(metrics.values())))
    future,compare,_=forecast_baselines(long,3)
    assert len(future)==3 and compare
    assert {"Lower 95%","Upper 95%","Selected model"}<=set(future.columns)
    missing_month=long[~long.Date.dt.to_period("M").eq(pd.Period("2023-06","M"))]
    withheld,missing_metrics,_=forecast_baselines(missing_month,3)
    assert withheld.empty and not missing_metrics


def test_empty_missing_import_export_and_sqlite_roundtrip(tmp_path):
    empty,notes=validate_data(pd.DataFrame(columns=["Date","Product","Category","Sales"]))
    assert empty.empty
    incoming=pd.DataFrame({"Date":["2024-01-01","bad"],"Product":["A","X"],"Category":["Clothing","X"],"Sales":[12,"bad"]})
    csv=incoming.to_csv(index=False).encode()
    parsed=pd.read_csv(io.BytesIO(csv))
    clean,warnings=validate_data(parsed)
    assert len(clean)==1 and warnings
    excel=io.BytesIO(); clean.to_excel(excel,index=False); excel.seek(0)
    excel_clean,_=validate_data(pd.read_excel(excel))
    db=DataStore(str(tmp_path/"stage2.sqlite")); db.replace(excel_clean)
    saved=db.load()
    assert len(saved)==1 and saved.Sales.iloc[0]==12
    csv_out=saved.to_csv(index=False).encode()
    assert pd.read_csv(io.BytesIO(csv_out)).Sales.iloc[0]==12
