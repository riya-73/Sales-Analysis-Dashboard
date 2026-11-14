from __future__ import annotations
from datetime import date
import io
import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (BASE_FIELDS, OPTIONAL_FIELDS, apply_filters, basket_rules,
                       category_anova, clean_frame, cohort_retention,
                       compare_period_revenue, correlation_analysis, customer_summary,
                       distribution_summary, forecast_baselines, growth_metrics, kpis,
                       mean_confidence_interval, monthly_anomalies, outlier_summary,
                       period_revenue, product_period_changes, product_summary,
                       monthly_seasonality, rolling_origin_backtest)
from storage import DataStore, DataValidationError, validate_data

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("sales_dashboard")

st.set_page_config(page_title="Sales Intelligence | BI Dashboard", page_icon="▦", layout="wide")
st.markdown("""<style>
:root{--ink:#172b4d;--muted:#65758b;--line:#e4eaf1;--accent:#147d78}
.block-container{padding-top:1.5rem;max-width:1500px}
[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e4eaf1;padding:16px 18px;border-radius:12px}
[data-testid="stSidebar"]{background:#f7f9fc}
h1,h2,h3{color:#172b4d}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def store(): return DataStore(os.getenv("SALES_DB_PATH", "sales_dashboard.sqlite3"))

try:
    db=store()
    raw=db.load()
except Exception as exc:
    logger.exception("Database initialization failed")
    st.error(f"Could not open the local database: {exc}")
    st.stop()

st.title("Sales Intelligence")
st.caption("Business performance from your actual sales records. Metrics requiring missing identifiers or costs remain unavailable.")
sections=["Executive Overview","Sales Performance","Product Analytics","Customer Analytics","Statistical Analysis","Market Basket Analysis","Sales Forecasting","Business Recommendations","Data Management"]
page=st.sidebar.radio("Navigation", sections)
if page == "Executive Overview":
    with st.sidebar:
        st.divider(); st.subheader("Optional KPI targets")
        target_revenue=st.number_input("Revenue target for current filter (₹; 0 disables)",min_value=0.0,value=0.0,step=1000.0)
        target_growth=st.number_input("Monthly growth target (%) (0 disables)",min_value=0.0,value=0.0,step=1.0)
    st.session_state["target_revenue"]=target_revenue
    st.session_state["target_growth"]=target_growth

# Shared filters
if raw.empty:
    filtered=raw.copy()
else:
    d=clean_frame(raw)
    with st.sidebar:
        st.divider(); st.subheader("Dashboard filters")
        valid_dates=d.Date.dropna()
        if not valid_dates.empty:
            bounds=st.date_input("Date range", value=(valid_dates.min().date(),valid_dates.max().date()), min_value=valid_dates.min().date(), max_value=valid_dates.max().date())
            if isinstance(bounds,tuple) and len(bounds)==2:
                d=d[(d.Date.dt.date>=bounds[0])&(d.Date.dt.date<=bounds[1])]
        cats=sorted(d.Category.dropna().astype(str).unique()) if "Category" in d else []
        selected=st.multiselect("Categories",cats,default=cats)
        if selected and "Category" in d: d=d[d.Category.astype(str).isin(selected)]
        elif not selected and cats and "Category" in d: d=d.iloc[0:0]
    filtered=d


def show_kpis(df):
    m=kpis(df); cols=st.columns(4)
    cols[0].metric("Revenue",f"₹{m['revenue']:,.2f}")
    cols[1].metric("Orders",f"{m['orders']:,}" if m['orders'] is not None else "Unavailable",help="Requires a populated Order ID column.")
    cols[2].metric("Average order value",f"₹{m['aov']:,.2f}" if m['aov'] is not None else "Unavailable",help="Without Order ID, average is per recorded row and labelled as such.")
    cols[3].metric("Revenue / customer",f"₹{m['revenue_per_customer']:,.2f}" if m['revenue_per_customer'] is not None else "Unavailable",help="Requires Customer ID.")


def trend_chart(df, freq="M"):
    s=period_revenue(df,freq)
    if s.empty: st.info("Date and Sales fields are required for time-series analytics."); return
    fig=px.line(s,x="Period",y="Revenue",markers=True,title="Revenue over time",template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)


@st.cache_data(show_spinner="Running rolling-origin forecast comparison…")
def cached_forecast(df, periods):
    return forecast_baselines(df, periods)


def overview(df):
    show_kpis(df)
    targets=[]; revenue_target=st.session_state.get("target_revenue",0.0)
    if revenue_target>0:
        actual=kpis(df)["revenue"]; attainment=actual/revenue_target
        st.metric("Revenue target attainment",f"{attainment:.1%}",delta=f"₹{actual-revenue_target:,.2f} vs target")
        st.progress(min(max(attainment,0),1))
    growth_target=st.session_state.get("target_growth",0.0)
    growth=growth_metrics(df)
    if growth_target>0 and not growth.empty and growth["MoM Growth %"].notna().any():
        observed=float(growth["MoM Growth %"].dropna().iloc[-1])
        st.metric("Latest observed MoM growth vs target",f"{observed:.1f}%",delta=f"{observed-growth_target:+.1f} percentage points vs target")
    left,right=st.columns([1.5,1])
    with left: trend_chart(df)
    with right:
        ps=product_summary(df)
        if not ps.empty: st.plotly_chart(px.pie(ps,values="Revenue",names="Category" if "Category" in ps else "Product",hole=.58,title="Revenue contribution by category",template="plotly_white"),use_container_width=True)
    if not df.empty:
        ps=product_summary(df)
        if not ps.empty:
            st.subheader("Revenue leaders")
            st.dataframe(ps.head(8),use_container_width=True,hide_index=True)
            top=ps.iloc[0]
            st.info(f"Observed insight: **{top.Product}** is the highest-revenue product in the current filtered data (₹{top.Revenue:,.2f}; {top['Contribution %']:.1f}% of measured product revenue).")
        anomalies=monthly_anomalies(df)
        flagged=anomalies[anomalies.Anomaly] if not anomalies.empty else pd.DataFrame()
        if not flagged.empty:
            latest=flagged.iloc[-1]
            st.warning(f"Revenue anomaly flag: {latest.Period:%b %Y} recorded ₹{latest.Revenue:,.2f} (robust z={latest['Robust Z']:.2f}) relative to the observed monthly distribution. Investigate context; a statistical flag is not a cause.")


def sales_page(df):
    show_kpis(df)
    freq=st.selectbox("Time aggregation",["Daily","Monthly","Yearly"],index=1)
    trend_chart(df,{"Daily":"D","Monthly":"ME","Yearly":"YE"}[freq])
    g=growth_metrics(df)
    if not g.empty:
        st.subheader("Period-over-period growth")
        st.dataframe(g,use_container_width=True,hide_index=True)
        st.caption("Growth is undefined for zero prior-period revenue and is left blank. YoY appears only where a comparison year exists.")
    anomalies=monthly_anomalies(df)
    if not anomalies.empty:
        st.subheader("Revenue anomaly investigation")
        fig=px.line(anomalies,x="Period",y="Revenue",markers=True,template="plotly_white")
        flagged=anomalies[anomalies.Anomaly]
        if not flagged.empty: fig.add_scatter(x=flagged.Period,y=flagged.Revenue,mode="markers",marker={"color":"red","size":11},name="Robust anomaly")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Monthly revenue is flagged at |modified robust z-score| ≥ 3.5 using median absolute deviation. Flags require at least six observed months and are not causal findings.")
    valid=clean_frame(df).dropna(subset=["Date"])
    if not valid.empty:
        periods=sorted(valid.Date.dt.to_period("M").unique())
        if len(periods)>=2:
            st.subheader("Compare selected months")
            options=[str(p) for p in periods]
            c1,c2=st.columns(2); p1=c1.selectbox("First month",options,index=max(0,len(options)-2)); p2=c2.selectbox("Second month",options,index=len(options)-1)
            a=pd.Period(p1,freq="M"); b=pd.Period(p2,freq="M")
            comparison=compare_period_revenue(df,a.start_time.date(),a.end_time.date(),b.start_time.date(),b.end_time.date())
            if comparison:
                c1.metric(f"Revenue {p1}",f"₹{comparison['first_revenue']:,.2f}")
                c2.metric(f"Revenue {p2}",f"₹{comparison['second_revenue']:,.2f}",delta=f"₹{comparison['change']:,.2f} ({comparison['change_pct']:.1f}%)" if comparison["change_pct"] is not None else f"₹{comparison['change']:,.2f}; baseline is zero")


def product_page(df):
    ps=product_summary(df)
    if ps.empty: st.info("Product and Sales columns are required."); return
    st.subheader("Revenue by product")
    st.plotly_chart(px.bar(ps.head(30).sort_values("Revenue"),x="Revenue",y="Product",color="Category" if "Category" in ps else None,orientation="h",title="Top products by revenue",template="plotly_white"),use_container_width=True)
    st.subheader("Pareto / revenue concentration")
    fig=go.Figure(); fig.add_bar(x=ps.Product,y=ps.Revenue,name="Revenue"); fig.add_scatter(x=ps.Product,y=ps["Cumulative Contribution %"],name="Cumulative %",yaxis="y2")
    fig.update_layout(template="plotly_white",yaxis_title="Revenue",yaxis2=dict(title="Cumulative share (%)",overlaying="y",side="right",range=[0,100]))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(ps,use_container_width=True,hide_index=True)
    if "Category" in df:
        st.subheader("Category performance comparison")
        categories=df.groupby("Category",dropna=False).Sales.sum().sort_values(ascending=False).rename("Revenue").reset_index()
        categories["Contribution %"]=categories.Revenue/categories.Revenue.sum()*100 if categories.Revenue.sum() else 0
        st.plotly_chart(px.bar(categories,x="Category",y="Revenue",color="Category",title="Observed category revenue",template="plotly_white"),use_container_width=True)
        st.dataframe(categories,use_container_width=True,hide_index=True)
    changes=product_period_changes(df)
    if not changes.empty:
        st.subheader("Recent product growth / decline")
        st.dataframe(changes,use_container_width=True,hide_index=True)
        st.caption("Compares the latest two calendar months in this filtered dataset with the immediately preceding two months. New products with no prior revenue show an undefined percent change.")
    seasonal=monthly_seasonality(df)
    if not seasonal.empty:
        st.subheader("Monthly revenue seasonality (descriptive)")
        st.plotly_chart(px.imshow(seasonal,labels={"x":"Calendar month","y":"Year","color":"Revenue"},color_continuous_scale="Teal",aspect="auto",text_auto=".0f"),use_container_width=True)
        st.caption("Seasonality view is a descriptive year-by-month layout; it does not establish a recurring seasonal effect or adjust for partial months/business exposure.")
    if not ({"Cost","Profit"}&set(df.columns)) or df[[c for c in ["Cost","Profit"] if c in df]].isna().all().all(): st.info("Profitability is unavailable: no populated cost or profit data. Figures above are revenue-based.")
    elif "Profit" in df and df.Profit.notna().any():
        st.caption("Profit is shown only because Profit values are present in the source data.")
        st.dataframe(df.groupby("Product").Profit.sum().sort_values(ascending=False).rename("Profit").reset_index(),use_container_width=True,hide_index=True)


def customer_page(df):
    cs=customer_summary(df)
    if cs.empty:
        st.info("Customer-level purchase frequency, RFM-style segmentation, and cohort analysis require Customer ID, Order ID, and transaction dates. The current legacy schema has no identifiers. No behavior is inferred from age or gender.")
        return
    repeat=float((cs.Frequency>1).mean()*100) if len(cs) else 0
    a,b,c=st.columns(3); a.metric("Unique customers",f"{len(cs):,}"); b.metric("Repeat purchase rate",f"{repeat:.1f}%"); c.metric("Avg purchases / customer",f"{cs.Frequency.mean():.2f}")
    types=cs.groupby("Customer type").agg(Customers=("Customer ID","nunique"),Revenue=("Revenue","sum")).reset_index()
    st.subheader("New / returning customers within selected data")
    st.dataframe(types,use_container_width=True,hide_index=True)
    st.caption("‘New / single observed order’ means one order in the available filtered records; customers whose earlier history is outside this dataset cannot be reliably distinguished from truly new customers.")
    st.plotly_chart(px.bar(cs.Segment.value_counts().rename_axis("Segment").reset_index(name="Customers"),x="Segment",y="Customers",color="Segment",title="RFM-inspired customer segments",template="plotly_white"),use_container_width=True)
    st.caption("Segments use recency, distinct Order IDs, and revenue as transparent percentile rules. This is a simple RFM-inspired heuristic, not a validated predictive model.")
    st.subheader("Customer revenue contribution")
    st.dataframe(cs.sort_values("Revenue",ascending=False),use_container_width=True,hide_index=True)
    retention=cohort_retention(df)
    if not retention.empty:
        st.subheader("Monthly cohort retention (%)")
        st.plotly_chart(px.imshow(retention,labels={"x":"Months since first order","y":"First purchase cohort","color":"Retained customers (%)"},color_continuous_scale="Teal",zmin=0,zmax=100,text_auto=".0f",aspect="auto"),use_container_width=True)
        st.dataframe(retention.style.format("{:.1f}"),use_container_width=True)
        st.caption("Retention is the share of cohort customers with an observed order in each month relative to the cohort's first observed purchase month.")


def stats_page(df):
    if "Sales" not in df: st.info("Sales field is required."); return
    st.subheader("Descriptive statistics")
    numeric=df.select_dtypes(include="number")
    if not numeric.empty: st.dataframe(numeric.describe().T,use_container_width=True)
    out=outlier_summary(df)
    if out: st.write("Sales distribution & outliers",out)
    sales_ci=mean_confidence_interval(df.Sales) if df.Sales.notna().any() else None
    if sales_ci:
        st.info(f"Mean recorded sales per row: ₹{sales_ci['mean']:,.2f}; 95% confidence interval ₹{sales_ci['lower']:,.2f}–₹{sales_ci['upper']:,.2f} (n={sales_ci['n']}). This interval concerns the average observed row, not customer-level or order-level spend unless the input grain matches that unit.")
    dist=distribution_summary(df.Sales)
    if dist:
        st.subheader("Sales distribution")
        st.plotly_chart(px.histogram(df,x="Sales",nbins=30,marginal="box",title="Observed row-level sales distribution",template="plotly_white"),use_container_width=True)
        if dist.get("Shapiro p-value") is not None:
            st.caption(f"Observed skewness {dist['skewness']:.2f}; Shapiro–Wilk p={dist['Shapiro p-value']:.4g} (n={dist['normality sample n']}). Normality tests are sensitive to sample size and are diagnostics, not business conclusions.")
    if len(numeric.columns)>1:
        st.subheader("Numeric correlation (Pearson)")
        corr=numeric.corr(numeric_only=True); st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu",zmin=-1,zmax=1),use_container_width=True)
        corr_detail=correlation_analysis(df,list(numeric.columns))
        if not corr_detail.empty: st.dataframe(corr_detail,use_container_width=True,hide_index=True)
        st.caption("Pairwise p-values and Fisher-z confidence intervals are unadjusted for multiple comparisons; associations are not evidence of causation.")
    anova=category_anova(df)
    st.subheader("Category revenue differences")
    if anova:
        null="category mean revenues are equal" if anova["test"]=="One-way ANOVA" else "category revenue distributions are equal in location (with comparable shapes)"
        st.write(f"H₀: {null}. H₁: at least one differs. Test: {anova['test']} (statistic={anova['statistic']:.3f}, p={anova['p_value']:.4g}; {anova['groups']} groups; {anova['effect_size_name']}={anova['effect_size']:.3f}).")
        st.write("Interpretation: statistical evidence of a difference." if anova["p_value"]<.05 else "Interpretation: insufficient evidence at α=0.05; this does not prove categories are equal.")
        st.write(f"Business reading: estimated effect size ({anova['effect_size_name']}={anova['effect_size']:.3f}) describes the observed group separation; weigh this magnitude against revenue scale, costs, and decision thresholds before acting.")
        st.caption("The test describes row-level Sales; repeated products/orders may violate independence. Statistical significance is not business impact.")
    else: st.info("Need at least two categories with two or more valid observations each for a category comparison.")
    if "Date" in df:
        monthly=period_revenue(df,"ME")
        if len(monthly)>=6: st.subheader("Monthly trend and seasonality (descriptive)"); st.plotly_chart(px.line(monthly,x="Period",y="Revenue",markers=True),use_container_width=True)
        else: st.caption("Seasonality interpretation is withheld: fewer than six monthly observations.")


def basket_page(df):
    if not {"Order ID","Product"}.issubset(df.columns) or df["Order ID"].isna().all():
        st.info("Market-basket analysis is unavailable because there are no populated Order ID values joined to product-level line items. The current one-row-per-manual-entry schema is not evidence of shared baskets.")
        st.markdown("**Optional transaction schema:** one row per purchased product line, with `Order ID`, `Date`, `Product`, and optionally `Customer ID`, `Quantity`, `Unit Price`, and `Sales`. Do not generate order IDs for historical rows.")
        return
    support=st.slider("Minimum support",.01,.5,.02,.01); confidence=st.slider("Minimum confidence",.05,1.0,.2,.05)
    rules=basket_rules(df,support,confidence)
    if rules.empty: st.info("No rules met the selected thresholds. Try lower support/confidence if the dataset is sufficiently large.")
    else: st.dataframe(rules,use_container_width=True,hide_index=True)
    st.caption("Support = share of orders containing both items; confidence = conditional share; lift > 1 means the pair occurs more often than independence suggests.")


def forecast_page(df):
    periods=st.select_slider("Forecast horizon (months)",options=[3,6,12],value=6)
    forecast,errors,detail=cached_forecast(df,periods)
    if not errors:
        st.warning("Forecast withheld: at least 14 contiguous observed monthly periods and enough chronological rolling origins are required. This dashboard currently has no bundled history.")
        return
    st.subheader("Rolling-origin model comparison"); rows=[]
    for name,m in errors.items(): rows.append({"Model":name,**m})
    st.dataframe(pd.DataFrame(rows).sort_values("WAPE %"),use_container_width=True,hide_index=True)
    if not forecast.empty:
        hist=period_revenue(df,"ME").rename(columns={"Revenue":"Actual"})
        fig=go.Figure(); fig.add_scatter(x=hist.Period,y=hist.Actual,name="Observed revenue");
        fig.add_scatter(x=forecast.Period,y=forecast["Lower 95%"],mode="lines",line={"width":0},showlegend=False,hoverinfo="skip")
        fig.add_scatter(x=forecast.Period,y=forecast["Upper 95%"],mode="lines",line={"width":0},fill="tonexty",fillcolor="rgba(20,125,120,.16)",name=str(forecast.Interval.iloc[0]))
        fig.add_scatter(x=forecast.Period,y=forecast.Forecast,name=f"{forecast['Selected model'].iloc[0]} forecast",line=dict(dash="dash")); st.plotly_chart(fig,use_container_width=True)
        st.caption("Intervals are model prediction intervals for ARIMA/SARIMA; other methods use an approximate interval based on rolling-origin residual spread. These are uncertainty ranges, not guaranteed outcomes.")
    if not detail.empty:
        best=min(errors,key=lambda name:errors[name]["WAPE %"])
        residuals=detail[detail.Model==best].sort_values("Origin")
        st.subheader(f"Backtest diagnostics — {best}")
        fig=go.Figure(); fig.add_scatter(x=residuals.Origin,y=residuals.Actual,mode="lines+markers",name="Actual"); fig.add_scatter(x=residuals.Origin,y=residuals.Predicted,mode="lines+markers",name="Predicted"); st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2); c1.plotly_chart(px.scatter(residuals,x="Predicted",y="Residual",trendline="ols",title="Residuals vs predicted",template="plotly_white"),use_container_width=True); c2.plotly_chart(px.histogram(residuals,x="Residual",title="Backtest residual distribution",template="plotly_white"),use_container_width=True)
        st.write(f"Mean residual (actual − predicted): ₹{residuals.Residual.mean():,.2f}; positive indicates systematic under-forecasting on these origins.")
        if len(residuals)>=8:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lag=min(6,len(residuals)//2)
            lb=acorr_ljungbox(residuals.Residual,lags=[lag],return_df=True)
            st.caption(f"Ljung–Box residual autocorrelation check at lag {lag}: p={lb['lb_pvalue'].iloc[0]:.4g}. Small samples and few origins make this diagnostic low-power.")
    st.caption("Backtesting uses expanding training windows with one-step-ahead predictions over up to six recent monthly origins. ARIMA is attempted with at least 18 training observations; seasonal ARIMA requires at least 36. Model selection uses rolling WAPE. Prophet is omitted because this project has no supplied history to justify its additional dependency/model complexity.")


def recommendations(df):
    ps=product_summary(df)
    if ps.empty: st.info("Recommendations require Product, Category, Sales, and dated records; no generic suggestions are generated."); return
    msgs=[]
    if "Date" in df and len(df):
        d=clean_frame(df).dropna(subset=["Date"])
        if not d.empty:
            maxdate=d.Date.max(); recent=d[d.Date>=maxdate-pd.DateOffset(months=2)]; prior=d[(d.Date<maxdate-pd.DateOffset(months=2))&(d.Date>=maxdate-pd.DateOffset(months=4))]
            if not recent.empty and not prior.empty:
                growth=recent.groupby("Category").Sales.sum().div(prior.groupby("Category").Sales.sum().replace(0,float("nan"))).sub(1).mul(100).dropna()
                for cat,val in growth.nsmallest(3).items():
                    if val<0: msgs.append(f"**Review {cat}:** revenue is down {abs(val):.1f}% between the selected recent and preceding two-month windows.")
                increasing=(recent.groupby("Product").Sales.sum()-prior.groupby("Product").Sales.sum()).sort_values(ascending=False)
                if len(increasing) and increasing.iloc[0]>0: msgs.append(f"**Monitor demand for {increasing.index[0]}:** revenue rose by ₹{increasing.iloc[0]:,.2f} in the recent two-month window versus the preceding two months.")
    top=ps.iloc[0]; msgs.append(f"**Revenue concentration:** {top.Product} leads with ₹{top.Revenue:,.2f}, {top['Contribution %']:.1f}% of observed product revenue.")
    tail=ps.iloc[-1]; msgs.append(f"**Investigate the lowest-revenue product:** {tail.Product} has ₹{tail.Revenue:,.2f} in observed revenue; consider availability, lifecycle and margin before changing assortment.")
    rules=basket_rules(df)
    if not rules.empty:
        row=rules.iloc[0]; msgs.append(f"**Cross-sell candidate:** orders containing {row['If purchased']} also contain {row['Also purchased']} with lift {row.Lift:.2f} (support {row.Support:.1%}); validate commercially before acting.")
    for msg in msgs: st.markdown("- "+msg)
    if not msgs: st.info("Not enough comparable history for category/product trend recommendations. Showing only directly observed revenue leaders.")
    st.caption("Recommendations are tied to measured revenue or observed basket rules and do not establish causality or profitability.")


def data_management():
    st.subheader("Data management")
    st.markdown("Required fields: `Date`, `Product`, `Category`, `Sales`. Optional fields: `Order ID`, `Customer ID`, `Quantity`, `Unit Price`, `Cost`, `Profit`, `Customer Age`, `Customer Gender`. Use one row per product line for basket analysis; never synthesize missing historical identifiers.")
    with st.expander("Add a sales record (legacy manual entry retained)",expanded=True):
        with st.form("manual_entry"):
            c1,c2,c3=st.columns(3); dt=c1.date_input("Date",date.today()); product=c2.text_input("Product"); category=c3.text_input("Category",value="Uncategorized")
            c4,c5,c6=st.columns(3); sales=c4.number_input("Sales",min_value=0.0,step=1.0); age=c5.number_input("Customer Age (optional)",min_value=0,max_value=120,value=0); gender=c6.selectbox("Customer Gender (optional)",["Not provided","Male","Female","Other"])
            c7,c8,c9=st.columns(3); order=c7.text_input("Order ID (optional)"); customer=c8.text_input("Customer ID (optional)"); qty=c9.number_input("Quantity (optional; 0 means unknown)",min_value=0.0,step=1.0)
            unit=st.number_input("Unit Price (optional; leave 0 if unknown)",min_value=0.0,step=1.0); cost=st.number_input("Cost (optional; leave 0 if unknown)",min_value=0.0,step=1.0)
            submitted=st.form_submit_button("Add record")
        if submitted:
            if not product.strip() or not category.strip(): st.error("Product and Category are required.")
            else:
                row={"Date":dt,"Product":product.strip(),"Category":category.strip(),"Sales":sales,"Customer Age":age if age else None,"Customer Gender":None if gender=="Not provided" else gender,"Order ID":order.strip() or None,"Customer ID":customer.strip() or None,"Quantity":qty or None,"Unit Price":unit or None,"Cost":cost or None}
                try: db.append(pd.DataFrame([row])); st.success("Record saved to SQLite."); st.rerun()
                except Exception as exc: st.error(f"Could not save record: {exc}")
    upload=st.file_uploader("Import CSV or Excel",type=["csv","xlsx","xls"])
    if upload:
        try:
            incoming=pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
            clean,notes=validate_data(incoming)
            st.write(f"Preview: {len(clean):,} valid row(s) from {len(incoming):,} row(s).")
            st.dataframe(clean.head(20),use_container_width=True,hide_index=True)
            for note in notes: st.warning(note)
            mode=st.radio("Import behavior",["Append","Replace all current records"],horizontal=True)
            if st.button("Import validated data"):
                if mode.startswith("Replace"): db.replace(clean)
                else: db.append(clean)
                st.success(f"Imported {len(clean):,} rows into SQLite."); st.rerun()
        except Exception as exc: st.error(f"Import validation failed: {exc}")
    data=db.load()
    st.subheader("Current records")
    st.write(f"{len(data):,} row(s); exact duplicate rows are detected and reported but not automatically removed.")
    if not data.empty:
        dup=int(data.duplicated().sum())
        if dup: st.warning(f"Detected {dup} exact duplicate rows. Inspect before removal; identical sales lines may be valid.")
        st.dataframe(data,use_container_width=True,hide_index=True)
        csv=data.to_csv(index=False).encode("utf-8")
        xbuf=io.BytesIO(); data.to_excel(xbuf,index=False)
        c1,c2=st.columns(2); c1.download_button("Download CSV",csv,"sales_data.csv","text/csv"); c2.download_button("Download Excel",xbuf.getvalue(),"sales_data.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if page=="Executive Overview": overview(filtered)
elif page=="Sales Performance": sales_page(filtered)
elif page=="Product Analytics": product_page(filtered)
elif page=="Customer Analytics": customer_page(filtered)
elif page=="Statistical Analysis": stats_page(filtered)
elif page=="Market Basket Analysis": basket_page(filtered)
elif page=="Sales Forecasting": forecast_page(filtered)
elif page=="Business Recommendations": recommendations(filtered)
elif page=="Data Management": data_management()
