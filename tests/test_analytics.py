import pandas as pd
from analytics import basket_rules, customer_summary, kpis, product_summary, growth_metrics
from storage import DataStore, validate_data


def sample():
    return pd.DataFrame({"Date":pd.to_datetime(["2024-01-05","2024-02-05","2024-02-08"]),"Product":["A","A","B"],"Category":["X","X","Y"],"Sales":[100.,150.,50.]})


def test_kpis_do_not_invent_orders_or_customers():
    result=kpis(sample())
    assert result["revenue"]==300
    assert result["orders"] is None
    assert result["revenue_per_customer"] is None


def test_product_contribution_sums_to_100():
    result=product_summary(sample())
    assert abs(result["Contribution %"].sum()-100)<1e-8


def test_customer_and_basket_require_identifiers():
    assert customer_summary(sample()).empty
    assert basket_rules(sample()).empty


def test_growth_has_month_comparison():
    result=growth_metrics(sample())
    assert result["MoM Growth %"].notna().any()
    assert result["MoM Growth %"].dropna().iloc[0]==100


def test_validate_drops_bad_required_rows_and_retains_duplicates():
    df=pd.DataFrame({"Date":["2024-01-01","not a date","2024-01-01"],"Product":["A","B","A"],"Category":["X","Y","X"],"Sales":[10,"bad",10]})
    clean,notes=validate_data(df)
    assert len(clean)==2 and clean.duplicated().sum()==1
    assert any("invalid/missing dates" in n for n in notes)
    assert any("duplicate" in n for n in notes)


def test_sqlite_roundtrip(tmp_path):
    db=DataStore(str(tmp_path/"test.sqlite"))
    db.append(sample())
    loaded=db.load()
    assert len(loaded)==3
    assert loaded.Sales.sum()==300
    assert loaded.iloc[0]["Product"]=="A"


def test_transaction_level_customer_frequency_and_basket_rules():
    tx=pd.DataFrame({
        "Date":pd.to_datetime(["2024-01-01","2024-01-01","2024-02-01","2024-02-01"]),
        "Product":["A","B","A","B"],"Category":["X"]*4,"Sales":[10,20,10,20],
        "Order ID":["o1","o1","o2","o2"],"Customer ID":["c1"]*4,
    })
    customers=customer_summary(tx)
    assert customers.iloc[0].Frequency==2
    rules=basket_rules(tx,min_support=.5,min_confidence=.5)
    assert len(rules)==2
    assert (rules.Lift==1).all()
