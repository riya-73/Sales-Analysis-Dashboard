"""SQLite persistence and data validation, with automatic legacy Excel import."""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
import pandas as pd

BASE_COLUMNS = ["Date", "Product", "Category", "Sales", "Customer Age", "Customer Gender"]
OPTIONAL_COLUMNS = ["Order ID", "Customer ID", "Quantity", "Unit Price", "Cost", "Profit"]
ALL_COLUMNS = BASE_COLUMNS + OPTIONAL_COLUMNS
DB_COLUMNS = ["date", "product", "category", "sales", "customer_age", "customer_gender", "order_id", "customer_id", "quantity", "unit_price", "cost", "profit"]

class DataValidationError(ValueError):
    pass

def validate_data(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if data is None or data.empty:
        return pd.DataFrame(columns=ALL_COLUMNS), []
    df = data.copy()
    missing = [c for c in ["Date", "Product", "Category", "Sales"] if c not in df.columns]
    if missing:
        raise DataValidationError("Missing required fields: " + ", ".join(missing))
    notes = []
    for c in ALL_COLUMNS:
        if c not in df.columns: df[c] = pd.NA
    df = df[ALL_COLUMNS]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = int(df.Date.isna().sum())
    if bad_dates:
        df = df.dropna(subset=["Date"]); notes.append(f"Removed {bad_dates} row(s) with invalid/missing dates.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    bad_sales = int(df.Sales.isna().sum())
    if bad_sales:
        df = df.dropna(subset=["Sales"]); notes.append(f"Removed {bad_sales} row(s) with non-numeric/missing Sales.")
    if (df.Sales < 0).any(): notes.append(f"{int((df.Sales < 0).sum())} negative Sales value(s) retained (e.g. refunds).")
    for c in ["Quantity", "Unit Price", "Cost", "Profit", "Customer Age"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    duplicates = int(df.duplicated().sum())
    if duplicates: notes.append(f"Detected {duplicates} exact duplicate row(s); duplicates were retained to avoid deleting potentially valid line items. Remove them explicitly if appropriate.")
    df["Date"] = df.Date.dt.strftime("%Y-%m-%d")
    return df, notes

class DataStore:
    def __init__(self, path: str = "sales_dashboard.sqlite3"):
        self.path = Path(path)
        self._initialize()
        self._migrate_excel()

    def _connect(self):
        con = sqlite3.connect(self.path); con.execute("PRAGMA journal_mode=WAL"); return con

    def _initialize(self):
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS sales (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, product TEXT, category TEXT, sales REAL, customer_age REAL, customer_gender TEXT, order_id TEXT, customer_id TEXT, quantity REAL, unit_price REAL, cost REAL, profit REAL)")

    def _migrate_excel(self):
        with self._connect() as con:
            if con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]: return
        if os.path.exists("ecommerce_data.xlsx"):
            try: self.replace(pd.read_excel("ecommerce_data.xlsx"))
            except Exception: pass

    def load(self) -> pd.DataFrame:
        with self._connect() as con:
            df = pd.read_sql_query("SELECT date AS Date, product AS Product, category AS Category, sales AS Sales, customer_age AS 'Customer Age', customer_gender AS 'Customer Gender', order_id AS 'Order ID', customer_id AS 'Customer ID', quantity AS Quantity, unit_price AS 'Unit Price', cost AS Cost, profit AS Profit FROM sales ORDER BY id", con)
        if not df.empty: df["Date"] = pd.to_datetime(df.Date, errors="coerce")
        return df

    def _write(self, df: pd.DataFrame, mode: str):
        normalized = df.copy()
        normalized.columns = DB_COLUMNS
        with self._connect() as con:
            if mode == "replace": con.execute("DELETE FROM sales")
            if not normalized.empty: normalized.to_sql("sales", con, if_exists="append", index=False, chunksize=500)

    def append(self, data: pd.DataFrame):
        df, _ = validate_data(data)
        self._write(df, "append")

    def replace(self, data: pd.DataFrame):
        df, _ = validate_data(data)
        self._write(df, "replace")
