from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    sales = pd.read_csv(
        DATA_DIR / "sales_daily.csv",
        parse_dates=["date"]
    )

    customers = pd.read_csv(
        DATA_DIR / "customers.csv"
    )

    products = pd.read_csv(
        DATA_DIR / "products.csv"
    )

    stores = pd.read_csv(
        DATA_DIR / "stores.csv"
    )

    calendar = pd.read_csv(
        DATA_DIR / "calendar.csv",
        parse_dates=["date"]
    )

    return (
        sales,
        customers,
        products,
        stores,
        calendar
    )


# ==========================================================
# DATA QUALITY
# ==========================================================

def data_quality_report(df):

    report = pd.DataFrame({

        "column": df.columns,

        "dtype": [
            str(x)
            for x in df.dtypes
        ],

        "missing": [
            df[c].isna().sum()
            for c in df.columns
        ],

        "missing_pct": [
            round(
                df[c].isna().mean() * 100,
                2
            )
            for c in df.columns
        ],

        "unique": [
            df[c].nunique()
            for c in df.columns
        ]
    })

    report.to_csv(
        DATA_DIR / "data_quality_report.csv",
        index=False
    )

    return report


# ==========================================================
# CLEANING
# ==========================================================

def clean_sales(sales):

    sales = sales.copy()

    # Remove duplicates

    sales = sales.drop_duplicates()

    numeric_columns = [

        "selling_price",

        "competitor_price",

        "discount_rate",

        "demand_units",

        "units_sold",

        "inventory_available",

        "rainfall_mm",

        "temperature_c",

        "revenue",

        "profit"
    ]

    for column in numeric_columns:

        sales[column] = pd.to_numeric(
            sales[column],
            errors="coerce"
        )

        sales[column] = (
            sales[column]
            .fillna(
                sales[column].median()
            )
        )

    # Prevent negative values

    for column in [

        "units_sold",

        "demand_units",

        "inventory_available",

        "revenue"
    ]:

        sales[column] = (
            sales[column]
            .clip(lower=0)
        )

    # Price gap

    sales["price_gap_pct"] = (

        (
            sales["competitor_price"]
            -
            sales["selling_price"]
        )
        /
        sales["competitor_price"]
    )

    # Calendar features

    sales["year"] = (
        sales["date"].dt.year
    )

    sales["month"] = (
        sales["date"].dt.month
    )

    sales["day"] = (
        sales["date"].dt.day
    )

    sales["dayofweek"] = (
        sales["date"].dt.dayofweek
    )

    sales["weekend"] = (
        sales["dayofweek"] >= 5
    ).astype(int)

    # Recalculate financial metrics

    sales["revenue"] = (
        sales["units_sold"]
        *
        sales["selling_price"]
    )

    sales.to_csv(
        DATA_DIR / "sales_clean.csv",
        index=False
    )

    return sales


# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

def customer_segmentation(
    sales,
    customers
):

    customer_sales = sales[
        sales["customer_id"].notna()
    ].copy()

    rfm = (

        customer_sales

        .groupby("customer_id")

        .agg(

            frequency=(
                "date",
                "count"
            ),

            total_units=(
                "units_sold",
                "sum"
            ),

            monetary=(
                "revenue",
                "sum"
            ),

            average_order_value=(
                "revenue",
                "mean"
            ),

            average_discount=(
                "discount_rate",
                "mean"
            )
        )

        .reset_index()
    )

    features = [

        "frequency",

        "total_units",

        "monetary",

        "average_order_value",

        "average_discount"
    ]

    X = rfm[features].fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    )

    rfm["cluster"] = model.fit_predict(
        X_scaled
    )

    cluster_value = (

        rfm

        .groupby("cluster")["monetary"]

        .mean()

        .sort_values()
    )

    names = [

        "Low Value",

        "Regular",

        "High Value",

        "VIP"
    ]

    segment_map = {

        cluster: names[i]

        for i, cluster

        in enumerate(
            cluster_value.index
        )
    }

    rfm["segment"] = (
        rfm["cluster"]
        .map(segment_map)
    )

    result = customers.merge(
        rfm,
        on="customer_id",
        how="left"
    )

    result["segment"] = (
        result["segment"]
        .fillna("New Customer")
    )

    result.to_csv(
        DATA_DIR / "customer_segments.csv",
        index=False
    )

    return result


# ==========================================================
# EDA
# ==========================================================

def generate_eda_tables(sales):

    # Monthly

    monthly = (

        sales

        .groupby(
            sales["date"]
            .dt.to_period("M")
        )

        .agg(

            revenue=(
                "revenue",
                "sum"
            ),

            units=(
                "units_sold",
                "sum"
            ),

            profit=(
                "profit",
                "sum"
            ),

            demand=(
                "demand_units",
                "sum"
            )
        )

        .reset_index()
    )

    monthly["month"] = (
        monthly["date"]
        .astype(str)
    )

    monthly.to_csv(
        DATA_DIR / "monthly_summary.csv",
        index=False
    )

    # Products

    product_summary = (

        sales

        .groupby("product_id")

        .agg(

            units=(
                "units_sold",
                "sum"
            ),

            demand=(
                "demand_units",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            ),

            profit=(
                "profit",
                "sum"
            ),

            stockouts=(
                "stockout",
                "sum"
            )
        )

        .reset_index()

        .sort_values(
            "revenue",
            ascending=False
        )
    )

    product_summary.to_csv(
        DATA_DIR / "product_summary.csv",
        index=False
    )

    # Promotion

    promotion_summary = (

        sales

        .groupby("promotion")

        .agg(

            average_units=(
                "units_sold",
                "mean"
            ),

            average_demand=(
                "demand_units",
                "mean"
            ),

            revenue=(
                "revenue",
                "sum"
            ),

            profit=(
                "profit",
                "sum"
            )
        )

        .reset_index()
    )

    promotion_summary.to_csv(
        DATA_DIR / "promotion_summary.csv",
        index=False
    )

    # Weather

    weather_summary = (

        sales

        .groupby("weather")

        .agg(

            units=(
                "units_sold",
                "sum"
            ),

            demand=(
                "demand_units",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )

        .reset_index()
    )

    weather_summary.to_csv(
        DATA_DIR / "weather_summary.csv",
        index=False
    )

    # Festival

    festival_summary = (

        sales

        .groupby("festival")

        .agg(

            units=(
                "units_sold",
                "sum"
            ),

            demand=(
                "demand_units",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )

        .reset_index()

        .sort_values(
            "revenue",
            ascending=False
        )
    )

    festival_summary.to_csv(
        DATA_DIR / "festival_summary.csv",
        index=False
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\nLoading data...")

    (
        sales,
        customers,
        products,
        stores,
        calendar
    ) = load_data()

    print(
        f"Original rows: {len(sales):,}"
    )

    print("\nCreating data quality report...")

    data_quality_report(sales)

    print("\nCleaning data...")

    sales_clean = clean_sales(
        sales
    )

    print(
        f"Clean rows: {len(sales_clean):,}"
    )

    print(
        "\nCreating customer segments..."
    )

    segments = customer_segmentation(
        sales_clean,
        customers
    )

    print(
        segments["segment"]
        .value_counts()
    )

    print(
        "\nCreating EDA summaries..."
    )

    generate_eda_tables(
        sales_clean
    )

    print(
        "\nPipeline completed successfully."
    )


if __name__ == "__main__":
    main()