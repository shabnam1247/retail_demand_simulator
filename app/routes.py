from flask import (
    Blueprint,
    render_template,
    jsonify,
    request
)

from pathlib import Path

import pandas as pd
import numpy as np
import joblib


bp = Blueprint(
    "main",
    __name__
)


BASE_DIR = Path(
    __file__
).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"


# ==========================================================
# LOAD SALES
# ==========================================================

def load_sales():

    path = DATA_DIR / "sales_clean.csv"

    if not path.exists():

        path = DATA_DIR / "sales_daily.csv"

    return pd.read_csv(
        path,
        parse_dates=["date"]
    )


# ==========================================================
# HOME
# ==========================================================

@bp.route("/")
def dashboard():

    sales = load_sales()

    revenue = sales["revenue"].sum()

    units = sales["units_sold"].sum()

    profit = sales["profit"].sum()

    orders = (
        sales["units_sold"] > 0
    ).sum()

    stockouts = (
        sales["stockout"] == 1
    ).sum()

    avg_order_value = (
        revenue / orders
        if orders > 0
        else 0
    )

    return render_template(

        "dashboard.html",

        kpi={

            "revenue": revenue,

            "units": units,

            "profit": profit,

            "orders": orders,

            "stockouts": stockouts,

            "avg_order_value":
                avg_order_value
        }
    )


# ==========================================================
# SUMMARY API
# ==========================================================

@bp.route("/api/summary")
def summary():

    sales = load_sales()

    # ------------------------------------------------------
    # Monthly
    # ------------------------------------------------------

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
            )
        )

        .reset_index()
    )

    monthly["month"] = (
        monthly["date"]
        .astype(str)
    )

    # ------------------------------------------------------
    # Products
    # ------------------------------------------------------

    products = (

        sales

        .groupby("product_id")

        .agg(

            units=(
                "units_sold",
                "sum"
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

        .sort_values(
            "revenue",
            ascending=False
        )

        .head(10)
    )

    # ------------------------------------------------------
    # Categories
    # ------------------------------------------------------

    product_file = DATA_DIR / "products.csv"

    products_master = pd.read_csv(
        product_file
    )

    category = (

        sales

        .merge(
            products_master[
                [
                    "product_id",
                    "category"
                ]
            ],
            on="product_id",
            how="left"
        )

        .groupby("category")

        .agg(

            revenue=(
                "revenue",
                "sum"
            ),

            units=(
                "units_sold",
                "sum"
            )
        )

        .reset_index()

        .sort_values(
            "revenue",
            ascending=False
        )
    )

    # ------------------------------------------------------
    # Segments
    # ------------------------------------------------------

    segment_file = (
        DATA_DIR /
        "customer_segments.csv"
    )

    if segment_file.exists():

        segments = (

            pd.read_csv(
                segment_file
            )

            .groupby("segment")

            .size()

            .reset_index(
                name="customers"
            )
        )

    else:

        segments = pd.DataFrame(
            columns=[
                "segment",
                "customers"
            ]
        )

    # ------------------------------------------------------
    # Promotion
    # ------------------------------------------------------

    promotion = (

        sales

        .groupby("promotion")

        .agg(

            avg_units=(
                "units_sold",
                "mean"
            ),

            avg_demand=(
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

    # ------------------------------------------------------
    # Weather
    # ------------------------------------------------------

    weather = (

        sales

        .groupby("weather")

        .agg(

            units=(
                "units_sold",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )

        .reset_index()
    )

    # ------------------------------------------------------
    # Festival
    # ------------------------------------------------------

    festival = (

        sales

        .groupby("festival")

        .agg(

            units=(
                "units_sold",
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

    return jsonify({

        "monthly":
            monthly.to_dict(
                "records"
            ),

        "products":
            products.to_dict(
                "records"
            ),

        "category":
            category.to_dict(
                "records"
            ),

        "segments":
            segments.to_dict(
                "records"
            ),

        "promotion":
            promotion.to_dict(
                "records"
            ),

        "weather":
            weather.to_dict(
                "records"
            ),

        "festival":
            festival.to_dict(
                "records"
            )
    })


# ==========================================================
# EDA
# ==========================================================

@bp.route("/eda")
def eda():

    return render_template(
        "eda.html"
    )


# ==========================================================
# SEGMENTS
# ==========================================================

@bp.route("/segments")
def segments():

    segment_file = (
        DATA_DIR /
        "customer_segments.csv"
    )

    if segment_file.exists():

        data = pd.read_csv(
            segment_file
        )

        rows = data.head(
            100
        ).to_dict(
            "records"
        )

    else:

        rows = []

    return render_template(
        "segments.html",
        rows=rows
    )


# ==========================================================
# FORECAST
# ==========================================================

@bp.route("/forecast")
def forecast():

    metrics_file = (
        MODEL_DIR /
        "metrics.csv"
    )

    if metrics_file.exists():

        metrics = pd.read_csv(
            metrics_file
        ).to_dict(
            "records"
        )

    else:

        metrics = []

    return render_template(
        "forecast.html",
        metrics=metrics
    )


# ==========================================================
# INVENTORY
# ==========================================================

@bp.route("/inventory")
def inventory():

    sales = load_sales()

    inventory = (

        sales

        .groupby(
            [
                "store_id",
                "product_id"
            ]
        )

        .agg(

            average_daily_demand=(
                "units_sold",
                "mean"
            ),

            total_demand=(
                "demand_units",
                "sum"
            ),

            stockout_days=(
                "stockout",
                "sum"
            ),

            average_inventory=(
                "inventory_available",
                "mean"
            )
        )

        .reset_index()
    )

    inventory["stockout_rate"] = (

        inventory["stockout_days"]
        /
        (
            len(
                sales["date"]
                .unique()
            )
        )
    )

    inventory["risk"] = np.select(

        [

            inventory["stockout_rate"]
            >= 0.20,

            inventory["stockout_rate"]
            >= 0.10
        ],

        [

            "High",

            "Medium"
        ],

        default="Low"
    )

    inventory = inventory.sort_values(
        "stockout_rate",
        ascending=False
    )

    return render_template(

        "inventory.html",

        rows=inventory.head(
            100
        ).to_dict(
            "records"
        )
    )


# ==========================================================
# FORECAST API
# ==========================================================

@bp.route(
    "/api/forecast",
    methods=["POST"]
)
def forecast_api():

    model_path = (
        MODEL_DIR /
        "demand_forecast.pkl"
    )

    if not model_path.exists():

        return jsonify({

            "error":
                "Forecast model not found. "
                "Run simulate.py, pipeline.py "
                "and train.py first."
        }), 400

    model = joblib.load(
        model_path
    )

    sales = load_sales()

    sales = sales.sort_values(
        "date"
    )

    last = sales.iloc[-1]

    # ------------------------------------------------------
    # Forecast next day
    # ------------------------------------------------------

    next_date = (
        last["date"]
        +
        pd.Timedelta(days=1)
    )

    X = pd.DataFrame([{

        "store_id":
            last["store_id"],

        "product_id":
            last["product_id"],

        "promotion":
            0,

        "discount_rate":
            0.0,

        "rainfall_mm":
            last["rainfall_mm"],

        "temperature_c":
            last["temperature_c"],

        "month":
            next_date.month,

        "dayofweek":
            next_date.dayofweek,

        "weekend":
            int(
                next_date.dayofweek
                >= 5
            ),

        "price_gap":
            0.0,

        "lag_1":
            last["units_sold"],

        "lag_7":
            last["units_sold"],

        "lag_14":
            last["units_sold"],

        "lag_28":
            last["units_sold"],

        "rolling_7":
            last["units_sold"],

        "rolling_28":
            last["units_sold"]
    }])

    prediction = model.predict(
        X
    )[0]

    prediction = max(
        0,
        float(prediction)
    )

    return jsonify({

        "forecast_date":
            next_date.strftime(
                "%Y-%m-%d"
            ),

        "store":
            last["store_id"],

        "product":
            last["product_id"],

        "forecast_units":
            round(
                prediction,
                2
            )
    })


# ==========================================================
# HEALTH CHECK
# ==========================================================

@bp.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "data_exists":
            (DATA_DIR / "sales_clean.csv")
            .exists(),

        "model_exists":
            (MODEL_DIR /
             "demand_forecast.pkl")
            .exists()
    })