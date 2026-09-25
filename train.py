from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ==========================================================
# BUILD FORECAST TABLE
# ==========================================================

def build_training_table():

    sales = pd.read_csv(
        DATA_DIR / "sales_clean.csv",
        parse_dates=["date"]
    )

    # Aggregate by store/product/day

    daily = (

        sales

        .groupby(
            [
                "date",
                "store_id",
                "product_id"
            ],
            as_index=False
        )

        .agg(

            units_sold=(
                "units_sold",
                "sum"
            ),

            selling_price=(
                "selling_price",
                "mean"
            ),

            competitor_price=(
                "competitor_price",
                "mean"
            ),

            promotion=(
                "promotion",
                "max"
            ),

            discount_rate=(
                "discount_rate",
                "mean"
            ),

            rainfall_mm=(
                "rainfall_mm",
                "mean"
            ),

            temperature_c=(
                "temperature_c",
                "mean"
            ),

            stockout=(
                "stockout",
                "max"
            )
        )
    )

    daily = daily.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    )

    group = [
        "store_id",
        "product_id"
    ]

    # ------------------------------------------------------
    # Lag features
    # ------------------------------------------------------

    for lag in [
        1,
        7,
        14,
        28
    ]:

        daily[
            f"lag_{lag}"
        ] = (

            daily

            .groupby(group)[
                "units_sold"
            ]

            .shift(lag)
        )

    # ------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------

    daily["rolling_7"] = (

        daily

        .groupby(group)[
            "units_sold"
        ]

        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .mean()
        )
    )

    daily["rolling_28"] = (

        daily

        .groupby(group)[
            "units_sold"
        ]

        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                28,
                min_periods=1
            )
            .mean()
        )
    )

    # ------------------------------------------------------
    # Calendar
    # ------------------------------------------------------

    daily["year"] = (
        daily["date"].dt.year
    )

    daily["month"] = (
        daily["date"].dt.month
    )

    daily["dayofweek"] = (
        daily["date"].dt.dayofweek
    )

    daily["weekend"] = (
        daily["dayofweek"] >= 5
    ).astype(int)

    # ------------------------------------------------------
    # Price gap
    # ------------------------------------------------------

    daily["price_gap"] = (

        (
            daily["competitor_price"]
            -
            daily["selling_price"]
        )
        /
        daily["competitor_price"]
    )

    # Remove rows without historical data

    daily = daily.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_7",
            "rolling_28"
        ]
    )

    return daily


# ==========================================================
# TRAIN
# ==========================================================

def train_model():

    data = build_training_table()

    features = [

        "store_id",

        "product_id",

        "promotion",

        "discount_rate",

        "rainfall_mm",

        "temperature_c",

        "month",

        "dayofweek",

        "weekend",

        "price_gap",

        "lag_1",

        "lag_7",

        "lag_14",

        "lag_28",

        "rolling_7",

        "rolling_28"
    ]

    target = "units_sold"

    X = data[features]

    y = data[target]

    categorical = [
        "store_id",
        "product_id"
    ]

    numerical = [
        c
        for c in features
        if c not in categorical
    ]

    preprocessing = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical
            ),

            (
                "numerical",

                "passthrough",

                numerical
            )
        ]
    )

    model = GradientBoostingRegressor(

        n_estimators=250,

        learning_rate=0.05,

        max_depth=3,

        min_samples_leaf=4,

        random_state=42,

        loss="huber"
    )

    pipeline = Pipeline([

        (
            "preprocessing",
            preprocessing
        ),

        (
            "model",
            model
        )
    ])

    # ------------------------------------------------------
    # Time-based split
    # ------------------------------------------------------

    cutoff = data["date"].quantile(
        0.80
    )

    train_mask = (
        data["date"] <= cutoff
    )

    test_mask = (
        data["date"] > cutoff
    )

    X_train = X[train_mask]

    X_test = X[test_mask]

    y_train = y[train_mask]

    y_test = y[test_mask]

    print(
        "\nTraining rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )

    # ------------------------------------------------------
    # Fit
    # ------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = pd.DataFrame({

        "metric": [
            "MAE",
            "RMSE",
            "R2"
        ],

        "value": [
            mae,
            rmse,
            r2
        ]
    })

    print("\nModel performance")

    print(
        metrics
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_DIR / "demand_forecast.pkl"
    )

    metrics.to_csv(
        MODEL_DIR / "metrics.csv",
        index=False
    )

    # Save test predictions

    result = data.loc[
        test_mask,
        [
            "date",
            "store_id",
            "product_id",
            "units_sold"
        ]
    ].copy()

    result["prediction"] = predictions

    result["error"] = (
        result["units_sold"]
        -
        result["prediction"]
    )

    result.to_csv(
        DATA_DIR / "forecast_predictions.csv",
        index=False
    )

    print(
        "\nModel saved successfully."
    )


if __name__ == "__main__":

    train_model()