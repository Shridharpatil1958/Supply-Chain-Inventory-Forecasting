"""
Demand forecasting for the Auto Parts Inventory Forecasting project.

Given only a 30-day reliable window (see data_prep.py), this forecasts a
SHORT horizon (next 7 days) per category using recent trend + day-of-week,
NOT annual seasonality (there isn't enough history for that -- documented
in the README).

Approach: a single global XGBoost model across all categories (category
as a feature) using lag + rolling + calendar features, compared against two
naive baselines. Global models generalize much better than one-model-per-category
when each category only has 30 daily observations.

Run:
    python src/forecast.py
Outputs:
    models/demand_model.pkl
    models/demand_feature_columns.json
    models/demand_metrics.json
    data/processed/forecast_next_7_days.csv
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from data_prep import build_daily_panel, MIN_HISTORY_LISTINGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)

FORECAST_HORIZON_DAYS = 7
TEST_HOLDOUT_DAYS = 7  # last 7 days of the 30-day window used as test set
RANDOM_STATE = 42


def add_time_series_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features per category. Panel must be
    sorted by (display_category, date) and have zero-filled gaps already."""
    df = panel.sort_values(["display_category", "date"]).copy()
    g = df.groupby("display_category")["listing_count"]

    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = g.shift(lag)

    df["rolling_mean_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["rolling_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_std_7"] = g.shift(1).rolling(7).std().reset_index(level=0, drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days  # captures the ramp-up trend

    return df


def main():
    panel = build_daily_panel(vehicle_scope="all")

    # Restrict modeling to categories with enough volume to be meaningful
    totals = panel.groupby("display_category")["listing_count"].sum()
    modelable_categories = totals[totals >= MIN_HISTORY_LISTINGS].index.tolist()
    print(f"Modeling {len(modelable_categories)}/{panel['display_category'].nunique()} categories "
          f"(>= {MIN_HISTORY_LISTINGS} total listings): {modelable_categories}")

    panel = panel[panel["display_category"].isin(modelable_categories)].copy()

    df = add_time_series_features(panel)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_7",
        "rolling_mean_3", "rolling_mean_7", "rolling_std_7",
        "day_of_week", "is_weekend", "days_since_start",
    ]
    cat_dummies = pd.get_dummies(df["display_category"], prefix="cat")
    feature_cols_full = feature_cols + list(cat_dummies.columns)

    model_df = pd.concat([df, cat_dummies], axis=1).dropna(subset=feature_cols)

    max_date = model_df["date"].max()
    test_start = max_date - pd.Timedelta(days=TEST_HOLDOUT_DAYS - 1)

    train = model_df[model_df["date"] < test_start]
    test = model_df[model_df["date"] >= test_start]
    print(f"Train: {train.shape[0]} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {test.shape[0]} rows ({test['date'].min().date()} to {test['date'].max().date()})")

    X_train, y_train = train[feature_cols_full], train["listing_count"]
    X_test, y_test = test[feature_cols_full], test["listing_count"]

    # --- Baselines ---
    naive_pred = test["lag_1"]  # "tomorrow = today"
    seasonal_naive_pred = test["lag_7"]  # "same weekday last week"

    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    results = {
        "NaiveLag1": {"RMSE": rmse(y_test, naive_pred), "MAE": float(mean_absolute_error(y_test, naive_pred))},
        "SeasonalNaiveLag7": {"RMSE": rmse(y_test, seasonal_naive_pred), "MAE": float(mean_absolute_error(y_test, seasonal_naive_pred))},
    }

    # --- XGBoost (global model across categories) ---
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)
    xgb_pred = np.clip(xgb_pred, 0, None)  # demand can't be negative
    results["XGBoost"] = {"RMSE": rmse(y_test, xgb_pred), "MAE": float(mean_absolute_error(y_test, xgb_pred))}

    print("\n--- Forecast accuracy on last 7 held-out days ---")
    print(pd.DataFrame(results).T.round(2))

    # Residual std per category (used for safety-stock calc downstream)
    test = test.copy()
    test["pred"] = xgb_pred
    test["residual"] = test["listing_count"] - test["pred"]
    residual_std_by_cat = test.groupby("display_category")["residual"].std().fillna(test["residual"].std())

    # Retrain on ALL data for the actual forward forecast
    X_all, y_all = model_df[feature_cols_full], model_df["listing_count"]
    final_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    final_model.fit(X_all, y_all)

    # --- Recursive forecast for the next FORECAST_HORIZON_DAYS days, per category ---
    print(f"\nGenerating recursive {FORECAST_HORIZON_DAYS}-day forecast per category...")
    forecasts = []
    for cat in modelable_categories:
        hist = df[df["display_category"] == cat].sort_values("date").copy()
        history_values = hist["listing_count"].tolist()
        last_date = hist["date"].max()

        for step in range(1, FORECAST_HORIZON_DAYS + 1):
            future_date = last_date + pd.Timedelta(days=step)
            lag_1 = history_values[-1]
            lag_2 = history_values[-2]
            lag_3 = history_values[-3]
            lag_7 = history_values[-7]
            rolling_mean_3 = np.mean(history_values[-3:])
            rolling_mean_7 = np.mean(history_values[-7:])
            rolling_std_7 = np.std(history_values[-7:])
            day_of_week = future_date.dayofweek
            is_weekend = int(day_of_week in [5, 6])
            days_since_start = (future_date - df["date"].min()).days

            row = {c: 0 for c in feature_cols_full}
            row.update({
                "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_7": lag_7,
                "rolling_mean_3": rolling_mean_3, "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7, "day_of_week": day_of_week,
                "is_weekend": is_weekend, "days_since_start": days_since_start,
            })
            cat_col = f"cat_{cat}"
            if cat_col in row:
                row[cat_col] = 1

            X_future = pd.DataFrame([row])[feature_cols_full]
            pred = max(0, float(final_model.predict(X_future)[0]))

            residual_std = float(residual_std_by_cat.get(cat, test["residual"].std()))
            forecasts.append({
                "display_category": cat,
                "forecast_date": future_date.date().isoformat(),
                "days_ahead": step,
                "predicted_listings": round(pred, 1),
                "low_estimate": round(max(0, pred - 1.28 * residual_std), 1),   # ~80% interval
                "high_estimate": round(pred + 1.28 * residual_std, 1),
            })
            history_values.append(pred)  # feed prediction back in for recursive lags

    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(os.path.join(PROCESSED_DIR, "forecast_next_7_days.csv"), index=False)
    print(forecast_df.head(14))

    # Save artifacts
    joblib.dump(final_model, os.path.join(MODELS_DIR, "demand_model.pkl"))
    with open(os.path.join(MODELS_DIR, "demand_feature_columns.json"), "w") as f:
        json.dump(feature_cols_full, f, indent=2)
    with open(os.path.join(MODELS_DIR, "demand_metrics.json"), "w") as f:
        json.dump({
            "comparison": results,
            "modelable_categories": modelable_categories,
            "residual_std_by_category": residual_std_by_cat.to_dict(),
            "train_window": [str(train["date"].min().date()), str(train["date"].max().date())],
            "test_window": [str(test["date"].min().date()), str(test["date"].max().date())],
        }, f, indent=2)

    print("\nSaved: models/demand_model.pkl, models/demand_feature_columns.json, "
          "models/demand_metrics.json, data/processed/forecast_next_7_days.csv")


if __name__ == "__main__":
    main()
