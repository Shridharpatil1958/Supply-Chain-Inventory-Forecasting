"""
Exploratory Data Analysis for the Auto Parts Inventory Forecasting project.

Generates:
  - Data-quality finding plot: full claimed history (2021-2024) vs the real
    dense window used for modeling (2024-09-07 to 2024-10-06)
  - Daily listing volume trend (overall + top categories)
  - Day-of-week pattern
  - Per-category demand pattern classification (smooth/intermittent/lumpy/erratic)
    using ADI (Average Demand Interval) and CV^2

Run:
    python src/eda.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import build_daily_panel, build_full_history_weekly_totals, MIN_HISTORY_LISTINGS

sns.set_theme(style="whitegrid")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def classify_demand_pattern(series: pd.Series) -> dict:
    """Classify a demand series using ADI / CV^2 (Syntetos-Boylan method)."""
    nonzero = series[series > 0]
    if len(nonzero) < 2:
        return {"ADI": np.nan, "CV2": np.nan, "pattern": "Insufficient data"}

    n_periods = len(series)
    n_nonzero = len(nonzero)
    adi = n_periods / n_nonzero
    cv2 = (nonzero.std() / nonzero.mean()) ** 2 if nonzero.mean() > 0 else np.nan

    if adi < 1.32 and cv2 < 0.49:
        pattern = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        pattern = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        pattern = "Erratic"
    else:
        pattern = "Lumpy"

    return {"ADI": round(adi, 2), "CV2": round(cv2, 2), "pattern": pattern}


def main():
    # --- Data quality finding: full claimed history vs reliable window ---
    full_weekly = build_full_history_weekly_totals()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(full_weekly["week_start"], full_weekly["listing_count"], color="#C44E52", linewidth=2)
    ax.axvspan(pd.Timestamp("2024-09-07"), pd.Timestamp("2024-10-06"), color="#4C72B0", alpha=0.15,
               label="Reliable window used for modeling (30 days)")
    ax.set_title("Data Quality Finding: 99.96% of listings fall in a single ~4-week window")
    ax.set_ylabel("New listings per week (full claimed range)")
    ax.legend(loc="upper left")
    save(fig, "00_data_quality_finding.png")

    panel = build_daily_panel(vehicle_scope="all")

    # --- Overall daily volume trend (reliable window only) ---
    daily_total = panel.groupby("date")["listing_count"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily_total["date"], daily_total["listing_count"], marker="o", color="#4C72B0")
    ax.set_title("Total Daily Listing Volume - Reliable Window (Sep 7 - Oct 6, 2024)")
    ax.set_ylabel("New listings per day")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "01_overall_daily_trend.png")

    # --- Day-of-week pattern ---
    panel["dow"] = panel["date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_totals = panel.groupby("dow")["listing_count"].mean().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=dow_totals.index, y=dow_totals.values, color="#55A868", ax=ax)
    ax.set_title("Average Daily Listings by Day of Week")
    ax.set_ylabel("Avg listings/day")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "02_day_of_week_pattern.png")

    # --- Top categories by volume ---
    totals = panel.groupby("display_category")["listing_count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=totals.values, y=totals.index, ax=ax, color="#4C72B0")
    ax.set_title("Total Listings by Category (30-day reliable window)")
    ax.set_xlabel("Total listings")
    save(fig, "03_category_volume_ranking.png")

    # --- Daily trend for top 6 categories ---
    top6 = totals.head(6).index.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, cat in zip(axes.flatten(), top6):
        sub = panel[panel["display_category"] == cat]
        ax.plot(sub["date"], sub["listing_count"], color="#55A868", marker="o", markersize=3)
        ax.set_title(cat, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Daily Listing Volume - Top 6 Categories", y=1.02)
    fig.tight_layout()
    save(fig, "04_top6_category_trends.png")

    # --- Price overview ---
    fig, ax = plt.subplots(figsize=(9, 6))
    price_by_cat = panel.groupby("display_category")["avg_price_usd"].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=price_by_cat.values, y=price_by_cat.index, color="#DD8452", ax=ax)
    ax.set_title("Average Listing Price (USD) by Category - Top 15")
    save(fig, "05_avg_price_by_category.png")

    # --- Demand pattern classification (ADI / CV2) on DAILY series ---
    print("\n--- Demand Pattern Classification (ADI / CV2, daily) ---")
    records = []
    for cat, sub in panel.groupby("display_category"):
        sub = sub.sort_values("date")
        result = classify_demand_pattern(sub["listing_count"])
        result["display_category"] = cat
        result["total_listings"] = int(sub["listing_count"].sum())
        result["avg_daily"] = round(sub["listing_count"].mean(), 2)
        records.append(result)

    pattern_df = pd.DataFrame(records)[["display_category", "total_listings", "avg_daily", "ADI", "CV2", "pattern"]]
    pattern_df = pattern_df.sort_values("total_listings", ascending=False)
    print(pattern_df.to_string(index=False))
    pattern_df.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                                    "demand_pattern_classification.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Smooth": "#55A868", "Intermittent": "#4C72B0", "Erratic": "#DD8452", "Lumpy": "#C44E52", "Insufficient data": "#999999"}
    for pattern, sub in pattern_df.dropna(subset=["ADI", "CV2"]).groupby("pattern"):
        ax.scatter(sub["ADI"], sub["CV2"], label=pattern, s=80, color=colors.get(pattern, "#333"))
    ax.axvline(1.32, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.49, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("ADI (Average Demand Interval, days)")
    ax.set_ylabel("CV2 (demand size variability)")
    ax.set_title("Demand Pattern Classification by Category (daily)")
    ax.legend()
    save(fig, "06_demand_pattern_scatter.png")

    print(f"\nCategories below the {MIN_HISTORY_LISTINGS}-listing modeling threshold "
          f"(excluded from per-category ML forecasting):")
    print(pattern_df[pattern_df["total_listings"] < MIN_HISTORY_LISTINGS]["display_category"].tolist())

    print("\nAll figures saved to reports/figures/")


if __name__ == "__main__":
    main()
