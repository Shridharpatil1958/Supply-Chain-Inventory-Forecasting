"""
Streamlit dashboard: Auto Parts Demand Forecast & Inventory Planner.

Run:
    streamlit run app/streamlit_dashboard.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inventory_optimizer import InventoryAssumptions, compute_inventory_plan, days_of_cover, stockout_risk_level  # noqa: E402

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Auto Parts Demand & Inventory Planner", page_icon="🔧", layout="wide")


@st.cache_data
def load_data():
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_category_panel.csv"), parse_dates=["date"])
    forecast = pd.read_csv(os.path.join(PROCESSED_DIR, "forecast_next_7_days.csv"), parse_dates=["forecast_date"])
    pattern = pd.read_csv(os.path.join(PROCESSED_DIR, "demand_pattern_classification.csv"))
    with open(os.path.join(MODELS_DIR, "demand_metrics.json")) as f:
        metrics = json.load(f)
    return panel, forecast, pattern, metrics


panel, forecast, pattern, metrics = load_data()
modelable_categories = metrics["modelable_categories"]

st.title("🔧 Auto Parts Demand Forecast & Inventory Planner")
st.caption(
    "Built on a 30-day marketplace-listings snapshot (Sep 7 - Oct 6, 2024, all vehicle types). "
    "Listing volume is used as a demand-activity **proxy** -- there is no direct sales/stock data "
    "in the source dataset. See the README for full methodology and limitations."
)

with st.expander("⚠️ Data limitations (read before trusting these numbers)", expanded=False):
    st.markdown(f"""
- **Only 30 real days of data** (Sep 7 - Oct 6, 2024). 99.96% of all listings in the raw file fall
  in this window; older dates are ~40 stray rows, not real history. This means forecasts capture
  short-term trend + day-of-week, **not** annual seasonality (there isn't enough data for that).
- **"Demand" = new listing volume**, a proxy for market activity, not actual units sold.
- Forecast accuracy on the last 7 held-out days: XGBoost RMSE={metrics['comparison']['XGBoost']['RMSE']:.0f},
  vs. a naive "same as yesterday" baseline RMSE={metrics['comparison']['NaiveLag1']['RMSE']:.0f} —
  with this little data, XGBoost is roughly on par with simple baselines, not dramatically better.
- Inventory formulas (safety stock, reorder point, EOQ) use **configurable assumed defaults**
  for lead time, order cost, and holding cost — the marketplace data has no real cost fields.
""")

tab1, tab2 = st.tabs(["📈 Category Forecast", "📦 Inventory Planner"])

# ============ TAB 1: Forecast explorer ============
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        category = st.selectbox("Category", modelable_categories)
        cat_pattern = pattern[pattern["display_category"] == category]
        if not cat_pattern.empty:
            p = cat_pattern.iloc[0]
            st.metric("Demand pattern", p["pattern"])
            st.metric("Total listings (30d)", int(p["total_listings"]))
            st.metric("Avg per day", round(p["avg_daily"], 1))

    with col2:
        hist = panel[panel["display_category"] == category].sort_values("date")
        fut = forecast[forecast["display_category"] == category].sort_values("forecast_date")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(hist["date"], hist["listing_count"], label="Historical", color="#4C72B0", marker="o", markersize=3)
        ax.plot(fut["forecast_date"], fut["predicted_listings"], label="Forecast (7d)", color="#C44E52",
                 marker="o", markersize=4, linestyle="--")
        ax.fill_between(fut["forecast_date"], fut["low_estimate"], fut["high_estimate"],
                          color="#C44E52", alpha=0.15, label="~80% interval")
        ax.axvline(hist["date"].max(), color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Daily Listings — {category}")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    st.dataframe(fut[["forecast_date", "predicted_listings", "low_estimate", "high_estimate"]],
                 use_container_width=True, hide_index=True)

# ============ TAB 2: Inventory planner ============
with tab2:
    st.subheader("Reorder Assumptions")
    c1, c2, c3, c4 = st.columns(4)
    lead_time = c1.number_input("Lead time (days)", min_value=1, max_value=90, value=14)
    order_cost = c2.number_input("Order cost ($)", min_value=1.0, max_value=500.0, value=25.0)
    holding_pct = c3.number_input("Holding cost (% of value/yr)", min_value=0.05, max_value=0.6, value=0.20, step=0.01)
    service_level = c4.selectbox("Target service level", [0.90, 0.95, 0.975, 0.99], index=1,
                                   format_func=lambda x: f"{x:.1%}")

    assumptions = InventoryAssumptions(
        lead_time_days=lead_time, order_cost_usd=order_cost,
        holding_cost_pct_annual=holding_pct, service_level=service_level,
    )

    st.divider()
    st.subheader("Per-Category Reorder Recommendations")
    st.caption("Enter your current stock for each category to get a live stockout-risk read-out. "
               "Values default to 0 (shows target coverage levels only).")

    avg_price_by_cat = panel.groupby("display_category")["avg_price_usd"].mean()

    rows = []
    for cat in modelable_categories:
        fut_cat = forecast[forecast["display_category"] == cat]
        avg_daily_demand = fut_cat["predicted_listings"].mean()
        demand_std = ((fut_cat["high_estimate"] - fut_cat["low_estimate"]) / (2 * 1.28)).mean()
        avg_price = float(avg_price_by_cat.get(cat, 50.0))

        plan = compute_inventory_plan(avg_daily_demand, demand_std, avg_price, assumptions)
        rows.append({
            "Category": cat,
            "Avg Daily Demand": plan["avg_daily_demand"],
            "Safety Stock": plan["safety_stock"],
            "Reorder Point": plan["reorder_point"],
            "EOQ": plan["eoq"],
            "Avg Price ($)": round(avg_price, 2),
        })

    plan_df = pd.DataFrame(rows).sort_values("Reorder Point", ascending=False)

    edited = st.data_editor(
        plan_df.assign(**{"Current Stock": 0}),
        column_config={
            "Current Stock": st.column_config.NumberColumn("Current Stock", min_value=0, step=1),
        },
        disabled=[c for c in plan_df.columns],
        use_container_width=True,
        hide_index=True,
        key="inventory_editor",
    )

    edited["Days of Cover"] = edited.apply(
        lambda r: days_of_cover(r["Current Stock"], r["Avg Daily Demand"]), axis=1
    )
    edited["Risk"] = edited.apply(
        lambda r: stockout_risk_level(r["Days of Cover"], lead_time), axis=1
    )

    def highlight_risk(val):
        color_map = {
            "Critical - reorder now": "#f8d7da",
            "High - reorder soon": "#fff3cd",
            "Moderate": "#fff9e6",
            "Low": "#d4edda",
        }
        return f"background-color: {color_map.get(val, '')}"

    st.markdown("**Live stockout risk** (based on the Current Stock you entered above):")
    st.dataframe(
        edited[["Category", "Current Stock", "Days of Cover", "Risk", "Reorder Point", "Safety Stock", "EOQ"]]
        .style.map(highlight_risk, subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        "**Safety Stock** = Z x demand_std x sqrt(lead_time) — buffer for demand variability. "
        "**Reorder Point** = (avg_daily_demand x lead_time) + safety_stock — stock level that should trigger a new order. "
        "**EOQ** = sqrt(2 x annual_demand x order_cost / holding_cost) — order quantity that minimizes total cost."
    )
