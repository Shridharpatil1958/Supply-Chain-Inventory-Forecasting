"""
Inventory optimization for the Auto Parts Inventory Forecasting project.

Turns a demand forecast (+ its uncertainty) into actionable inventory decisions:
  - Safety Stock:   Z * sigma_demand * sqrt(lead_time)
  - Reorder Point:  (avg_daily_demand * lead_time) + safety_stock
  - EOQ:            sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
  - Days of Cover:  current_stock / avg_daily_demand

ASSUMPTIONS (no real inventory/cost data exists in the source marketplace
dataset -- these are configurable, clearly-labeled defaults a real business
would replace with their own numbers):
  - lead_time_days: 14 (typical import lead time for auto parts)
  - order_cost: $25 per purchase order (administrative/shipping overhead)
  - holding_cost_pct: 20% of item value per year (industry rule-of-thumb range 15-30%)
  - service_level: 95% (Z = 1.645)
  - current_stock: NOT in the data -- the app lets a user enter their own
    current stock to get a live reorder recommendation; standalone reports
    assume a "0 stock" baseline and simply report the target coverage levels.

Run:
    python src/inventory_optimizer.py
"""
import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class InventoryAssumptions:
    lead_time_days: float = 14.0
    order_cost_usd: float = 25.0
    holding_cost_pct_annual: float = 0.20
    service_level: float = 0.95

    @property
    def z_score(self) -> float:
        return float(norm.ppf(self.service_level))


def compute_inventory_plan(avg_daily_demand: float, demand_std: float, avg_unit_price: float,
                            assumptions: InventoryAssumptions = InventoryAssumptions()) -> dict:
    """Compute safety stock, reorder point, and EOQ for one part/category.

    Args:
        avg_daily_demand: forecasted average daily demand (units)
        demand_std: standard deviation of DAILY demand (forecast residual std,
            or historical daily std if no forecast uncertainty is available)
        avg_unit_price: average price of the item (USD) -- drives holding cost
    """
    z = assumptions.z_score
    lt = assumptions.lead_time_days

    safety_stock = z * demand_std * np.sqrt(lt)
    reorder_point = (avg_daily_demand * lt) + safety_stock

    annual_demand = avg_daily_demand * 365
    holding_cost_per_unit = max(avg_unit_price * assumptions.holding_cost_pct_annual, 0.01)
    eoq = np.sqrt((2 * annual_demand * assumptions.order_cost_usd) / holding_cost_per_unit) if annual_demand > 0 else 0.0

    return {
        "avg_daily_demand": round(avg_daily_demand, 2),
        "demand_std": round(demand_std, 2),
        "safety_stock": round(safety_stock, 1),
        "reorder_point": round(reorder_point, 1),
        "eoq": round(eoq, 1),
        "holding_cost_per_unit_per_year": round(holding_cost_per_unit, 2),
    }


def days_of_cover(current_stock: float, avg_daily_demand: float) -> float:
    if avg_daily_demand <= 0:
        return float("inf")
    return round(current_stock / avg_daily_demand, 1)


def stockout_risk_level(days_cover: float, lead_time_days: float) -> str:
    if days_cover == float("inf"):
        return "No risk (no demand)"
    if days_cover < lead_time_days * 0.5:
        return "Critical - reorder now"
    elif days_cover < lead_time_days:
        return "High - reorder soon"
    elif days_cover < lead_time_days * 2:
        return "Moderate"
    else:
        return "Low"


def build_inventory_dashboard_table(assumptions: InventoryAssumptions = InventoryAssumptions()) -> pd.DataFrame:
    """Build a category-level inventory recommendation table using the
    demand forecast + historical price data. Assumes current_stock = 0
    (i.e. reports TARGET coverage levels; a real user plugs in actual stock
    in the Streamlit app for a live reorder-risk read-out).
    """
    forecast = pd.read_csv(os.path.join(PROCESSED_DIR, "forecast_next_7_days.csv"))
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_category_panel.csv"))

    avg_price_by_cat = panel.groupby("display_category")["avg_price_usd"].mean()

    rows = []
    for cat, sub in forecast.groupby("display_category"):
        avg_daily_demand = sub["predicted_listings"].mean()
        demand_std = ((sub["high_estimate"] - sub["low_estimate"]) / (2 * 1.28)).mean()  # invert the 80% CI back to std
        avg_price = float(avg_price_by_cat.get(cat, 50.0))

        plan = compute_inventory_plan(avg_daily_demand, demand_std, avg_price, assumptions)
        plan["display_category"] = cat
        plan["avg_unit_price_usd"] = round(avg_price, 2)
        rows.append(plan)

    df = pd.DataFrame(rows)
    cols = ["display_category", "avg_daily_demand", "demand_std", "avg_unit_price_usd",
            "safety_stock", "reorder_point", "eoq", "holding_cost_per_unit_per_year"]
    return df[cols].sort_values("reorder_point", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    assumptions = InventoryAssumptions()
    print("Assumptions:", assumptions)
    print(f"Z-score for {assumptions.service_level:.0%} service level: {assumptions.z_score:.3f}")
    print()

    table = build_inventory_dashboard_table(assumptions)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, "inventory_recommendations.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
