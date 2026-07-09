"""
Data preparation for the Auto Parts Inventory Forecasting project.

IMPORTANT FRAMING NOTE:
This dataset is a marketplace LISTINGS dataset (parts posted for sale), not a
sales/inventory transaction log. There is no "units sold" or "stock level" field.
We use the *count of new listings per category per week* as a demand-activity
proxy -- a standard substitute when direct sales data isn't available (similar
to using search volume or job-posting volume as an economic activity proxy).
This assumption is documented here and flagged again in the README.

Produces a tidy weekly panel: (week, category_id, category_name, top_level_group,
listing_count, avg_price_usd, pct_new_condition) ready for time-series feature
engineering and forecasting.
"""
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Categories with fewer total listings than this across the whole history are
# too sparse to forecast reliably at daily granularity; they're kept in the
# panel for transparency but excluded from the modeling set in forecast.py
MIN_HISTORY_LISTINGS = 100

# DATA REALITY CHECK (see README): 99.96% of all listings fall between
# 2024-09-07 and 2024-10-06. Dates outside this window are sparse legacy/stray
# rows (41 total, spanning 2021-2023) that do not represent a real historical
# time series. We restrict the modeling panel to the dense, reliable window.
RELIABLE_WINDOW_START = "2024-09-07"
RELIABLE_WINDOW_END = "2024-10-06"  # 2024-10-07 is a partial/cut-off day, excluded


def load_raw_tables(raw_dir: str = RAW_DIR) -> dict:
    tables = {
        "applications": pd.read_csv(os.path.join(raw_dir, "applications.csv")),
        "product_category": pd.read_csv(os.path.join(raw_dir, "product_category.csv")),
        "application_status": pd.read_csv(os.path.join(raw_dir, "application_status.csv")),
        "vehicle_type": pd.read_csv(os.path.join(raw_dir, "vehicle_type.csv")),
        "seller": pd.read_csv(os.path.join(raw_dir, "seller.csv")),
    }
    return tables


def build_category_lookup(product_category: pd.DataFrame) -> pd.DataFrame:
    """Map every category_id (however deep) to its level-2 'display category'
    (the direct child of a top-level node) and its top-level group name.

    The raw hierarchy has some categories written in Georgian that duplicate an
    English-named sibling (e.g. 'Filters' and a Georgian equivalent). We keep
    them as distinct categories here since collapsing scripts reliably would
    require a translation step beyond the scope of this dataset -- flagged as
    a known limitation.
    """
    pc = product_category.copy()
    pc = pc.rename(columns={"id": "category_id"})

    top_level = pc[pc["parent_category_id"].isna()][["category_id", "category_name"]]
    top_level = top_level.rename(columns={"category_id": "top_id", "category_name": "top_level_group"})

    # level-2 = direct children of a top-level node
    level2 = pc[pc["parent_category_id"].isin(top_level["top_id"])].copy()
    level2 = level2.rename(columns={"category_id": "display_category_id", "category_name": "display_category"})
    level2 = level2.merge(top_level, left_on="parent_category_id", right_on="top_id", how="left")
    level2 = level2[["display_category_id", "display_category", "top_level_group"]]

    # Build a full id -> ancestor-at-level-2 mapping by walking up parent links
    id_to_parent = pc.set_index("category_id")["parent_category_id"].to_dict()
    id_to_name = pc.set_index("category_id")["category_name"].to_dict()
    level2_ids = set(level2["display_category_id"])
    top_ids = set(top_level["top_id"])

    def resolve(cat_id):
        """Walk up the tree until hitting a level-2 node or a top-level node itself."""
        seen = set()
        current = cat_id
        while current is not None and current not in seen:
            if current in level2_ids or current in top_ids:
                return current
            seen.add(current)
            current = id_to_parent.get(current)
        return cat_id  # fallback: leave unresolved ids as their own bucket

    pc["resolved_id"] = pc["category_id"].apply(resolve)

    # Map resolved_id -> (display_category, top_level_group); top-level nodes map to themselves
    display_map = level2.set_index("display_category_id")[["display_category", "top_level_group"]].to_dict("index")
    top_map = top_level.set_index("top_id")["top_level_group"].to_dict()

    def get_display(resolved_id):
        if resolved_id in display_map:
            d = display_map[resolved_id]
            return d["display_category"], d["top_level_group"]
        elif resolved_id in top_map:
            return id_to_name.get(resolved_id, "Unknown"), top_map[resolved_id]
        else:
            return "Unknown", "Unknown"

    pc[["display_category", "top_level_group"]] = pc["resolved_id"].apply(
        lambda r: pd.Series(get_display(r))
    )

    return pc[["category_id", "display_category", "top_level_group"]]


def build_full_history_weekly_totals() -> pd.DataFrame:
    """Unfiltered weekly totals across the ENTIRE claimed date range (2021-2024),
    used only to illustrate the data-quality finding in EDA (not used for modeling).
    """
    tables = load_raw_tables()
    apps = tables["applications"].copy()
    apps["app_register_date"] = pd.to_datetime(apps["app_register_date"], errors="coerce")
    apps = apps.dropna(subset=["app_register_date"])
    apps["week_start"] = apps["app_register_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    return apps.groupby("week_start").size().reset_index(name="listing_count")


def build_daily_panel(vehicle_scope: str = "all", restrict_to_reliable_window: bool = True) -> pd.DataFrame:
    """Build the daily (date, display_category) demand-proxy panel.

    Args:
        vehicle_scope: "all" (default, per user choice) or "cars_only"
        restrict_to_reliable_window: if True (default), only keeps
            RELIABLE_WINDOW_START..RELIABLE_WINDOW_END, since dates outside this
            window are sparse stray rows, not a real time series (see module docstring).
    """
    tables = load_raw_tables()
    apps = tables["applications"].copy()
    apps["app_register_date"] = pd.to_datetime(apps["app_register_date"], errors="coerce")
    apps = apps.dropna(subset=["app_register_date"])

    if vehicle_scope == "cars_only":
        apps = apps[apps["vehicle_type_id"] == 1]

    if restrict_to_reliable_window:
        apps = apps[
            (apps["app_register_date"] >= RELIABLE_WINDOW_START)
            & (apps["app_register_date"] <= RELIABLE_WINDOW_END)
        ]

    cat_lookup = build_category_lookup(tables["product_category"])
    apps = apps.merge(cat_lookup, on="category_id", how="left")
    apps["display_category"] = apps["display_category"].fillna("Unknown")
    apps["top_level_group"] = apps["top_level_group"].fillna("Unknown")

    apps["date"] = apps["app_register_date"].dt.normalize()

    grouped = apps.groupby(["date", "display_category", "top_level_group"]).agg(
        listing_count=("app_id", "count"),
        avg_price_usd=("price_usd", "mean"),
        pct_new_condition=("item_condition", lambda x: (x == "New").mean()),
    ).reset_index()

    # Fill in days with ZERO listings for each category
    all_days = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
    all_categories = grouped[["display_category", "top_level_group"]].drop_duplicates()

    full_index = (
        pd.MultiIndex.from_product([all_days, all_categories["display_category"]], names=["date", "display_category"])
        .to_frame(index=False)
        .merge(all_categories, on="display_category", how="left")
    )
    panel = full_index.merge(grouped, on=["date", "display_category", "top_level_group"], how="left")
    panel["listing_count"] = panel["listing_count"].fillna(0).astype(int)

    # A handful of level-2 categories are duplicate branches written in Georgian
    # script instead of English (data entry artifact in the source marketplace).
    # Translate for readability; NOT merged into the English siblings since we
    # can't confirm 1:1 equivalence, but they're flagged and, given their tiny
    # volume, excluded from per-category modeling via MIN_HISTORY_LISTINGS below.
    GEORGIAN_TRANSLATIONS = {
        "ძრავი და კომპონენტები": "Engine and components (GE)",
        "განათება და ელექტროობა": "Lighting and electricity (GE)",
        "სამუხრუჭე სისტემა": "Brake system (GE)",
        "სავალი ნაწილები და კომპონენტები": "Chassis parts and components (GE)",
        "მართვის მექანიზმი": "Steering mechanism (GE)",
    }
    panel["display_category"] = panel["display_category"].replace(GEORGIAN_TRANSLATIONS)

    panel = panel.sort_values(["display_category", "date"]).reset_index(drop=True)
    panel["avg_price_usd"] = panel.groupby("display_category")["avg_price_usd"].ffill().bfill()
    panel["pct_new_condition"] = panel.groupby("display_category")["pct_new_condition"].ffill().bfill()

    return panel


if __name__ == "__main__":
    panel = build_daily_panel(vehicle_scope="all")
    print(f"Panel shape: {panel.shape}")
    print(f"Categories: {panel['display_category'].nunique()}")
    print(f"Day range: {panel['date'].min().date()} to {panel['date'].max().date()} "
          f"({panel['date'].nunique()} days)")
    print(panel.head(10))

    out_path = os.path.join(PROCESSED_DIR, "daily_category_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
