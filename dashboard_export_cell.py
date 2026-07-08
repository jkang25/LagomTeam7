
# ------------------------------------------------------------
# Streamlit Dashboard Data Export Cell
# ------------------------------------------------------------
# Run this cell at the end of the notebook after all analysis cells.
# It creates a small dashboard_data folder that Streamlit reads from.
# These files are dashboard inputs, not separate analytical outputs.

from pathlib import Path

dashboard_dir = Path("dashboard_data")
dashboard_dir.mkdir(exist_ok=True)

def export_if_exists(df_name, file_name):
    if df_name in globals():
        df = globals()[df_name].copy()
        df.to_csv(dashboard_dir / file_name, index=False)
        print(f"Exported {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print(f"Skipped {file_name}: {df_name} not found")

export_if_exists("geo_scorecard", "geo_scorecard.csv")
export_if_exists("market_scorecard", "market_scorecard.csv")
export_if_exists("cbsa_monthly", "cbsa_monthly.csv")
export_if_exists("top_current_markets", "top_current_markets.csv")
export_if_exists("latest_velocity", "top_absorption_velocity.csv")
export_if_exists("tier_summary", "tier_summary.csv")
export_if_exists("cluster_profile", "cluster_profile.csv")
export_if_exists("classification_model_comparison", "classification_model_comparison.csv")
export_if_exists("xgb_feature_importance", "xgb_feature_importance.csv")
export_if_exists("logit_coefficients", "logit_coefficients.csv")
export_if_exists("tier1_drilldown", "tier1_drilldown.csv")
export_if_exists("top_geo_markets", "top_geo_markets.csv")

print("\nDashboard data export complete.")
print("Point Streamlit to the dashboard_data folder.")
