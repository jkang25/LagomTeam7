import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Lagom Market Opportunity Dashboard",
    layout="wide"
)

DATA_DIR = Path("dashboard_data")

# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data
def load_csv(file_name):
    path = DATA_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in df.columns:
        if "date" in c.lower() or c.lower() in ["month"]:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df

def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"

def fmt_num(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "N/A"

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def get_state_options(df, state_col):
    # Some rows contain multiple states such as "MO, KS".
    # Split them so the State filter shows each abbreviation by itself.
    if df.empty or state_col not in df.columns:
        return ["All"]

    states = set()
    for value in df[state_col].dropna().astype(str):
        for part in value.split(","):
            state = part.strip().upper()
            if state:
                states.add(state)

    return ["All"] + sorted(states)

def state_matches(value, selected_state):
    if selected_state == "All":
        return True
    if pd.isna(value):
        return False
    parts = [p.strip().upper() for p in str(value).split(",")]
    return selected_state.upper() in parts

def get_cbsa_options(df):
    if df.empty or "cbsa_title" not in df.columns:
        return ["All"]
    return ["All"] + sorted(df["cbsa_title"].dropna().astype(str).unique())

def prepare_cluster_table(df):
    if df.empty:
        return df

    out = df.copy()
    if "cluster_id" in out.columns:
        out = out.drop(columns=["cluster_id"])

    if "cluster_name" in out.columns:
        out = out[["cluster_name"] + [c for c in out.columns if c != "cluster_name"]]

    return out

def add_rank_if_missing(df, rank_col):
    if df.empty:
        return df
    if rank_col not in df.columns:
        df = df.copy()
        df[rank_col] = np.arange(1, len(df) + 1)
    return df

def safe_filter(df, col, selected):
    if df.empty or col not in df.columns or selected in (None, [], "All"):
        return df
    if isinstance(selected, list):
        if "All" in selected:
            return df
        return df[df[col].isin(selected)]
    return df[df[col] == selected]

def bar_top(df, x_col, y_col, title, n=15, color_col=None, show_legend=True):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.info(f"Not enough data to show: {title}")
        return

    # Sort descending and show the largest values at the top.
    plot_df = df.copy().sort_values(y_col, ascending=False).head(n)

    fig = px.bar(
        plot_df,
        x=y_col,
        y=x_col,
        color=color_col if color_col in plot_df.columns else None,
        orientation="h",
        title=title,
        hover_data=[c for c in plot_df.columns if c in [
            "states", "price_band", "geo_market_tier", "market_tier",
            "absorption_rate", "months_of_supply", "forecast_avg_absorption_24mo",
            "geographic_priority_score", "market_priority_score"
        ]]
    )

    fig.update_layout(
        height=max(450, n * 28),
        yaxis_title="",
        xaxis_title=y_col.replace("_", " ").title(),
        showlegend=show_legend
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Load data
# -----------------------------

geo = load_csv("geo_scorecard.csv")
market = load_csv("market_scorecard.csv")
monthly = load_csv("cbsa_monthly.csv")
top_current = load_csv("top_current_markets.csv")
velocity = load_csv("top_absorption_velocity.csv")
tier_summary = load_csv("tier_summary.csv")
cluster_profile = load_csv("cluster_profile.csv")
model_compare = load_csv("classification_model_comparison.csv")
xgb_importance = load_csv("xgb_feature_importance.csv")
logit_coefficients = load_csv("logit_coefficients.csv")
tier1_drilldown = load_csv("tier1_drilldown.csv")
top_geo = load_csv("top_geo_markets.csv")

geo = add_rank_if_missing(geo, "geo_rank")
market = add_rank_if_missing(market, "rank")

# -----------------------------
# App Header
# -----------------------------

st.title("Lagom Holdings, LLC Market Opportunity & Absorption Marketing Accelerator Dashboard")
st.caption(
    "Applied Analytics Practicum - Lagom Team 7  - "
    "Christian Duong, Caroline Kim, and Jonathan Kang"
)

if geo.empty and market.empty and monthly.empty:
    st.error(
        "No dashboard data found. Run the dashboard export cell at the end of your notebook, "
        "then place the dashboard_data folder next to this Streamlit app."
    )
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------

st.sidebar.header("Filters")

base_filter_df = geo if not geo.empty else market

state_col = pick_col(base_filter_df, ["states", "state"])
tier_col = pick_col(base_filter_df, ["geo_market_tier", "market_tier"])
cluster_col = pick_col(base_filter_df, ["cluster_name"])
price_col = pick_col(base_filter_df, ["price_band"])

if state_col:
    states = get_state_options(base_filter_df, state_col)
    selected_state = st.sidebar.selectbox("State", states)
else:
    selected_state = "All"

cbsa_options = get_cbsa_options(base_filter_df)
selected_cbsa_filter = st.sidebar.selectbox("Metro / CBSA", cbsa_options)

# Price Band filter removed from sidebar.
selected_price = "All"

if tier_col:
    tiers = ["All"] + list(base_filter_df[tier_col].dropna().astype(str).drop_duplicates())
    selected_tier = st.sidebar.selectbox("Market Tier", tiers)
else:
    selected_tier = "All"

if cluster_col:
    clusters = ["All"] + list(base_filter_df[cluster_col].dropna().astype(str).drop_duplicates())
    selected_cluster = st.sidebar.selectbox("Cluster", clusters)
else:
    selected_cluster = "All"

# Metro / CBSA is handled above as a dropdown filter.

def apply_global_filters(df):
    if df.empty:
        return df

    out = df.copy()

    # State can appear as a single abbreviation or as a comma-separated list.
    if state_col and state_col in out.columns and selected_state != "All":
        out = out[out[state_col].apply(lambda x: state_matches(x, selected_state))]

    for col, val in [
        (price_col, selected_price),
        (tier_col, selected_tier),
        (cluster_col, selected_cluster),
    ]:
        if col and col in out.columns and val != "All":
            out = out[out[col].astype(str) == str(val)]

    if selected_cbsa_filter != "All" and "cbsa_title" in out.columns:
        out = out[out["cbsa_title"].astype(str) == str(selected_cbsa_filter)]

    return out

geo_f = apply_global_filters(geo)
market_f = apply_global_filters(market)
monthly_f = apply_global_filters(monthly)

# -----------------------------
# Tab-style page navigation
# -----------------------------

page_options = [
    "Executive Summary",
    "OA1: Absorption Intelligence",
    "OA2: Geographic Prioritization",
    "Model Validation",
    "Market Drilldown"
]

selected_page = st.radio(
    "Dashboard Section",
    page_options,
    key="selected_page",
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================
# Executive Summary
# ============================================================

if selected_page == "Executive Summary":
    st.header("Executive Summary")

    kpi_df = geo_f if not geo_f.empty else market_f

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Markets Shown",
            f"{len(kpi_df):,}",
            help="Number of markets (Metro / CBSA) currently included after filters are applied."
        )

    with c2:
        if tier_col and tier_col in kpi_df.columns:
            tier1_count = (kpi_df[tier_col].astype(str) == "Tier 1").sum()
            st.metric(
                "Tier 1 Markets",
                f"{tier1_count:,}",
                help="Top 10% markets recommended for first-priority investment."
            )
        else:
            st.metric("Tier 1 Markets", "N/A", help="Tier labels were not found in the filtered data.")

    with c3:
        col = pick_col(kpi_df, ["absorption_rate", "avg_absorption_rate"])
        st.metric(
            "Avg Absorption",
            fmt_pct(kpi_df[col].mean()) if col else "N/A",
            help="Average current absorption rate for the filtered markets. Note: Absorption Rate = (Adjusted Avg Homes Sold / Active Listings) * 100"
        )

    with c4:
        col = pick_col(kpi_df, ["forecast_avg_absorption_24mo"])
        st.metric(
            "Avg Forecast Absorption",
            fmt_pct(kpi_df[col].mean()) if col else "N/A",
            help="Average forecasted absorption from the best performing time-series model."
        )

    with c5:
        col = pick_col(kpi_df, ["months_of_supply", "avg_months_of_supply"])
        st.metric(
            "Avg Months Supply",
            fmt_num(kpi_df[col].mean()) if col else "N/A",
            help="Average estimated months needed to sell current inventory. Note: Months Supply = Active Listings / Adjusted Avg Homes Sold"
        )

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        if not geo_f.empty:
            score_col = pick_col(geo_f, ["geographic_priority_score"])
            if score_col:
                bar_top(
                    geo_f,
                    "cbsa_title",
                    score_col,
                    "Top 15 Geographic Priority Markets",
                    n=15,
                    color_col=tier_col,
                    show_legend=True
                )
            else:
                st.dataframe(geo_f.head(15), use_container_width=True)
        else:
            st.info("Geographic scorecard data not available.")

    with right:
        if not tier_summary.empty:
            st.subheader("Tier Summary")
            st.dataframe(tier_summary, use_container_width=True)
        elif tier_col and tier_col in kpi_df.columns:
            tier_counts = kpi_df[tier_col].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Market Count"]
            fig = px.pie(tier_counts, names="Tier", values="Market Count", title="Market Tier Distribution")
            st.plotly_chart(fig, use_container_width=True)

        if not cluster_profile.empty:
            st.subheader("Cluster Profile")
            st.dataframe(prepare_cluster_table(cluster_profile), use_container_width=True)

# ============================================================
# OA1: Absorption Intelligence
# ============================================================

if selected_page == "OA1: Absorption Intelligence":
    st.header("Opportunity Area #1 — Absorption Rate Intelligence")
    st.write(
        "This page answers where Lagom's \\$200K–\\$300K product is moving fastest today "
        "and where absorption appears to be heading based on historical trends."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Fastest Moving Markets Today")
        src = top_current if not top_current.empty else market_f
        abs_col = pick_col(src, ["absorption_rate", "avg_absorption_rate"])
        if abs_col:
            bar_top(src.sort_values(abs_col, ascending=False), "cbsa_title", abs_col, "Top 15 Current Absorption Markets", n=15)
        else:
            st.info("Absorption ranking data not available.")

    with c2:
        st.subheader("Top Markets by Absorption Velocity")
        src = velocity if not velocity.empty else market_f
        vel_col = pick_col(src, ["absorption_velocity", "absorption_velocity_dimension"])
        if vel_col:
            bar_top(src.sort_values(vel_col, ascending=False), "cbsa_title", vel_col, "Top 15 Absorption Velocity Markets", n=15)
        else:
            st.info("Absorption velocity data not available.")

    st.divider()
    st.subheader("Historical Absorption Trend")

    if not monthly_f.empty and "cbsa_title" in monthly_f.columns:
        cbsa_options = ["All"] + sorted(monthly_f["cbsa_title"].dropna().astype(str).unique())
        default_idx = 0
        selected_cbsa = st.selectbox("Select Metro / CBSA", cbsa_options, index=default_idx)

        if selected_cbsa == "All":
            trend = monthly_f.copy()
        else:
            trend = monthly_f[monthly_f["cbsa_title"].astype(str) == selected_cbsa].copy()

        if price_col and price_col in trend.columns:
            price_options = sorted(trend[price_col].dropna().astype(str).unique())

        date_col = pick_col(trend, ["month_date", "month"])
        if date_col and "absorption_rate" in trend.columns:
            y_cols = ["absorption_rate"]
            for c in ["rolling_12mo_absorption", "rolling_24mo_absorption", "absorption_12mo_avg", "absorption_24mo_avg"]:
                if c in trend.columns:
                    y_cols.append(c)

            if selected_cbsa == "All":
                trend = trend.groupby(date_col, as_index=False)[y_cols].mean()

            trend = trend.sort_values(date_col)
            fig = px.line(
                trend,
                x=date_col,
                y=y_cols,
                title=f"Historical Absorption Trend — {selected_cbsa}"
            )
            fig.update_layout(yaxis_title="Absorption Rate", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(trend.tail(24), use_container_width=True)
        else:
            st.info("Monthly absorption trend columns not found.")
    else:
        st.info("Monthly CBSA data not available.")



# ============================================================
# OA2: Geographic Prioritization
# ============================================================

if selected_page == "OA2: Geographic Prioritization":
    st.header("Opportunity Area #2 — Geographic Market Prioritization")
    st.write(
        "This page answers which metros and counties deserve Lagom's first marketing dollar "
        "using a weighted market scorecard, tier labels, and clustering."
    )

    if not geo_f.empty:
        score_col = pick_col(geo_f, ["geographic_priority_score"])
        if score_col:
            bar_top(
                geo_f.sort_values(score_col, ascending=False),
                "cbsa_title",
                score_col,
                "Ranked Geographic Priority List",
                n=25,
                color_col=tier_col
            )

        st.subheader("Geographic Priority Scorecard")
        display_cols = [
            c for c in [
                "geo_rank", "geo_market_tier", "cluster_name", "cbsa_title", "states", "price_band",
                "geographic_priority_score", "forecast_avg_absorption_24mo",
                "absorption_rate", "months_of_supply", "avg_median_sale_price",
                "absorption_velocity_dimension", "supply_pipeline_pressure_dimension",
                "demand_tailwind_dimension", "ami_affordability_fit_dimension",
                "employment_dimension", "migration_dimension"
            ] if c in geo_f.columns
        ]
        st.dataframe(geo_f[display_cols].sort_values(display_cols[0]) if display_cols else geo_f, use_container_width=True)
    else:
        st.info("Geographic scorecard data not available.")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tier Distribution")
        if tier_col and tier_col in geo_f.columns:
            tier_counts = geo_f[tier_col].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Market Count"]
            tier_counts = tier_counts.sort_values("Market Count", ascending=False)
            fig = px.bar(tier_counts, x="Tier", y="Market Count", title="Tier 1 / Tier 2 / Watchlist Counts")
            st.plotly_chart(fig, use_container_width=True)
        elif not tier_summary.empty:
            st.dataframe(tier_summary, use_container_width=True)

    with c2:
        st.subheader("Market Clusters")
        if not cluster_profile.empty:
            if cluster_col and cluster_col in geo_f.columns:
                score_cluster_col = pick_col(geo_f, ["geographic_priority_score", "market_priority_score"])
                name_cluster_col = cluster_col
                if score_cluster_col and name_cluster_col:
                    cluster_chart = (
                        geo_f.groupby(name_cluster_col, as_index=False)[score_cluster_col]
                        .mean()
                        .rename(columns={score_cluster_col: "avg_priority_score"})
                        .sort_values("avg_priority_score", ascending=False)
                    )
                    fig = px.bar(
                        cluster_chart,
                        x="avg_priority_score",
                        y=name_cluster_col,
                        orientation="h",
                        title="Clusters by Average Priority Score"
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(prepare_cluster_table(cluster_profile), use_container_width=True)
        else:
            st.info("Cluster profile data not available.")

# ============================================================
# Model Validation
# ============================================================

if selected_page == "Model Validation":
    st.header("Model Validation")
    st.write(
        "This page validates the prioritization framework using supervised classifiers "
        "that predict whether a market is Tier 1."
    )

    if not model_compare.empty:
        st.subheader("Classification Model Comparison")
        if "model" in model_compare.columns:
            model_order = ["Logistic Regression", "KNN", "XGBoost Classifier"]
            model_compare["model_order"] = model_compare["model"].map({m: i for i, m in enumerate(model_order)})
            model_compare = model_compare.sort_values("model_order", na_position="last").drop(columns=["model_order"])
        st.dataframe(model_compare, use_container_width=True)

        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in model_compare.columns]
        if "model" in model_compare.columns and metric_cols:
            long = model_compare.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig = px.bar(long, x="model", y="Score", color="Metric", barmode="group", title="Classification Model Performance")
            fig.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Classification model comparison data not available.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Logistic Regression Coefficients")
        if not logit_coefficients.empty and {"feature", "coefficient"}.issubset(logit_coefficients.columns):
            fig = px.bar(
                logit_coefficients.sort_values("coefficient", ascending=False),
                x="coefficient",
                y="feature",
                orientation="h",
                title="Logistic Regression Coefficients"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(logit_coefficients, use_container_width=True)
        else:
            st.info("Logistic Regression coefficient data not available.")

    with c2:
        st.subheader("XGBoost Feature Importance")
        if not xgb_importance.empty and {"feature", "importance"}.issubset(xgb_importance.columns):
            fig = px.bar(
                xgb_importance.sort_values("importance", ascending=False),
                x="importance",
                y="feature",
                orientation="h",
                title="XGBoost Feature Importance"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(xgb_importance, use_container_width=True)
        else:
            st.info("XGBoost feature importance data not available.")

# ============================================================
# Market Drilldown
# ============================================================

if selected_page == "Market Drilldown":
    st.header("Market Drilldown")

    drill_source = geo if not geo.empty else market
    if drill_source.empty or "cbsa_title" not in drill_source.columns:
        st.info("No market drilldown data available.")
    else:
        cbsa_options = sorted(drill_source["cbsa_title"].dropna().astype(str).unique())
        selected = st.selectbox("Select Market", cbsa_options)

        row_df = drill_source[drill_source["cbsa_title"].astype(str) == selected].copy()
        st.subheader(selected)

        if not row_df.empty:
            row = row_df.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Tier", row.get("geo_market_tier", row.get("market_tier", "N/A")))
            with c2:
                score_col = pick_col(row_df, ["geographic_priority_score", "market_priority_score"])
                st.metric("Priority Score", fmt_num(row.get(score_col), 3) if score_col else "N/A")
            with c3:
                st.metric("Absorption Rate", fmt_pct(row.get("absorption_rate", np.nan)))
            with c4:
                st.metric("Forecast Absorption", fmt_pct(row.get("forecast_avg_absorption_24mo", np.nan)))

            st.dataframe(row_df, use_container_width=True)

        if not monthly.empty:
            trend = monthly[monthly["cbsa_title"].astype(str) == selected].copy()
            date_col = pick_col(trend, ["month_date", "month"])
            if date_col and "absorption_rate" in trend.columns:
                st.subheader("Market Absorption History")
                trend = trend.sort_values(date_col)
                fig = px.line(trend, x=date_col, y="absorption_rate", title=f"Absorption Rate Over Time — {selected}")
                st.plotly_chart(fig, use_container_width=True)

        if not tier1_drilldown.empty:
            st.subheader("Tier 1 County Drilldown")
            drill = tier1_drilldown[tier1_drilldown["cbsa_title"].astype(str) == selected] if "cbsa_title" in tier1_drilldown.columns else tier1_drilldown
            st.dataframe(drill, use_container_width=True)

st.caption("Data sources include US Counties, Realtor.com Research, and Redfin Data Center.")