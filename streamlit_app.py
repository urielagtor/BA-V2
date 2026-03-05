# app.py
# CoreWeave-branded Streamlit dashboard
# Includes:
# 1) Predictive: time-series style forecasting (Ridge regression with time + quarter features)
# 2) Prescriptive: scenario simulator + if-then recommendations
# 3) Insights/Decision implications: interpretable drivers, uncertainty, limitations
#
# Run:
#   pip install streamlit pandas openpyxl scikit-learn plotly numpy
#   streamlit run app.py
#
# If you run this inside an environment where the provided Excel exists, it will load by default.
# Otherwise, upload the file in the sidebar.

import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="CoreWeave | Debt-to-Income Strategy Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ----------------------------
# Branding (CoreWeave-ish)
# NOTE: adjust to official brand guide if you have one.
# ----------------------------
CW_ACCENT = "#2F5BEA"   # blue accent
CW_BG = "#070A12"
CW_PANEL = "#0E1424"
CW_TEXT = "#E9ECF5"
CW_MUTED = "#A7B0C3"
CW_WARN = "#FFB020"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {CW_BG} 0%, #050712 100%);
        color: {CW_TEXT};
      }}
      h1,h2,h3,h4 {{
        color: {CW_TEXT};
      }}
      .cw-badge {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(47, 91, 234, 0.14);
        border: 1px solid rgba(47, 91, 234, 0.35);
        color: {CW_TEXT};
        font-size: 12px;
        letter-spacing: 0.2px;
      }}
      .cw-card {{
        background: {CW_PANEL};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
      }}
      .cw-muted {{
        color: {CW_MUTED};
      }}
      .cw-warn {{
        color: {CW_WARN};
      }}
      div[data-testid="metric-container"] {{
        background: {CW_PANEL};
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 14px;
      }}
      .stTabs [data-baseweb="tab-list"] button {{
        background: transparent !important;
      }}
      .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom: 3px solid {CW_ACCENT} !important;
      }}
      a {{
        color: {CW_ACCENT} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Defaults / constants
# ----------------------------
DEFAULT_XLSX_PATH = "/mnt/data/Copy of CoreWeave_BalanceSheet_SEC_Filings (with simulated data).xlsx"

METRICS = [
    "Revenue_USD",
    "Cost_of_Revenue_USD",
    "Technology_Infra_USD",
    "Sales_Marketing_USD",
    "General_Admin_USD",
    "Total_Operating_Expenses_USD",
    "Operating_Income_USD",
    "Total_Assets_USD",
    "Total_Liabilities_USD",
]

FEATURE_COLS = [
    "t",
    "Quarter",
    # optional drivers if present (kept numeric and explainable)
    "Cost_of_Revenue_USD",
    "Technology_Infra_USD",
    "Sales_Marketing_USD",
    "General_Admin_USD",
    "Total_Operating_Expenses_USD",
    "Operating_Income_USD",
]


# ----------------------------
# Helpers
# ----------------------------
def money(x) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"${x:,.0f}"
    except Exception:
        return "—"


def pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:,.1f}%"
    except Exception:
        return "—"


def parse_period_date(period_str: str):
    """Extract the date inside parentheses: 'Q1 2022 (Mar 31, 2022)' -> 2022-03-31."""
    if not isinstance(period_str, str):
        return pd.NaT
    m = re.search(r"\(([^)]+)\)", period_str)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), errors="coerce")


def period_type(period_str: str):
    """Return 'Q1'..'Q4' or 'FY' if detected."""
    if not isinstance(period_str, str):
        return None
    m = re.match(r"^(Q[1-4]|FY)", period_str.strip())
    return m.group(1) if m else None


def safe_div(a, b):
    if b is None:
        return np.nan
    try:
        if float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    # fallback
    return pd.read_excel(DEFAULT_XLSX_PATH)


def prep_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "Period" not in df.columns:
        raise ValueError("Expected a 'Period' column in the Excel file.")

    df["Date"] = df["Period"].apply(parse_period_date)
    df["Period_Type"] = df["Period"].apply(period_type)

    # Coerce numeric fields
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Date"].notna()].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Quarterly-only for forecasting (FY duplicates dates; keep FY for overview if you want)
    df_q = df[df["Period_Type"].str.startswith("Q", na=False)].copy()
    df_q = df_q.sort_values("Date").reset_index(drop=True)

    df_q["Quarter"] = df_q["Date"].dt.quarter
    df_q["t"] = np.arange(len(df_q), dtype=int)

    # Core KPI: Debt-to-Income proxy = Total Liabilities / Revenue
    df_q["Debt_to_Income"] = df_q.apply(
        lambda r: safe_div(r.get("Total_Liabilities_USD", np.nan), r.get("Revenue_USD", np.nan)), axis=1
    )

    # Operating margin (optional insight)
    df_q["Op_Margin"] = df_q.apply(
        lambda r: safe_div(r.get("Operating_Income_USD", np.nan), r.get("Revenue_USD", np.nan)), axis=1
    )

    return df, df_q


def build_model():
    """
    Ridge regression with:
      - numeric: t + selected spend/expense drivers
      - categorical: Quarter (seasonality)
    """
    numeric_features = [c for c in FEATURE_COLS if c != "Quarter"]
    categorical_features = ["Quarter"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=1.0, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe, numeric_features, categorical_features


def time_series_backtest(df_q: pd.DataFrame, target_col: str, min_train: int = 6):
    """
    Simple expanding-window backtest:
      train on [0:i), test on i for i >= min_train
    Returns predictions aligned to df_q rows and metrics.
    """
    pipe, _, _ = build_model()

    use_cols = [c for c in FEATURE_COLS if c in df_q.columns] + ["Quarter"]
    use_cols = list(dict.fromkeys(use_cols))  # de-dupe

    # drop rows missing target
    dfx = df_q.dropna(subset=[target_col]).copy()
    if len(dfx) < (min_train + 2):
        return None

    preds = []
    actuals = []
    dates = []

    for i in range(min_train, len(dfx)):
        train = dfx.iloc[:i].copy()
        test = dfx.iloc[i : i + 1].copy()

        # Drop rows missing key features
        train = train.dropna(subset=use_cols + [target_col])
        test = test.dropna(subset=use_cols + [target_col])

        if len(train) < min_train or len(test) != 1:
            continue

        X_train = train[use_cols]
        y_train = train[target_col]
        X_test = test[use_cols]
        y_test = float(test[target_col].iloc[0])

        pipe.fit(X_train, y_train)
        y_pred = float(pipe.predict(X_test)[0])

        preds.append(y_pred)
        actuals.append(y_test)
        dates.append(test["Date"].iloc[0])

    if len(preds) < 2:
        return None

    mae = mean_absolute_error(actuals, preds)
    rmse = math.sqrt(mean_squared_error(actuals, preds))
    mape_val = mape(actuals, preds)

    out = pd.DataFrame({"Date": dates, "Actual": actuals, "Predicted": preds}).sort_values("Date")
    return {"series": out, "mae": mae, "rmse": rmse, "mape": mape_val, "pipe": pipe, "use_cols": use_cols}


def fit_and_forecast_next(df_q: pd.DataFrame, target_col: str):
    """
    Fit on all available quarterly data (with target present) and forecast next quarter.
    """
    pipe, _, _ = build_model()
    use_cols = [c for c in FEATURE_COLS if c in df_q.columns] + ["Quarter"]
    use_cols = list(dict.fromkeys(use_cols))

    dfx = df_q.dropna(subset=[target_col]).copy()
    dfx = dfx.dropna(subset=use_cols).copy()
    if len(dfx) < 6:
        return None

    X = dfx[use_cols]
    y = dfx[target_col]
    pipe.fit(X, y)

    # Create next-quarter feature row
    last = df_q.sort_values("Date").iloc[-1].copy()
    next_date = (last["Date"] + pd.offsets.QuarterEnd(1))
    next_quarter = int(((int(last["Quarter"]) % 4) + 1))

    next_row = last.copy()
    next_row["Date"] = next_date
    next_row["t"] = int(last["t"]) + 1
    next_row["Quarter"] = next_quarter

    # Keep spend drivers as "latest known" (conceptually: carry forward).
    # The prescriptive tab lets users alter these assumptions interactively.
    X_next = pd.DataFrame([next_row])[use_cols]
    y_next = float(pipe.predict(X_next)[0])

    return {"next_date": next_date, "next_pred": y_next, "pipe": pipe, "use_cols": use_cols, "next_row": next_row}


def rule_recommendation(current_ratio, projected_ratio, levers):
    """
    Simple if-then recommendations; keep transparent and rubric-friendly.
    """
    rev_growth, opex_cut, liab_paydown = levers

    msg = []
    if projected_ratio <= 0:
        return ["Check inputs: projected ratio is non-positive."]

    # Severity tiers (tune if your instructor expects different)
    if projected_ratio >= 2.0:
        msg.append("High risk: debt-to-income remains very elevated. Prioritize liability reduction and improve revenue quality (higher-margin contracts).")
    elif projected_ratio >= 1.2:
        msg.append("Moderate risk: ratio is still above a typical comfort zone. Combine revenue growth with disciplined spend controls.")
    else:
        msg.append("Improving: projected ratio is trending healthier. Maintain controls and avoid over-leveraging future expansion.")

    if opex_cut <= 0 and projected_ratio > current_ratio:
        msg.append("Costs are not being reduced and the ratio worsens—consider an operating expense reduction plan (especially non-core spend).")

    if rev_growth < 5 and projected_ratio > 1.2:
        msg.append("Revenue growth assumption is modest; consider initiatives that increase utilization/throughput and longer-term enterprise commitments.")

    if liab_paydown < 2 and projected_ratio > 1.2:
        msg.append("Low liability paydown; explore refinancing, improved terms, or targeted paydowns using operating cash flow / asset-backed facilities.")

    # Tie to levers
    msg.append(f"Primary levers used: Revenue growth {rev_growth:.1f}%, OpEx change {opex_cut:.1f}%, Liabilities paydown {liab_paydown:.1f}%.")
    return msg


# ----------------------------
# Header
# ----------------------------
st.markdown("<span class='cw-badge'>Debt-to-Income Improvement</span>", unsafe_allow_html=True)
st.title("CoreWeave | Debt-to-Income Strategy Dashboard")
st.caption("Objective: provide data-driven advice on improving Debt-to-Income ratio (Liabilities ÷ Revenue).")

# ----------------------------
# Sidebar (inputs)
# ----------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    st.caption("If no file is uploaded, the app will try to load the default provided workbook path.")

    st.divider()
    st.header("Dashboard Controls")

    ratio_threshold = st.slider(
        "Debt-to-Income alert threshold",
        min_value=0.5, max_value=3.0, value=1.2, step=0.1
    )

    backtest_min_train = st.slider(
        "Backtest minimum training quarters",
        min_value=4, max_value=12, value=6, step=1
    )

# ----------------------------
# Load + prep
# ----------------------------
try:
    df_raw = load_data(uploaded)
    df_all, df_q = prep_df(df_raw)
except Exception as e:
    st.error(f"Could not load/parse the data: {e}")
    st.stop()

if df_q.empty:
    st.warning("No quarterly rows found (expected Period values starting with Q1..Q4).")
    st.stop()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Predictive (Forecasting)",
    "Prescriptive (Scenarios)",
    "Insights & Decision Implications"
])

# ----------------------------
# Overview tab
# ----------------------------
with tab1:
    latest = df_q.sort_values("Date").iloc[-1]
    prev = df_q.sort_values("Date").iloc[-2] if len(df_q) >= 2 else None

    current_rev = float(latest.get("Revenue_USD", np.nan))
    current_liab = float(latest.get("Total_Liabilities_USD", np.nan))
    current_ratio = float(latest.get("Debt_to_Income", np.nan))
    current_opex = float(latest.get("Total_Operating_Expenses_USD", np.nan))
    current_opinc = float(latest.get("Operating_Income_USD", np.nan))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Quarterly Revenue", money(current_rev))
    with c2:
        st.metric("Latest Total Liabilities", money(current_liab))
    with c3:
        st.metric("Debt-to-Income (Liabilities ÷ Revenue)", f"{current_ratio:,.2f}" if not np.isnan(current_ratio) else "—")
    with c4:
        st.metric("Operating Income", money(current_opinc))

    st.markdown("")

    left, right = st.columns([2, 1])

    with left:
        # Trend chart: Revenue + Liabilities + Ratio (secondary)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_q["Date"], y=df_q["Revenue_USD"],
            mode="lines+markers", name="Revenue"
        ))
        fig.add_trace(go.Scatter(
            x=df_q["Date"], y=df_q["Total_Liabilities_USD"],
            mode="lines+markers", name="Total Liabilities"
        ))
        fig.update_layout(
            title="Revenue vs Total Liabilities (Quarterly)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            df_q, x="Date", y="Debt_to_Income",
            title="Debt-to-Income Trend (Liabilities ÷ Revenue)",
        )
        fig2.update_layout(template="plotly_dark", height=360)
        fig2.add_hline(y=ratio_threshold, line_dash="dash", annotation_text="Alert threshold")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.subheader("KPI Definition")
        st.markdown(
            """
            **Debt-to-Income (proxy)** = **Total Liabilities ÷ Revenue**  
            - Higher is worse (more liabilities per $1 of revenue)  
            - Goal: improve by **increasing revenue**, **reducing liabilities**, and/or **improving operating efficiency**
            """,
        )
        if not np.isnan(current_ratio):
            if current_ratio >= ratio_threshold:
                st.markdown(f"<span class='cw-warn'>Alert:</span> Current ratio **{current_ratio:.2f}** ≥ threshold **{ratio_threshold:.2f}**.", unsafe_allow_html=True)
            else:
                st.markdown(f"Current ratio **{current_ratio:.2f}** is below threshold **{ratio_threshold:.2f}**.")

        st.markdown("---")
        st.subheader("Data Preview (Quarterly)")
        st.dataframe(
            df_q[["Period", "Date", "Revenue_USD", "Total_Liabilities_USD", "Debt_to_Income",
                  "Technology_Infra_USD", "Total_Operating_Expenses_USD", "Operating_Income_USD"]]
            .sort_values("Date", ascending=False),
            use_container_width=True,
            height=300
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Download cleaned quarterly data
    csv = df_q.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned quarterly dataset (CSV)",
        data=csv,
        file_name="coreweave_quarterly_clean.csv",
        mime="text/csv"
    )

# ----------------------------
# Predictive tab
# ----------------------------
with tab2:
    st.subheader("Predictive Approach: Time-series style forecasting with Ridge Regression")

    st.markdown(
        """
        **Why this model?**
        - The dataset is quarterly and relatively small, so a simple, explainable model is appropriate.
        - We include a time index (**t**) and **Quarter** dummies to capture trend + seasonality.
        - **Ridge regression** helps stabilize coefficients and reduce overfitting when features correlate (common in financial statements).

        **What is being predicted?**
        - Next-quarter **Total Liabilities**
        - Next-quarter **Revenue**
        - Derived: next-quarter **Debt-to-Income** (Liabilities ÷ Revenue)
        """
    )

    # Backtest + forecast for liabilities
    bt_liab = time_series_backtest(df_q, "Total_Liabilities_USD", min_train=backtest_min_train)
    bt_rev = time_series_backtest(df_q, "Revenue_USD", min_train=backtest_min_train)

    if bt_liab is None or bt_rev is None:
        st.warning("Not enough clean quarterly data to run the backtest/forecast (need more quarters with non-missing fields).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Liabilities MAE", money(bt_liab["mae"]))
            st.metric("Liabilities RMSE", money(bt_liab["rmse"]))
        with c2:
            st.metric("Liabilities MAPE", pct(bt_liab["mape"]))
            st.metric("Revenue MAE", money(bt_rev["mae"]))
        with c3:
            st.metric("Revenue RMSE", money(bt_rev["rmse"]))
            st.metric("Revenue MAPE", pct(bt_rev["mape"]))

        st.caption(
            "Evaluation uses an expanding-window backtest (walk-forward). "
            "MAE/RMSE are in USD; MAPE is percent error. Lower is better."
        )

        # Plot Actual vs Predicted (backtest)
        colA, colB = st.columns(2)

        with colA:
            fig = go.Figure()
            s = bt_liab["series"]
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Total Liabilities (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig = go.Figure()
            s = bt_rev["series"]
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Revenue (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Fit on all and forecast next quarter
        fc_liab = fit_and_forecast_next(df_q, "Total_Liabilities_USD")
        fc_rev = fit_and_forecast_next(df_q, "Revenue_USD")

        if fc_liab and fc_rev:
            next_ratio = safe_div(fc_liab["next_pred"], fc_rev["next_pred"])
            st.markdown("### Next-Quarter Forecast (Baseline)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Forecast Revenue", money(fc_rev["next_pred"]))
            with c2:
                st.metric("Forecast Total Liabilities", money(fc_liab["next_pred"]))
            with c3:
                st.metric("Forecast Debt-to-Income", f"{next_ratio:,.2f}" if not np.isnan(next_ratio) else "—")

            st.caption(
                "Baseline forecast holds expense drivers roughly constant at the latest observed quarter "
                "(trend + seasonality still apply). Use the Prescriptive tab to change assumptions."
            )

# ----------------------------
# Prescriptive tab
# ----------------------------
with tab3:
    st.subheader("Prescriptive Approach: Scenario Simulator + Decision Rules")
    st.markdown(
        """
        This section turns predictions into **actionable levers** to improve the Debt-to-Income ratio.

        **Levers (simple, explainable):**
        - **Revenue growth** (pricing, utilization, long-term contracts)
        - **Operating expense change** (discipline on non-core spend)
        - **Liability paydown/refinancing effect** (term improvements or targeted reduction)

        We recompute a projected next-quarter ratio:
        \n- Projected Revenue = baseline revenue forecast × (1 + growth %)
        \n- Projected Liabilities = baseline liabilities forecast × (1 - paydown %)  *(conceptual refinancing/paydown lever)*
        \n- Projected Debt-to-Income = projected liabilities ÷ projected revenue
        """
    )

    # Need baseline forecast to anchor scenarios
    fc_liab = fit_and_forecast_next(df_q, "Total_Liabilities_USD")
    fc_rev = fit_and_forecast_next(df_q, "Revenue_USD")

    if not (fc_liab and fc_rev):
        st.warning("Not enough data to create baseline forecasts for prescriptive scenarios.")
    else:
        base_rev = float(fc_rev["next_pred"])
        base_liab = float(fc_liab["next_pred"])
        base_ratio = safe_div(base_liab, base_rev)

        st.markdown("#### Scenario Controls")
        s1, s2, s3 = st.columns(3)
        with s1:
            rev_growth = st.slider("Revenue growth (%)", -10.0, 60.0, 10.0, 0.5)
        with s2:
            opex_change = st.slider("OpEx change (%) (conceptual)", -40.0, 20.0, -5.0, 0.5)
        with s3:
            liab_paydown = st.slider("Liabilities paydown / refinancing effect (%)", 0.0, 30.0, 5.0, 0.5)

        proj_rev = base_rev * (1.0 + rev_growth / 100.0)

        # Optional: tie OpEx to liabilities in a LIGHT conceptual way (transparent):
        # If OpEx is cut, assume slightly reduced liability growth pressure.
        # This is not "ML", just a business rule.
        opex_factor = 1.0 - (max(0.0, -opex_change) / 100.0) * 0.10  # 10% of OpEx cut translates to less liability pressure
        proj_liab = base_liab * (1.0 - liab_paydown / 100.0) * opex_factor

        proj_ratio = safe_div(proj_liab, proj_rev)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Baseline Forecast Revenue", money(base_rev))
        with k2:
            st.metric("Baseline Forecast Liabilities", money(base_liab))
        with k3:
            st.metric("Baseline Debt-to-Income", f"{base_ratio:,.2f}" if not np.isnan(base_ratio) else "—")
        with k4:
            delta = (proj_ratio - base_ratio) if (not np.isnan(proj_ratio) and not np.isnan(base_ratio)) else np.nan
            st.metric("Scenario Debt-to-Income", f"{proj_ratio:,.2f}" if not np.isnan(proj_ratio) else "—",
                      f"{delta:+.2f}" if not np.isnan(delta) else None)

        # Gauge visualization
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(proj_ratio) if not np.isnan(proj_ratio) else 0.0,
            title={"text": "Projected Debt-to-Income"},
            gauge={
                "axis": {"range": [0, 3.0]},
                "threshold": {"line": {"color": CW_WARN, "width": 4}, "value": ratio_threshold},
                "steps": [
                    {"range": [0, 1.0], "color": "rgba(47, 91, 234, 0.25)"},
                    {"range": [1.0, ratio_threshold], "color": "rgba(255, 176, 32, 0.15)"},
                    {"range": [ratio_threshold, 3.0], "color": "rgba(255, 0, 0, 0.12)"},
                ],
            },
        ))
        gauge.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("#### Recommendation Engine (If–Then Rules)")
        current_ratio = float(df_q.sort_values("Date").iloc[-1]["Debt_to_Income"])
        recs = rule_recommendation(current_ratio, proj_ratio, (rev_growth, opex_change, liab_paydown))
        for r in recs:
            st.write(f"• {r}")

        st.markdown("---")
        st.markdown("#### Scenario Comparison (Quick Grid)")
        # A small comparison table across a few canned scenarios
        scenarios = [
            ("Baseline", 0.0, 0.0, 0.0),
            ("Revenue push", 20.0, 0.0, 0.0),
            ("OpEx discipline", 10.0, -10.0, 0.0),
            ("Refinance/paydown", 10.0, 0.0, 10.0),
            ("Balanced", 15.0, -10.0, 10.0),
        ]

        rows = []
        for name, rg, oc, lp in scenarios:
            pr = base_rev * (1 + rg / 100.0)
            of = 1.0 - (max(0.0, -oc) / 100.0) * 0.10
            pl = base_liab * (1 - lp / 100.0) * of
            rr = safe_div(pl, pr)
            rows.append({
                "Scenario": name,
                "Revenue growth %": rg,
                "OpEx change %": oc,
                "Liab paydown %": lp,
                "Projected Revenue": pr,
                "Projected Liabilities": pl,
                "Projected DTI": rr
            })

        scen_df = pd.DataFrame(rows)
        st.dataframe(
            scen_df.style.format({
                "Projected Revenue": "${:,.0f}",
                "Projected Liabilities": "${:,.0f}",
                "Projected DTI": "{:,.2f}"
            }),
            use_container_width=True
        )

# ----------------------------
# Insights tab
# ----------------------------
with tab4:
    st.subheader("Insights & Decision Implications")
    st.markdown(
        """
        This section is written for stakeholder decision-making: **what it means**, **what to do**, and **what could go wrong**.
        """
    )

    latest = df_q.sort_values("Date").iloc[-1]
    current_ratio = float(latest.get("Debt_to_Income", np.nan))

    # Simple driver view: correlations with liabilities & DTI (directional, not causal)
    driver_cols = [c for c in [
        "Technology_Infra_USD",
        "Total_Operating_Expenses_USD",
        "Operating_Income_USD",
        "Revenue_USD",
        "Cost_of_Revenue_USD",
        "Sales_Marketing_USD",
        "General_Admin_USD"
    ] if c in df_q.columns]

    corr_block = df_q[driver_cols + ["Total_Liabilities_USD", "Debt_to_Income"]].corr(numeric_only=True)
    corr_liab = corr_block["Total_Liabilities_USD"].dropna().sort_values(ascending=False)
    corr_dti = corr_block["Debt_to_Income"].dropna().sort_values(ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### Key Finding: Current Debt-to-Income")
        if not np.isnan(current_ratio):
            st.write(f"Latest Debt-to-Income (Liabilities ÷ Revenue): **{current_ratio:.2f}**")
            if current_ratio >= ratio_threshold:
                st.write(f"Status: **Above** the alert threshold of **{ratio_threshold:.2f}** → focus on reducing liabilities and/or accelerating revenue.")
            else:
                st.write(f"Status: **Below** the threshold of **{ratio_threshold:.2f}**, but still monitor growth of liabilities relative to revenue.")
        else:
            st.write("Debt-to-Income is not available for the latest quarter due to missing liabilities or revenue data.")

        st.markdown("---")
        st.markdown("### Directional Drivers (Correlation)")
        st.caption("Correlation is not causation; use as directional signal.")
        st.write("**Strongest correlations with Total Liabilities:**")
        for k, v in corr_liab.head(5).items():
            st.write(f"• {k}: {v:,.2f}")

        st.write("**Strongest correlations with Debt-to-Income:**")
        for k, v in corr_dti.head(5).items():
            st.write(f"• {k}: {v:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # Scatter: Revenue vs Liabilities with DTI color
        plot_df = df_q.dropna(subset=["Revenue_USD", "Total_Liabilities_USD", "Debt_to_Income"]).copy()
        if len(plot_df) >= 3:
            fig = px.scatter(
                plot_df,
                x="Revenue_USD",
                y="Total_Liabilities_USD",
                color="Debt_to_Income",
                hover_data=["Period", "Date"],
                title="Revenue vs Liabilities (colored by Debt-to-Income)"
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough non-missing rows to plot Revenue vs Liabilities scatter.")

    st.markdown("### Recommended Actions to Improve Debt-to-Income")
    st.markdown(
        """
        **Improve the ratio by increasing the denominator (Revenue), decreasing the numerator (Liabilities), and improving efficiency:**

        **1) Revenue quality & utilization**
        - Increase GPU utilization and throughput (more revenue per unit of infra).
        - Prioritize longer-term enterprise commitments to smooth revenue and reduce volatility.

        **2) Operating discipline**
        - Slow growth in non-core OpEx and align spend with near-term revenue realization.
        - Use KPI guardrails: *OpEx growth should not exceed revenue growth for extended periods.*

        **3) Liability strategy**
        - Explore refinancing, term extensions, or targeted paydowns to lower liabilities relative to revenue.
        - Consider structuring financing to match asset life cycles (reduces mismatch risk).

        **4) Governance**
        - Track the ratio quarterly with thresholds and triggers:
          - If **DTI > threshold**, require an “explain & mitigate” plan.
          - If **DTI improves 2+ quarters**, cautiously re-accelerate growth.
        """
    )

    st.markdown("### Uncertainty & Limitations")
    st.markdown(
        """
        - The workbook is **simulated SEC-style** financial data; real-world behavior may differ.
        - The predictive model assumes a stable relationship between time/spend and outcomes; structural breaks (market shifts, financing changes) can reduce accuracy.
        - Correlations are directional signals, not causal proof.
        - Scenario simulator uses transparent business rules; treat outputs as **decision support**, not guaranteed outcomes.
        """
    )

    st.markdown("---")
    st.markdown("### Quick Narrative (for your write-up)")
    st.write(
        "CoreWeave can improve its debt-to-income ratio primarily by (1) increasing revenue faster than liabilities through utilization and contract strategy, "
        "(2) applying operating discipline so expense growth is aligned with realized revenue, and (3) using refinancing/paydown strategies to reduce the liabilities burden. "
        "The dashboard quantifies the historical ratio, forecasts a baseline next-quarter outcome, and tests scenarios to identify combinations of levers that most improve the ratio."
    )