# streamlit_app.py
# CoreWeave executive dashboard with a TOP "website-style" menu (no tabs)
# - Brings back Streamlit menu/header (we do NOT hide it)
# - Removes data upload; always loads from repo path:
#       data/CoreWeave_BalanceSheet_SEC_Filings_simulated.xlsx
# - Uses logo in repo root:
#       CoreWeave Logo White.svg
#
# Run:
#   pip install streamlit pandas openpyxl scikit-learn plotly numpy
#   streamlit run streamlit_app.py

import re
import math
import os
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
    layout="wide",
)

# ----------------------------
# Branding (CoreWeave-ish)
# ----------------------------
CW_ACCENT = "#2F5BEA"   # blue accent
CW_BG = "#070A12"
CW_PANEL = "#0E1424"
CW_TEXT = "#E9ECF5"
CW_MUTED = "#A7B0C3"
CW_WARN = "#FFB020"
CW_DANGER = "#FF4D4D"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {CW_BG} 0%, #050712 100%);
        color: {CW_TEXT};
      }}
      h1,h2,h3,h4 {{ color: {CW_TEXT}; }}
      .cw-badge{{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(47, 91, 234, 0.14);
        border: 1px solid rgba(47, 91, 234, 0.35);
        color: {CW_TEXT};
        font-size: 12px;
        letter-spacing: 0.2px;
      }}
      .cw-card{{
        background: {CW_PANEL};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
      }}
      .cw-muted {{ color: {CW_MUTED}; }}
      .cw-warn {{ color: {CW_WARN}; }}
      .cw-danger {{ color: {CW_DANGER}; }}

      div[data-testid="metric-container"]{{
        background: {CW_PANEL};
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 14px;
      }}

      /* Top menu button row polish */
      .cw-nav {{
        display:flex; gap:10px; align-items:center;
        padding: 6px 0 10px 0;
      }}
      .cw-nav-title {{
        font-weight: 600;
        color: {CW_TEXT};
        margin: 0;
      }}
      .cw-divider {{
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 10px 0 18px 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Paths (repo-relative)
# ----------------------------
DATA_PATH = "data/CoreWeave_BalanceSheet_SEC_Filings_simulated.xlsx"
LOGO_PATH = "CoreWeave Logo White.svg"

# ----------------------------
# Columns / model config
# ----------------------------
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
    if not isinstance(period_str, str):
        return pd.NaT
    m = re.search(r"\(([^)]+)\)", period_str)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), errors="coerce")


def period_type(period_str: str):
    if not isinstance(period_str, str):
        return None
    m = re.match(r"^(Q[1-4]|FY)", period_str.strip())
    return m.group(1) if m else None


def safe_div(a, b):
    try:
        if b is None:
            return np.nan
        b = float(b)
        if b == 0:
            return np.nan
        return float(a) / b
    except Exception:
        return np.nan


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at '{DATA_PATH}'. Make sure it exists in your repo."
        )
    return pd.read_excel(DATA_PATH)


def prep_df(df_raw: pd.DataFrame):
    df = df_raw.copy()
    if "Period" not in df.columns:
        raise ValueError("Expected a 'Period' column in the Excel file.")

    df["Date"] = df["Period"].apply(parse_period_date)
    df["Period_Type"] = df["Period"].apply(period_type)

    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Date"].notna()].copy().sort_values("Date").reset_index(drop=True)

    df_q = df[df["Period_Type"].str.startswith("Q", na=False)].copy()
    df_q = df_q.sort_values("Date").reset_index(drop=True)

    df_q["Quarter"] = df_q["Date"].dt.quarter
    df_q["t"] = np.arange(len(df_q), dtype=int)

    df_q["Debt_to_Income"] = df_q.apply(
        lambda r: safe_div(r.get("Total_Liabilities_USD", np.nan), r.get("Revenue_USD", np.nan)),
        axis=1,
    )

    df_q["Op_Margin"] = df_q.apply(
        lambda r: safe_div(r.get("Operating_Income_USD", np.nan), r.get("Revenue_USD", np.nan)),
        axis=1,
    )
    return df, df_q


def build_model():
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
    return Pipeline([("pre", pre), ("model", model)])


def time_series_backtest(df_q: pd.DataFrame, target_col: str, min_train: int = 6):
    pipe = build_model()
    use_cols = [c for c in FEATURE_COLS if c in df_q.columns] + ["Quarter"]
    use_cols = list(dict.fromkeys(use_cols))

    dfx = df_q.dropna(subset=[target_col]).copy()
    if len(dfx) < (min_train + 2):
        return None

    preds, actuals, dates = [], [], []
    for i in range(min_train, len(dfx)):
        train = dfx.iloc[:i].dropna(subset=use_cols + [target_col]).copy()
        test = dfx.iloc[i:i+1].dropna(subset=use_cols + [target_col]).copy()
        if len(train) < min_train or len(test) != 1:
            continue

        pipe.fit(train[use_cols], train[target_col])
        y_pred = float(pipe.predict(test[use_cols])[0])
        y_true = float(test[target_col].iloc[0])

        preds.append(y_pred)
        actuals.append(y_true)
        dates.append(test["Date"].iloc[0])

    if len(preds) < 2:
        return None

    mae = mean_absolute_error(actuals, preds)
    rmse = math.sqrt(mean_squared_error(actuals, preds))
    mape_val = mape(actuals, preds)
    out = pd.DataFrame({"Date": dates, "Actual": actuals, "Predicted": preds}).sort_values("Date")
    return {"series": out, "mae": mae, "rmse": rmse, "mape": mape_val}


def fit_and_forecast_next(df_q: pd.DataFrame, target_col: str):
    pipe = build_model()
    use_cols = [c for c in FEATURE_COLS if c in df_q.columns] + ["Quarter"]
    use_cols = list(dict.fromkeys(use_cols))

    dfx = df_q.dropna(subset=[target_col]).dropna(subset=use_cols).copy()
    if len(dfx) < 6:
        return None

    pipe.fit(dfx[use_cols], dfx[target_col])

    last = df_q.sort_values("Date").iloc[-1].copy()
    next_date = last["Date"] + pd.offsets.QuarterEnd(1)
    next_quarter = int(((int(last["Quarter"]) % 4) + 1))

    next_row = last.copy()
    next_row["Date"] = next_date
    next_row["t"] = int(last["t"]) + 1
    next_row["Quarter"] = next_quarter

    y_next = float(pipe.predict(pd.DataFrame([next_row])[use_cols])[0])
    return {"next_date": next_date, "next_pred": y_next}


def status_label(dti, threshold):
    if np.isnan(dti):
        return ("Unknown", "Data incomplete")
    if dti >= threshold:
        return ("Needs action", "DTI above threshold")
    if dti >= (threshold * 0.9):
        return ("Watch", "DTI near threshold")
    return ("Healthy", "DTI below threshold")


def rule_recommendation(projected_ratio, levers, threshold):
    rev_growth, opex_change, liab_paydown = levers
    msgs = []

    if projected_ratio is None or np.isnan(projected_ratio) or projected_ratio <= 0:
        return ["Check inputs: projected ratio is invalid."]

    if projected_ratio >= 2.0:
        msgs.append("Prioritize liability reduction/refinancing and revenue quality (higher-margin, longer-term contracts).")
    elif projected_ratio >= threshold:
        msgs.append("Combine revenue growth with disciplined cost controls and a liability strategy to bring DTI below threshold.")
    else:
        msgs.append("Maintain discipline: keep liabilities from outpacing revenue and avoid over-leveraging expansions.")

    if opex_change > 0 and projected_ratio >= threshold:
        msgs.append("OpEx increases while DTI is high—consider freezing discretionary spend until DTI improves.")

    if rev_growth < 5 and projected_ratio >= threshold:
        msgs.append("Revenue growth assumption is modest—focus on utilization, pricing, and enterprise commitments.")

    if liab_paydown < 2 and projected_ratio >= threshold:
        msgs.append("Low liabilities improvement—explore refinancing, term extensions, or targeted paydowns.")

    msgs.append(f"Levers: Revenue growth {rev_growth:.1f}%, OpEx change {opex_change:.1f}%, Liabilities improvement {liab_paydown:.1f}%.")
    return msgs


# ----------------------------
# State init
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "ratio_threshold" not in st.session_state:
    st.session_state.ratio_threshold = 1.2

if "backtest_min_train" not in st.session_state:
    st.session_state.backtest_min_train = 6


# ----------------------------
# Load data
# ----------------------------
try:
    df_raw = load_data()
    df_all, df_q = prep_df(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

if df_q.empty:
    st.warning("No quarterly rows found (expected Period values starting with Q1..Q4).")
    st.stop()


# ----------------------------
# Top header + "website" menu
# ----------------------------
h1, h2 = st.columns([1, 6], vertical_alignment="center")

with h1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
    else:
        st.markdown("<span class='cw-badge'>CoreWeave</span>", unsafe_allow_html=True)

with h2:
    st.markdown("<span class='cw-badge'>Executive Decision Dashboard</span>", unsafe_allow_html=True)
    st.title("Debt-to-Income Strategy Dashboard")
    st.caption("Goal: improve Debt-to-Income (Total Liabilities ÷ Revenue) using forecasts + what-if scenarios.")

# NAV ROW (top menu)
nav = st.columns([1.2, 1.8, 2.0, 2.4, 3.6])
pages = ["Overview", "Forecast", "Scenario Planner", "Recommendations & Risks"]

def nav_button(col, label):
    active = (st.session_state.page == label)
    if col.button(label, type=("primary" if active else "secondary"), use_container_width=True):
        st.session_state.page = label

nav_button(nav[0], "Overview")
nav_button(nav[1], "Forecast")
nav_button(nav[2], "Scenario Planner")
nav_button(nav[3], "Recommendations & Risks")

st.markdown("<div class='cw-divider'></div>", unsafe_allow_html=True)

# Advanced controls (collapsed; keeps UI clean for execs)
with st.expander("Controls (advanced)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.ratio_threshold = st.slider(
            "Debt-to-Income alert threshold",
            0.5, 3.0, float(st.session_state.ratio_threshold), 0.1,
            help="If DTI exceeds this threshold, the dashboard flags it as requiring mitigation."
        )
    with c2:
        st.session_state.backtest_min_train = st.slider(
            "Backtest minimum training quarters",
            4, 12, int(st.session_state.backtest_min_train), 1,
            help="Higher values can stabilize backtests but reduce the number of backtest points."
        )

ratio_threshold = float(st.session_state.ratio_threshold)
backtest_min_train = int(st.session_state.backtest_min_train)

# Compute baseline forecast once (used across pages)
fc_liab = fit_and_forecast_next(df_q, "Total_Liabilities_USD")
fc_rev = fit_and_forecast_next(df_q, "Revenue_USD")
base_rev = float(fc_rev["next_pred"]) if fc_rev else np.nan
base_liab = float(fc_liab["next_pred"]) if fc_liab else np.nan
base_ratio = safe_div(base_liab, base_rev) if (fc_rev and fc_liab) else np.nan
next_label = fc_liab["next_date"].strftime("%b %d, %Y") if fc_liab else "Next quarter"

latest = df_q.sort_values("Date").iloc[-1]
current_ratio = float(latest.get("Debt_to_Income", np.nan))
status, status_sub = status_label(current_ratio, ratio_threshold)

# Executive Summary (shown on every page)
st.markdown("## Executive Summary")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Current DTI", f"{current_ratio:,.2f}" if not np.isnan(current_ratio) else "—")
k2.metric(f"Baseline Forecast DTI ({next_label})", f"{base_ratio:,.2f}" if not np.isnan(base_ratio) else "—")
k3.metric("Status", status)
k4.metric("Alert Threshold", f"{ratio_threshold:.2f}")

st.caption("DTI = Total Liabilities ÷ Revenue. Lower is better. Use Scenario Planner for “If X, then Y” decision testing.")
st.markdown("<div class='cw-divider'></div>", unsafe_allow_html=True)


# ----------------------------
# Pages (no tabs)
# ----------------------------
def render_overview():
    st.subheader("Overview")

    current_rev = float(latest.get("Revenue_USD", np.nan))
    current_liab = float(latest.get("Total_Liabilities_USD", np.nan))
    current_opinc = float(latest.get("Operating_Income_USD", np.nan))

    a, b, c, d = st.columns(4)
    a.metric("Latest Quarterly Revenue", money(current_rev))
    b.metric("Latest Total Liabilities", money(current_liab))
    c.metric("Debt-to-Income", f"{current_ratio:,.2f}" if not np.isnan(current_ratio) else "—")
    d.metric("Operating Income", money(current_opinc))

    left, right = st.columns([2, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_q["Date"], y=df_q["Revenue_USD"], mode="lines+markers", name="Revenue"))
        fig.add_trace(go.Scatter(x=df_q["Date"], y=df_q["Total_Liabilities_USD"], mode="lines+markers", name="Total Liabilities"))
        fig.update_layout(title="Revenue vs Total Liabilities (Quarterly)", template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df_q, x="Date", y="Debt_to_Income", title="Debt-to-Income Trend (Liabilities ÷ Revenue)")
        fig2.update_layout(template="plotly_dark", height=360)
        fig2.add_hline(y=ratio_threshold, line_dash="dash", annotation_text="Alert threshold")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### What executives should watch")
        if not np.isnan(current_ratio):
            if current_ratio >= ratio_threshold:
                st.markdown(f"<span class='cw-danger'>Needs action:</span> DTI {current_ratio:.2f} is above threshold {ratio_threshold:.2f}.", unsafe_allow_html=True)
            elif current_ratio >= ratio_threshold * 0.9:
                st.markdown(f"<span class='cw-warn'>Watch:</span> DTI {current_ratio:.2f} is near threshold {ratio_threshold:.2f}.", unsafe_allow_html=True)
            else:
                st.markdown(f"DTI {current_ratio:.2f} is below threshold {ratio_threshold:.2f}.")
        st.markdown("---")
        st.markdown("### Data (details)")
        with st.expander("View underlying quarterly data", expanded=False):
            preview_cols = [c for c in [
                "Period", "Date", "Revenue_USD", "Total_Liabilities_USD", "Debt_to_Income",
                "Technology_Infra_USD", "Total_Operating_Expenses_USD", "Operating_Income_USD"
            ] if c in df_q.columns]
            st.dataframe(df_q[preview_cols].sort_values("Date", ascending=False), use_container_width=True, height=340)
        st.markdown("</div>", unsafe_allow_html=True)

    csv = df_q.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned quarterly dataset (CSV)",
        data=csv,
        file_name="coreweave_quarterly_clean.csv",
        mime="text/csv",
    )


def render_forecast():
    st.subheader("Forecast")
    st.markdown(
        """
        This page forecasts **Revenue** and **Total Liabilities** next quarter using a simple, explainable approach
        (trend + seasonality via quarter indicators; Ridge regression to reduce overfitting).

        **How to use this:** focus on the *direction* and *order-of-magnitude*, then test actions in the Scenario Planner.
        """
    )

    bt_liab = time_series_backtest(df_q, "Total_Liabilities_USD", min_train=backtest_min_train)
    bt_rev = time_series_backtest(df_q, "Revenue_USD", min_train=backtest_min_train)

    if bt_liab and bt_rev:
        m1, m2, m3 = st.columns(3)
        m1.metric("Revenue MAPE (typical error)", pct(bt_rev["mape"]))
        m2.metric("Liabilities MAPE (typical error)", pct(bt_liab["mape"]))
        m3.metric("Backtest window (min train qtrs)", str(backtest_min_train))
        st.caption("MAPE is average percent error from walk-forward backtests. Lower is better.")
    else:
        st.info("Not enough clean quarterly data to compute stable backtest metrics.")

    # Backtest charts (exec-friendly: 2 charts)
    colA, colB = st.columns(2)

    with colA:
        if bt_rev:
            s = bt_rev["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Revenue (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.empty()

    with colB:
        if bt_liab:
            s = bt_liab["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Total Liabilities (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.empty()

    st.markdown("### Next-quarter baseline forecast")
    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast Revenue", money(base_rev))
    f2.metric("Forecast Total Liabilities", money(base_liab))
    f3.metric("Forecast DTI", f"{base_ratio:,.2f}" if not np.isnan(base_ratio) else "—")


def render_scenario_planner():
    st.subheader("Scenario Planner (If X, then Y)")
    st.markdown("Adjust the levers below to see how decisions could change next-quarter DTI.")

    if np.isnan(base_rev) or np.isnan(base_liab) or np.isnan(base_ratio):
        st.warning("Baseline forecasts are unavailable (not enough clean data).")
        return

    # Presets (exec-friendly)
    p1, p2, p3, _ = st.columns([1.2, 1.0, 1.1, 3.7])
    if p1.button("Conservative", use_container_width=True):
        st.session_state.rev_growth = 8.0
        st.session_state.opex_change = -3.0
        st.session_state.liab_paydown = 3.0
    if p2.button("Balanced", use_container_width=True):
        st.session_state.rev_growth = 15.0
        st.session_state.opex_change = -8.0
        st.session_state.liab_paydown = 8.0
    if p3.button("Aggressive", use_container_width=True):
        st.session_state.rev_growth = 25.0
        st.session_state.opex_change = -12.0
        st.session_state.liab_paydown = 12.0

    # Sliders
    s1, s2, s3 = st.columns(3)
    with s1:
        rev_growth = st.slider("Revenue growth (%)", -10.0, 60.0, float(st.session_state.get("rev_growth", 15.0)), 0.5)
    with s2:
        opex_change = st.slider("Operating expense change (%)", -40.0, 20.0, float(st.session_state.get("opex_change", -8.0)), 0.5)
    with s3:
        liab_paydown = st.slider("Liabilities improvement (%) (paydown/refi)", 0.0, 30.0, float(st.session_state.get("liab_paydown", 8.0)), 0.5)

    # Scenario math (transparent)
    proj_rev = base_rev * (1.0 + rev_growth / 100.0)

    # Light business rule: OpEx cuts reduce liability pressure slightly
    opex_factor = 1.0 - (max(0.0, -opex_change) / 100.0) * 0.10
    proj_liab = base_liab * (1.0 - liab_paydown / 100.0) * opex_factor
    proj_ratio = safe_div(proj_liab, proj_rev)

    st.markdown("### Scenario results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Baseline DTI", f"{base_ratio:,.2f}")
    r2.metric(f"Scenario DTI ({next_label})", f"{proj_ratio:,.2f}" if not np.isnan(proj_ratio) else "—")
    delta = proj_ratio - base_ratio if not (np.isnan(proj_ratio) or np.isnan(base_ratio)) else np.nan
    r3.metric("Change vs baseline", f"{delta:+.2f}" if not np.isnan(delta) else "—")
    r4.metric("Scenario Revenue", money(proj_rev), money(proj_rev - base_rev))

    # The executive “If X then Y” sentence
    if not np.isnan(proj_ratio):
        st.markdown(
            f"""
            **If** revenue grows by **{rev_growth:.1f}%**, OpEx changes by **{opex_change:.1f}%**, and liabilities improve by **{liab_paydown:.1f}%**,  
            **then** next-quarter Debt-to-Income is projected to be **{proj_ratio:.2f}** (baseline: **{base_ratio:.2f}**).
            """
        )

    # Status + actions
    scen_status, scen_sub = status_label(proj_ratio, ratio_threshold)
    st.markdown(f"**Status under this scenario:** {scen_status} — {scen_sub}")

    st.markdown("### Recommended actions under this scenario")
    recs = rule_recommendation(proj_ratio, (rev_growth, opex_change, liab_paydown), ratio_threshold)
    for r in recs[:3]:
        st.write(f"• {r}")

    st.markdown("### Liabilities impact (dollars)")
    d1, d2 = st.columns(2)
    d1.metric("Scenario Liabilities", money(proj_liab), money(proj_liab - base_liab))
    d2.metric("Scenario DTI", f"{proj_ratio:,.2f}" if not np.isnan(proj_ratio) else "—", f"{delta:+.2f}" if not np.isnan(delta) else None)

    # Simple gauge for execs
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
                {"range": [ratio_threshold, 3.0], "color": "rgba(255, 0, 0, 0.10)"},
            ],
        },
    ))
    gauge.update_layout(template="plotly_dark", height=280)
    st.plotly_chart(gauge, use_container_width=True)


def render_recommendations():
    st.subheader("Recommendations & Risks")

    left, right = st.columns([1.3, 1.7])

    with left:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### Recommended actions (next 90 days)")
        if np.isnan(current_ratio):
            items = [
                "Validate liabilities and revenue inputs for the latest quarter (DTI missing).",
                "Once validated, use Scenario Planner to identify a realistic path below the threshold.",
            ]
        elif current_ratio >= ratio_threshold:
            items = [
                "Liability strategy: refinance/term extension or targeted paydowns to reduce liabilities relative to revenue.",
                "Revenue quality: prioritize longer-term enterprise commitments + maximize utilization (revenue per infra unit).",
                "Operating guardrails: freeze discretionary OpEx growth until DTI returns below threshold.",
            ]
        elif current_ratio >= ratio_threshold * 0.9:
            items = [
                "Tighten approval for debt-funded expansion while DTI is near threshold.",
                "Focus on utilization, pricing, and contract duration to lift revenue without proportional liability growth.",
                "Set quarterly trigger: if DTI crosses threshold, require a mitigation plan.",
            ]
        else:
            items = [
                "Maintain discipline: keep liabilities from outpacing revenue.",
                "Prioritize expansions with contracted revenue or clear utilization upside.",
                "Monitor DTI quarterly using the current threshold trigger.",
            ]
        for i in items:
            st.write(f"• {i}")

        st.markdown("---")
        st.markdown("### Governance triggers")
        st.write(f"• **Healthy:** DTI < {ratio_threshold*0.9:.2f}")
        st.write(f"• **Watch:** {ratio_threshold*0.9:.2f} ≤ DTI < {ratio_threshold:.2f}")
        st.write(f"• **Needs action:** DTI ≥ {ratio_threshold:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # Directional driver visualization (correlation)
        driver_cols = [c for c in [
            "Technology_Infra_USD",
            "Total_Operating_Expenses_USD",
            "Operating_Income_USD",
            "Revenue_USD",
            "Cost_of_Revenue_USD",
            "Sales_Marketing_USD",
            "General_Admin_USD",
        ] if c in df_q.columns]

        plot_df = df_q.dropna(subset=["Revenue_USD", "Total_Liabilities_USD", "Debt_to_Income"]).copy()
        if len(plot_df) >= 3:
            fig = px.scatter(
                plot_df,
                x="Revenue_USD",
                y="Total_Liabilities_USD",
                color="Debt_to_Income",
                hover_data=["Period", "Date"],
                title="Revenue vs Liabilities (colored by DTI)",
            )
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Uncertainty & limitations")
        st.write("• Data is simulated SEC-style; real-world results can differ.")
        st.write("• Forecasts assume stable relationships; structural breaks (market/financing) reduce accuracy.")
        st.write("• Scenario planner uses transparent business rules; use as decision support, not guarantees.")


# ----------------------------
# Route
# ----------------------------
if st.session_state.page == "Overview":
    render_overview()
elif st.session_state.page == "Forecast":
    render_forecast()
elif st.session_state.page == "Scenario Planner":
    render_scenario_planner()
else:
    render_recommendations()