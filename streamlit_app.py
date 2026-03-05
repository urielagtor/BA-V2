# streamlit_app.py
# CoreWeave executive dashboard with SIDEBAR navigation + 25/75 two-column layouts
# + NEW "Dictionary" page (definitions + how to use)
#
# - Sidebar contains page navigation (traditional app menu)
# - No data upload: loads repo file data/CoreWeave_BalanceSheet_SEC_Filings_simulated.xlsx
# - Logo: CoreWeave Logo White.svg (repo root)
# - Forecast + Scenario Planner use 25% KPI/control column + 75% visuals column
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
CW_ACCENT = "#2F5BEA"
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

      /* Sidebar styling polish */
      [data-testid="stSidebar"] {{
        background: rgba(14, 20, 36, 0.85);
        border-right: 1px solid rgba(255,255,255,0.08);
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

latest = df_q.sort_values("Date").iloc[-1]
current_ratio = float(latest.get("Debt_to_Income", np.nan))

# Baseline forecast shared across pages
fc_liab = fit_and_forecast_next(df_q, "Total_Liabilities_USD")
fc_rev = fit_and_forecast_next(df_q, "Revenue_USD")
base_rev = float(fc_rev["next_pred"]) if fc_rev else np.nan
base_liab = float(fc_liab["next_pred"]) if fc_liab else np.nan
base_ratio = safe_div(base_liab, base_rev) if (fc_rev and fc_liab) else np.nan
next_label = fc_liab["next_date"].strftime("%b %d, %Y") if fc_liab else "Next quarter"

# ----------------------------
# Sidebar Navigation + Controls
# ----------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<span class='cw-badge'>CoreWeave</span>", unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Forecast", "Scenario Planner", "Recommendations & Risks", "Dictionary"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='cw-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Controls")
    ratio_threshold = st.slider(
        "DTI alert threshold",
        0.5, 3.0, 1.2, 0.1,
        help="If DTI exceeds this threshold, the dashboard flags it as requiring mitigation.",
    )
    backtest_min_train = st.slider(
        "Backtest min training quarters",
        4, 12, 6, 1,
        help="Higher values can stabilize backtests but reduce the number of backtest points.",
    )

    st.markdown("<div class='cw-divider'></div>", unsafe_allow_html=True)
    status, status_sub = status_label(current_ratio, ratio_threshold)
    st.markdown("### Current status")
    st.write(f"**DTI:** {current_ratio:,.2f}" if not np.isnan(current_ratio) else "**DTI:** —")
    st.write(f"**Status:** {status}")
    st.caption(status_sub)

# ----------------------------
# Main Header + Executive Summary
# ----------------------------
st.markdown("<span class='cw-badge'>Executive Decision Dashboard</span>", unsafe_allow_html=True)
st.title("Debt-to-Income Strategy Dashboard")
st.caption("Goal: improve Debt-to-Income (Total Liabilities ÷ Revenue) using forecasts + what-if scenarios.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current DTI", f"{current_ratio:,.2f}" if not np.isnan(current_ratio) else "—")
k2.metric(f"Baseline Forecast DTI ({next_label})", f"{base_ratio:,.2f}" if not np.isnan(base_ratio) else "—")
k3.metric("Alert Threshold", f"{ratio_threshold:.2f}")
k4.metric("Status", status_label(current_ratio, ratio_threshold)[0])

st.markdown("<div class='cw-divider'></div>", unsafe_allow_html=True)


# ----------------------------
# Page renderers
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
        st.markdown("### Executive notes")
        if not np.isnan(current_ratio):
            if current_ratio >= ratio_threshold:
                st.markdown(
                    f"<span class='cw-danger'>Needs action:</span> DTI {current_ratio:.2f} is above threshold {ratio_threshold:.2f}.",
                    unsafe_allow_html=True
                )
            elif current_ratio >= ratio_threshold * 0.9:
                st.markdown(
                    f"<span class='cw-warn'>Watch:</span> DTI {current_ratio:.2f} is near threshold {ratio_threshold:.2f}.",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"DTI {current_ratio:.2f} is below threshold {ratio_threshold:.2f}.")
        st.markdown("---")
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

    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### How to read this page")
        st.write("• Forecasts are **directional** and meant for planning.")
        st.write("• Use **Scenario Planner** to test actions (If X, then Y).")
        st.markdown("---")

        st.markdown("### Baseline next quarter")
        st.metric("Forecast Revenue", money(base_rev))
        st.metric("Forecast Liabilities", money(base_liab))
        st.metric("Forecast DTI", f"{base_ratio:,.2f}" if not np.isnan(base_ratio) else "—")
        st.markdown("---")

        bt_liab = time_series_backtest(df_q, "Total_Liabilities_USD", min_train=backtest_min_train)
        bt_rev = time_series_backtest(df_q, "Revenue_USD", min_train=backtest_min_train)

        st.markdown("### Reliability (backtest)")
        st.metric("Revenue MAPE", pct(bt_rev["mape"]) if bt_rev else "—")
        st.metric("Liabilities MAPE", pct(bt_liab["mape"]) if bt_liab else "—")
        st.caption("MAPE = average percent error. Lower is better.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        bt_liab = time_series_backtest(df_q, "Total_Liabilities_USD", min_train=backtest_min_train)
        bt_rev = time_series_backtest(df_q, "Revenue_USD", min_train=backtest_min_train)

        if bt_rev:
            s = bt_rev["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Revenue (Actual vs Predicted)", template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)

        if bt_liab:
            s = bt_liab["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Total Liabilities (Actual vs Predicted)", template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)


def render_scenario_planner():
    st.subheader("Scenario Planner (If X, then Y)")

    if np.isnan(base_rev) or np.isnan(base_liab) or np.isnan(base_ratio):
        st.warning("Baseline forecasts are unavailable (not enough clean data).")
        return

    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### Scenario inputs")

        p1, p2, p3 = st.columns(3)
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

        rev_growth = st.slider("Revenue growth (%)", -10.0, 60.0, float(st.session_state.get("rev_growth", 15.0)), 0.5)
        opex_change = st.slider("OpEx change (%)", -40.0, 20.0, float(st.session_state.get("opex_change", -8.0)), 0.5)
        liab_paydown = st.slider("Liabilities improvement (%)", 0.0, 30.0, float(st.session_state.get("liab_paydown", 8.0)), 0.5)

        proj_rev = base_rev * (1.0 + rev_growth / 100.0)
        opex_factor = 1.0 - (max(0.0, -opex_change) / 100.0) * 0.10
        proj_liab = base_liab * (1.0 - liab_paydown / 100.0) * opex_factor
        proj_ratio = safe_div(proj_liab, proj_rev)

        st.markdown("---")
        st.markdown("### Scenario output")
        st.metric(f"Scenario DTI ({next_label})", f"{proj_ratio:,.2f}" if not np.isnan(proj_ratio) else "—")
        delta = proj_ratio - base_ratio if not (np.isnan(proj_ratio) or np.isnan(base_ratio)) else np.nan
        st.metric("Change vs baseline", f"{delta:+.2f}" if not np.isnan(delta) else "—")

        scen_status, scen_sub = status_label(proj_ratio, ratio_threshold)
        st.write(f"**Status:** {scen_status}")
        st.caption(scen_sub)

        st.markdown("---")
        st.markdown("### Recommendations")
        recs = rule_recommendation(proj_ratio, (rev_growth, opex_change, liab_paydown), ratio_threshold)
        for r in recs[:3]:
            st.write(f"• {r}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### If X, then Y")
        st.markdown(
            f"""
            **If** revenue grows by **{rev_growth:.1f}%**, OpEx changes by **{opex_change:.1f}%**, and liabilities improve by **{liab_paydown:.1f}%**,  
            **then** next-quarter Debt-to-Income is projected to be **{proj_ratio:.2f}** (baseline: **{base_ratio:.2f}**).
            """
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Scenario Revenue", money(proj_rev), money(proj_rev - base_rev))
        d2.metric("Scenario Liabilities", money(proj_liab), money(proj_liab - base_liab))
        d3.metric("Scenario DTI", f"{proj_ratio:,.2f}" if not np.isnan(proj_ratio) else "—", f"{delta:+.2f}" if not np.isnan(delta) else None)

        gcol, bcol = st.columns([1, 1])

        with gcol:
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
            gauge.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(gauge, use_container_width=True)

        with bcol:
            compare = pd.DataFrame({"Type": ["Baseline", "Scenario"], "Debt-to-Income": [base_ratio, proj_ratio]})
            fig = px.bar(compare, x="Type", y="Debt-to-Income", title="Baseline vs Scenario DTI")
            fig.update_layout(template="plotly_dark", height=320)
            fig.add_hline(y=ratio_threshold, line_dash="dash", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)


def render_recommendations():
    st.subheader("Recommendations & Risks")

    left, right = st.columns([1, 3], vertical_alignment="top")

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


def render_dictionary():
    st.subheader("Dictionary (What everything means + how to use this dashboard)")

    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### Quick Start (Executives)")
        st.write("1) Start on **Overview** to see where DTI is today.")
        st.write("2) Go to **Forecast** to see baseline next-quarter direction.")
        st.write("3) Use **Scenario Planner** to test decisions: *If X, then Y*.")
        st.write("4) Review **Recommendations & Risks** for actions + triggers.")
        st.markdown("---")
        st.markdown("### KPI at a glance")
        st.write("**Debt-to-Income (DTI)** = **Total Liabilities ÷ Revenue**")
        st.write("• Lower is better (less liability per $1 of revenue).")
        st.write("• If DTI rises, liabilities are growing faster than revenue.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("## Key terms & metrics")

        with st.expander("Core KPI: Debt-to-Income (DTI)", expanded=True):
            st.write("**Definition:** Total Liabilities ÷ Revenue (quarterly).")
            st.write("**Interpretation:**")
            st.write("• DTI = 1.5 means $1.50 of liabilities for every $1.00 of quarterly revenue.")
            st.write("• Improving DTI requires increasing revenue, reducing liabilities, or both.")
            st.write("**Used in:** Executive Summary, Overview trend chart, Scenario Planner.")

        with st.expander("Revenue (Revenue_USD)", expanded=False):
            st.write("Top-line quarterly revenue. Increasing revenue improves DTI (it’s the denominator).")

        with st.expander("Total Liabilities (Total_Liabilities_USD)", expanded=False):
            st.write("All liabilities on the balance sheet. Reducing liabilities improves DTI (it’s the numerator).")

        with st.expander("Operating Income (Operating_Income_USD)", expanded=False):
            st.write("Operating profit. Not directly in DTI, but a key signal of operating health and ability to pay down liabilities.")

        with st.expander("Operating Expenses (Total_Operating_Expenses_USD)", expanded=False):
            st.write("Total operating costs. In Scenario Planner, OpEx is a **conceptual lever**: cutting OpEx can reduce pressure on liabilities.")

        with st.expander("Forecasting model (Ridge regression)", expanded=False):
            st.write("A simple, explainable model that captures trend + seasonality.")
            st.write("**Why:** Small quarterly dataset + correlated financial variables.")
            st.write("**Output:** Baseline next-quarter Revenue and Liabilities (used for baseline DTI).")

        with st.expander("Backtest metrics (MAE, RMSE, MAPE)", expanded=False):
            st.write("These describe forecasting error using walk-forward testing.")
            st.write("• **MAE:** average absolute error (in dollars).")
            st.write("• **RMSE:** like MAE but penalizes large misses more (in dollars).")
            st.write("• **MAPE:** average percent error (easy to interpret). Lower is better.")
            st.write("Use them as a **confidence indicator**, not a guarantee.")

        st.markdown("## How to use Scenario Planner (If X, then Y)")
        st.markdown(
            """
            **Scenario Planner inputs:**
            - **Revenue growth (%)**: your revenue improvement assumption next quarter.
            - **OpEx change (%)**: conceptual efficiency lever (cuts reduce liability pressure slightly).
            - **Liabilities improvement (%)**: paydown/refinancing effect next quarter.

            **Scenario Planner outputs:**
            - **Scenario DTI**: projected next-quarter DTI given your assumptions.
            - **Change vs baseline**: improvement or worsening compared with baseline.
            - **Baseline vs Scenario visuals**: gauge + bar chart.

            **Practical executive use:**
            - Try a few combinations until Scenario DTI is below the threshold.
            - Use the recommended actions to decide which levers are most realistic.
            """
        )

        st.markdown("## Limitations (important)")
        st.markdown(
            """
            - The dataset is simulated SEC-style data; real results may vary.
            - Forecast assumes stable patterns; major financing/market changes can break patterns.
            - Scenario Planner uses transparent business rules and should be treated as decision support.
            """
        )


# ----------------------------
# Route
# ----------------------------
if page == "Overview":
    render_overview()
elif page == "Forecast":
    render_forecast()
elif page == "Scenario Planner":
    render_scenario_planner()
elif page == "Recommendations & Risks":
    render_recommendations()
else:
    render_dictionary()