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
    layout="wide"
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

# Hide sidebar + Streamlit chrome (no menu)
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      #MainMenu { visibility: hidden; }
      header { visibility: hidden; }
      footer { visibility: hidden; }

      .stApp {
        background: linear-gradient(180deg, #070A12 0%, #050712 100%);
        color: #E9ECF5;
      }
      h1,h2,h3,h4 { color: #E9ECF5; }

      .cw-badge{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(47, 91, 234, 0.14);
        border: 1px solid rgba(47, 91, 234, 0.35);
        color: #E9ECF5;
        font-size: 12px;
        letter-spacing: 0.2px;
      }
      .cw-card{
        background: #0E1424;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .cw-muted { color: #A7B0C3; }
      .cw-warn { color: #FFB020; }

      div[data-testid="metric-container"]{
        background: #0E1424;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 14px;
      }

      .stTabs [data-baseweb="tab-list"] button { background: transparent !important; }
      .stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{
        border-bottom: 3px solid #2F5BEA !important;
      }
      a { color: #2F5BEA !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Paths (repo-relative)
# ----------------------------
DEFAULT_XLSX_PATH = "data/CoreWeave_BalanceSheet_SEC_Filings_simulated.xlsx"
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
    if not os.path.exists(DEFAULT_XLSX_PATH):
        raise FileNotFoundError(
            f"Data file not found at '{DEFAULT_XLSX_PATH}'. "
            f"Make sure it exists in your repo."
        )
    return pd.read_excel(DEFAULT_XLSX_PATH)


def prep_df(df_raw: pd.DataFrame):
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

    # Quarterly-only for forecasting (FY duplicates dates; keep FY in df_all if you want)
    df_q = df[df["Period_Type"].str.startswith("Q", na=False)].copy()
    df_q = df_q.sort_values("Date").reset_index(drop=True)

    df_q["Quarter"] = df_q["Date"].dt.quarter
    df_q["t"] = np.arange(len(df_q), dtype=int)

    # Core KPI: Debt-to-Income proxy = Total Liabilities / Revenue
    df_q["Debt_to_Income"] = df_q.apply(
        lambda r: safe_div(r.get("Total_Liabilities_USD", np.nan), r.get("Revenue_USD", np.nan)),
        axis=1
    )

    # Optional: operating margin
    df_q["Op_Margin"] = df_q.apply(
        lambda r: safe_div(r.get("Operating_Income_USD", np.nan), r.get("Revenue_USD", np.nan)),
        axis=1
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
    return pipe


def time_series_backtest(df_q: pd.DataFrame, target_col: str, min_train: int = 6):
    """
    Expanding-window backtest:
      train on [0:i), test on i for i >= min_train
    """
    pipe = build_model()

    use_cols = [c for c in FEATURE_COLS if c in df_q.columns] + ["Quarter"]
    use_cols = list(dict.fromkeys(use_cols))  # de-dupe

    dfx = df_q.dropna(subset=[target_col]).copy()
    if len(dfx) < (min_train + 2):
        return None

    preds, actuals, dates = [], [], []

    for i in range(min_train, len(dfx)):
        train = dfx.iloc[:i].dropna(subset=use_cols + [target_col]).copy()
        test = dfx.iloc[i:i+1].dropna(subset=use_cols + [target_col]).copy()

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
    return {"series": out, "mae": mae, "rmse": rmse, "mape": mape_val}


def fit_and_forecast_next(df_q: pd.DataFrame, target_col: str):
    """
    Fit on all available quarterly data (with target present) and forecast next quarter.
    Baseline assumes expense drivers carry forward from last observed quarter.
    """
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

    X_next = pd.DataFrame([next_row])[use_cols]
    y_next = float(pipe.predict(X_next)[0])

    return {"next_date": next_date, "next_pred": y_next}


def rule_recommendation(current_ratio, projected_ratio, levers, threshold):
    rev_growth, opex_change, liab_paydown = levers
    msgs = []

    if projected_ratio is None or np.isnan(projected_ratio) or projected_ratio <= 0:
        return ["Check inputs: projected ratio is invalid."]

    if projected_ratio >= 2.0:
        msgs.append("High risk: ratio remains very elevated. Prioritize liability reduction/refinancing and revenue quality (higher-margin contracts).")
    elif projected_ratio >= threshold:
        msgs.append("Moderate risk: ratio is still above the threshold. Combine revenue growth with disciplined cost controls and liability strategy.")
    else:
        msgs.append("Improving: projected ratio is below threshold. Maintain controls and avoid over-leveraging future expansion.")

    if opex_change > 0 and projected_ratio >= threshold:
        msgs.append("OpEx increases while ratio is high—consider freezing discretionary spend and tying new spend to contracted revenue.")

    if rev_growth < 5 and projected_ratio >= threshold:
        msgs.append("Revenue growth assumption is modest; consider improving utilization, pricing, and longer-term enterprise commitments.")

    if liab_paydown < 2 and projected_ratio >= threshold:
        msgs.append("Low liability paydown; explore refinancing, term extensions, or targeted paydowns using operating cash flow.")

    msgs.append(f"Levers: Revenue growth {rev_growth:.1f}%, OpEx change {opex_change:.1f}%, Liabilities paydown {liab_paydown:.1f}%.")
    return msgs


# ----------------------------
# Header (logo + title + controls on main page)
# ----------------------------
top_left, top_right = st.columns([1, 5], vertical_alignment="center")

with top_left:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)
    else:
        st.markdown("<span class='cw-badge'>CoreWeave</span>", unsafe_allow_html=True)

with top_right:
    st.markdown("<span class='cw-badge'>Debt-to-Income Improvement</span>", unsafe_allow_html=True)
    st.title("Debt-to-Income Strategy Dashboard")
    st.caption("Objective: provide advice on improving Debt-to-Income ratio (Total Liabilities ÷ Revenue).")

st.markdown("")

# ----------------------------
# Main-page controls (no sidebar)
# ----------------------------
st.markdown("## Control Panel")
ctrl1, ctrl2 = st.columns(2)

with ctrl1:
    ratio_threshold = st.slider(
        "Debt-to-Income alert threshold",
        min_value=0.5, max_value=3.0, value=1.2, step=0.1
    )

with ctrl2:
    backtest_min_train = st.slider(
        "Backtest minimum training quarters",
        min_value=4, max_value=12, value=6, step=1
    )

st.markdown("")

# ----------------------------
# Load + prep data
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
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Predictive (Forecasting)",
    "Prescriptive (Scenarios)",
    "Insights & Decision Implications"
])

# ----------------------------
# Overview
# ----------------------------
with tab1:
    latest = df_q.sort_values("Date").iloc[-1]
    current_rev = float(latest.get("Revenue_USD", np.nan))
    current_liab = float(latest.get("Total_Liabilities_USD", np.nan))
    current_ratio = float(latest.get("Debt_to_Income", np.nan))
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
        st.subheader("KPI Definition")
        st.markdown(
            """
            **Debt-to-Income (proxy)** = **Total Liabilities ÷ Revenue**  
            - Higher is worse (more liabilities per $1 of revenue)  
            - Improve by **increasing revenue**, **reducing liabilities**, and **improving operating efficiency**
            """,
        )

        if not np.isnan(current_ratio):
            if current_ratio >= ratio_threshold:
                st.markdown(
                    f"<span class='cw-warn'>Alert:</span> Current ratio **{current_ratio:.2f}** ≥ threshold **{ratio_threshold:.2f}**.",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"Current ratio **{current_ratio:.2f}** is below threshold **{ratio_threshold:.2f}**.")

        st.markdown("---")
        st.subheader("Data Preview (Quarterly)")
        preview_cols = [c for c in [
            "Period", "Date", "Revenue_USD", "Total_Liabilities_USD", "Debt_to_Income",
            "Technology_Infra_USD", "Total_Operating_Expenses_USD", "Operating_Income_USD"
        ] if c in df_q.columns]
        st.dataframe(
            df_q[preview_cols].sort_values("Date", ascending=False),
            use_container_width=True,
            height=320
        )
        st.markdown("</div>", unsafe_allow_html=True)

    csv = df_q.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned quarterly dataset (CSV)",
        data=csv,
        file_name="coreweave_quarterly_clean.csv",
        mime="text/csv"
    )

# ----------------------------
# Predictive (required)
# ----------------------------
with tab2:
    st.subheader("Predictive Approach: Time-series style forecasting with Ridge Regression")

    st.markdown(
        """
        **Why this model?**
        - Quarterly data + small sample → a simple, explainable approach is best.
        - Features: time index (**t**) + quarter indicators (**seasonality**).
        - **Ridge regression** reduces overfitting when financial variables move together.

        **What is being predicted?**
        - Next-quarter **Total Liabilities**
        - Next-quarter **Revenue**
        - Derived: **Debt-to-Income** = Liabilities ÷ Revenue

        **How accurate and useful is it?**
        - We use a walk-forward (expanding window) backtest and report **MAE**, **RMSE**, and **MAPE**.
        """
    )

    bt_liab = time_series_backtest(df_q, "Total_Liabilities_USD", min_train=backtest_min_train)
    bt_rev = time_series_backtest(df_q, "Revenue_USD", min_train=backtest_min_train)

    if bt_liab is None or bt_rev is None:
        st.warning("Not enough clean quarterly data to run the backtest/forecast (need more quarters with non-missing fields).")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Liabilities MAE", money(bt_liab["mae"]))
            st.metric("Liabilities RMSE", money(bt_liab["rmse"]))
        with m2:
            st.metric("Liabilities MAPE", pct(bt_liab["mape"]))
            st.metric("Revenue MAE", money(bt_rev["mae"]))
        with m3:
            st.metric("Revenue RMSE", money(bt_rev["rmse"]))
            st.metric("Revenue MAPE", pct(bt_rev["mape"]))

        st.caption("Lower MAE/RMSE/MAPE indicates better forecasting performance.")

        colA, colB = st.columns(2)
        with colA:
            s = bt_liab["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Total Liabilities (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            s = bt_rev["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=s["Date"], y=s["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(title="Backtest: Revenue (Actual vs Predicted)", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        fc_liab = fit_and_forecast_next(df_q, "Total_Liabilities_USD")
        fc_rev = fit_and_forecast_next(df_q, "Revenue_USD")

        if fc_liab and fc_rev:
            next_ratio = safe_div(fc_liab["next_pred"], fc_rev["next_pred"])
            st.markdown("### Next-Quarter Forecast (Baseline)")

            f1, f2, f3 = st.columns(3)
            with f1:
                st.metric("Forecast Revenue", money(fc_rev["next_pred"]))
            with f2:
                st.metric("Forecast Total Liabilities", money(fc_liab["next_pred"]))
            with f3:
                st.metric("Forecast Debt-to-Income", f"{next_ratio:,.2f}" if not np.isnan(next_ratio) else "—")

            st.caption(
                "Baseline forecast carries forward last observed drivers and uses trend/seasonality. "
                "Use the Prescriptive tab to test improvements."
            )

# ----------------------------
# Prescriptive (required: light implementation)
# ----------------------------
with tab3:
    st.subheader("Prescriptive Approach: Scenario Simulator + If–Then Recommendations")

    st.markdown(
        """
        This translates forecasts into **actionable levers** to improve Debt-to-Income.

        **Levers (simple + explainable):**
        - **Revenue growth (%)**
        - **OpEx change (%)** *(conceptual efficiency lever)*
        - **Liabilities paydown/refinancing effect (%)**

        We compute:
        - Projected Revenue = baseline forecast × (1 + revenue growth)
        - Projected Liabilities = baseline forecast × (1 − paydown) × (small OpEx effect)
        - Projected Debt-to-Income = projected liabilities ÷ projected revenue
        """
    )

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
            rev_growth = st.slider("Revenue growth (%)", -10.0, 60.0, 10.0, 0.5, key="rev_growth")
        with s2:
            opex_change = st.slider("OpEx change (%)", -40.0, 20.0, -5.0, 0.5, key="opex_change")
        with s3:
            liab_paydown = st.slider("Liabilities paydown / refinancing effect (%)", 0.0, 30.0, 5.0, 0.5, key="liab_paydown")

        proj_rev = base_rev * (1.0 + rev_growth / 100.0)

        # Transparent (light) business rule: OpEx reductions slightly reduce liability pressure
        opex_factor = 1.0 - (max(0.0, -opex_change) / 100.0) * 0.10  # 10% of OpEx cut -> modest pressure reduction
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
        recs = rule_recommendation(
            current_ratio=current_ratio,
            projected_ratio=proj_ratio,
            levers=(rev_growth, opex_change, liab_paydown),
            threshold=ratio_threshold
        )
        for r in recs:
            st.write(f"• {r}")

        st.markdown("---")
        st.markdown("#### Scenario Comparison (Quick Grid)")
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
# Insights & Decision Implications (required)
# ----------------------------
with tab4:
    st.subheader("Insights & Decision Implications")

    latest = df_q.sort_values("Date").iloc[-1]
    current_ratio = float(latest.get("Debt_to_Income", np.nan))

    driver_cols = [c for c in [
        "Technology_Infra_USD",
        "Total_Operating_Expenses_USD",
        "Operating_Income_USD",
        "Revenue_USD",
        "Cost_of_Revenue_USD",
        "Sales_Marketing_USD",
        "General_Admin_USD"
    ] if c in df_q.columns]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("### What the results mean")
        if not np.isnan(current_ratio):
            st.write(f"Latest Debt-to-Income: **{current_ratio:.2f}**")
            if current_ratio >= ratio_threshold:
                st.write(f"Status: **Above** threshold (**{ratio_threshold:.2f}**) → prioritize reducing liabilities and/or accelerating revenue.")
            else:
                st.write(f"Status: **Below** threshold (**{ratio_threshold:.2f}**) → maintain discipline to keep liabilities from outpacing revenue.")
        else:
            st.write("Debt-to-Income unavailable due to missing liabilities or revenue in the latest quarter.")

        st.markdown("---")
        st.markdown("### Directional drivers (correlation)")
        st.caption("Directional signals only; correlation ≠ causation.")

        corr_block = df_q[driver_cols + ["Total_Liabilities_USD", "Debt_to_Income"]].corr(numeric_only=True)

        if "Total_Liabilities_USD" in corr_block.columns:
            corr_liab = corr_block["Total_Liabilities_USD"].dropna().sort_values(ascending=False)
            st.write("**Top correlations with Total Liabilities:**")
            for k, v in corr_liab.head(5).items():
                st.write(f"• {k}: {v:,.2f}")

        if "Debt_to_Income" in corr_block.columns:
            corr_dti = corr_block["Debt_to_Income"].dropna().sort_values(ascending=False)
            st.write("**Top correlations with Debt-to-Income:**")
            for k, v in corr_dti.head(5).items():
                st.write(f"• {k}: {v:,.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
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
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough non-missing rows to plot Revenue vs Liabilities scatter.")

    st.markdown("### Recommended actions to improve Debt-to-Income")
    st.markdown(
        """
        **Improve the ratio by increasing the denominator (Revenue), decreasing the numerator (Liabilities), and improving efficiency:**

        **1) Revenue quality & utilization**
        - Increase GPU utilization (more billable output per unit of infra).
        - Prioritize longer-term enterprise commitments to stabilize revenue.

        **2) Operating discipline**
        - Limit non-core OpEx growth; tie new spending to contracted or near-term realizable revenue.
        - Guardrail: *OpEx growth should not exceed revenue growth for extended periods.*

        **3) Liability strategy**
        - Explore refinancing, term extensions, or targeted paydowns to reduce liabilities relative to revenue.
        - Align financing structures with asset life cycles to reduce mismatch risk.

        **4) Governance**
        - Track Debt-to-Income quarterly with thresholds and triggers:
          - If **DTI > threshold**, require a mitigation plan.
          - If **DTI improves for 2+ quarters**, cautiously re-accelerate growth.
        """
    )

    st.markdown("### Uncertainty & limitations")
    st.markdown(
        """
        - Data is simulated SEC-style; real-world results can differ.
        - Forecast assumes stable relationships and may miss structural breaks (market, financing, demand changes).
        - Scenario simulator uses transparent business rules; treat as decision support, not guarantees.
        """
    )

    st.markdown("---")
    st.markdown("### Short narrative (for your write-up)")
    st.write(
        "CoreWeave can improve its debt-to-income ratio by accelerating revenue growth (especially revenue per unit of infrastructure), "
        "maintaining operating discipline so costs don’t outpace realized revenue, and reducing liabilities via refinancing or targeted paydowns. "
        "This dashboard quantifies the historical ratio, forecasts the next quarter, and uses scenario analysis to identify lever combinations that most improve the ratio."
    )