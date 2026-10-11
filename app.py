"""
Streamlit Web Application: Predictive Forecasting of Care Load & Placement Demand
UAC Program Intelligence Dashboard for HHS/ORR and Child-Welfare Decision-Makers.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Set root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_processor import (
    load_and_clean_data, prepare_continuous_timeseries,
    perform_stl_decomposition, compute_exploratory_stats
)
from src.feature_engineering import build_forecasting_features, get_feature_columns
from src.models import (
    NaivePersistenceForecaster, MovingAverageForecaster,
    HoltWintersForecaster, SARIMAForecaster, MLForecaster
)
from src.evaluation import calculate_metrics, calculate_operational_kpis
from src.scenario_simulator import run_scenario_simulation, get_standard_policy_scenarios

# Streamlit Page Config
st.set_page_config(
    page_title="UAC Care Load & Placement Demand Forecasting",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta-positive {
        color: #ef4444;
        font-size: 12px;
        font-weight: 600;
    }
    .metric-delta-negative {
        color: #10b981;
        font-size: 12px;
        font-weight: 600;
    }
    .status-badge-normal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #059669;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #d97706;
        display: inline-block;
    }
    .status-badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #dc2626;
        display: inline-block;
    }
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    raw_path = os.path.join(BASE_DIR, 'data', 'uac_daily_raw.csv')
    df_raw = load_and_clean_data(raw_path)
    df_continuous = prepare_continuous_timeseries(df_raw, method='time')
    feat_df = build_forecasting_features(df_continuous, target_col='hhs_in_care')
    
    # Load precomputed benchmarks if available
    summary_care_path = os.path.join(BASE_DIR, 'data', 'benchmark_summary_care.csv')
    horizons_care_path = os.path.join(BASE_DIR, 'data', 'benchmark_horizons_care.csv')
    summary_disch_path = os.path.join(BASE_DIR, 'data', 'benchmark_summary_discharges.csv')
    horizons_disch_path = os.path.join(BASE_DIR, 'data', 'benchmark_horizons_discharges.csv')
    decomp_care_path = os.path.join(BASE_DIR, 'data', 'stl_decomposition_care.csv')
    feat_imp_path = os.path.join(BASE_DIR, 'data', 'feature_importances_care.csv')
    
    bench_care = pd.read_csv(summary_care_path) if os.path.exists(summary_care_path) else None
    horiz_care = pd.read_csv(horizons_care_path) if os.path.exists(horizons_care_path) else None
    bench_disch = pd.read_csv(summary_disch_path) if os.path.exists(summary_disch_path) else None
    horiz_disch = pd.read_csv(horizons_disch_path) if os.path.exists(horizons_disch_path) else None
    decomp_care = pd.read_csv(decomp_care_path) if os.path.exists(decomp_care_path) else None
    
    feat_imp = None
    if os.path.exists(feat_imp_path):
        feat_imp = pd.read_csv(feat_imp_path, header=None, names=['feature', 'importance'])
        
    return df_raw, df_continuous, feat_df, bench_care, horiz_care, bench_disch, horiz_disch, decomp_care, feat_imp


df_raw, df_continuous, feat_df, bench_care, horiz_care, bench_disch, horiz_disch, decomp_care, feat_imp = load_cached_data()

# Sidebar Controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Seal_of_the_United_States_Department_of_Health_and_Human_Services.svg/300px-Seal_of_the_United_States_Department_of_Health_and_Human_Services.svg.png", width=70)
st.sidebar.title("Operational Controls")
st.sidebar.markdown("**UAC Predictive Forecasting**")
st.sidebar.markdown("---")

# Forecast Horizon Selector
horizon_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
    help="Select forecasting projection window into the future."
)

# Operational Capacity Slider
capacity_limit = st.sidebar.number_input(
    "HHS Operational Bed Capacity",
    min_value=1500,
    max_value=12000,
    value=3000,
    step=100,
    help="Maximum available shelter bed capacity threshold before emergency intake overflows."
)

surge_threshold = st.sidebar.number_input(
    "Surge Alert Threshold (Beds)",
    min_value=1200,
    max_value=11000,
    value=int(capacity_limit * 0.90),
    step=100,
    help="Threshold triggering early warning staffing and shelter mobilization."
)

# Models to toggle
st.sidebar.markdown("### Model Overlays")
available_models = [
    'SARIMA (Statistical)',
    'Gradient Boosting (ML)',
    'Random Forest (ML)',
    'Naive Persistence (Baseline)',
    '7-Day Moving Average (Baseline)',
    'Holt-Winters (Exponential Smoothing)'
]

selected_models = st.sidebar.multiselect(
    "Active Overlay Models",
    options=available_models,
    default=['SARIMA (Statistical)', 'Gradient Boosting (ML)', 'Naive Persistence (Baseline)']
)

show_confidence = st.sidebar.checkbox("Show 80% & 95% Confidence Intervals", value=True)
ci_level = st.sidebar.radio("Interval Coverage", ["95% Confidence Interval", "80% Confidence Interval", "Both"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset Window:**  \n{df_continuous['date'].min().strftime('%b %d, %Y')} – {df_continuous['date'].max().strftime('%b %d, %Y')}")
st.sidebar.markdown(f"**Total Records:** {len(df_continuous):,} days")

# Header Section
st.title("🏛️ Predictive Forecasting of Care Load & Placement Demand")
st.markdown("""
*Decision-Intelligence System for the **Unaccompanied Alien Children (UAC) Program** — U.S. Department of Health and Human Services (HHS / ORR) & CBP.*
""")

# Top Level KPI Banner
current_care_load = float(df_continuous['hhs_in_care'].iloc[-1])
recent_transfers = float(df_continuous['transfers_to_hhs'].iloc[-7:].mean())
recent_discharges = float(df_continuous['hhs_discharges'].iloc[-7:].mean())
net_pressure = recent_transfers - recent_discharges
recent_apprehensions = float(df_continuous['cbp_apprehensions'].iloc[-7:].mean())

# Generate dynamic forecasts for the selected horizon
last_date = df_continuous['date'].iloc[-1]
fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq='D')
feature_cols = get_feature_columns(feat_df, target_col='hhs_in_care')

# Compute forecasts on the fly for active models
model_predictions = {}


def generate_forecasts(target_col: str):
    preds = {}
    y_series = df_continuous[target_col]
    
    if 'Naive Persistence (Baseline)' in selected_models:
        m = NaivePersistenceForecaster().fit(y_series)
        preds['Naive Persistence'] = m.predict(horizon_days, last_date)
        
    if '7-Day Moving Average (Baseline)' in selected_models:
        m = MovingAverageForecaster(window=7).fit(y_series)
        preds['7-Day Moving Average'] = m.predict(horizon_days, last_date)
        
    if 'Holt-Winters (Exponential Smoothing)' in selected_models:
        m = HoltWintersForecaster(seasonal_periods=7).fit(y_series)
        preds['Holt-Winters'] = m.predict(horizon_days, last_date)
        
    if 'SARIMA (Statistical)' in selected_models:
        m = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(y_series)
        preds['SARIMA'] = m.predict(horizon_days, last_date)
        
    if 'Random Forest (ML)' in selected_models:
        m = MLForecaster(model_type='rf', target_col=target_col).fit(feat_df, feature_cols)
        preds['Random Forest'] = m.predict_recursive(feat_df, horizon_days, last_date)
        
    if 'Gradient Boosting (ML)' in selected_models:
        m = MLForecaster(model_type='gbr', target_col=target_col).fit(feat_df, feature_cols)
        preds['Gradient Boosting'] = m.predict_recursive(feat_df, horizon_days, last_date)
        
    return preds


forecast_results_care = generate_forecasts('hhs_in_care')

# Primary champion forecast for KPIs (SARIMA or Gradient Boosting)
primary_key = 'SARIMA' if 'SARIMA' in forecast_results_care else (
    'Gradient Boosting' if 'Gradient Boosting' in forecast_results_care else list(forecast_results_care.keys())[0]
)
prim_pt, prim_l80, prim_u80, prim_l95, prim_u95 = forecast_results_care[primary_key]
prim_fc_df = pd.DataFrame({
    'date': fc_dates,
    'forecast': prim_pt,
    'lower_80': prim_l80,
    'upper_80': prim_u80,
    'lower_95': prim_l95,
    'upper_95': prim_u95
})

op_kpis = calculate_operational_kpis(df_continuous, prim_fc_df, capacity_limit=capacity_limit, surge_threshold=surge_threshold)

# Display Top KPI Scorecards
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Care Load</div>
        <div class="metric-value">{int(current_care_load):,}</div>
        <div style="font-size:12px; color:#94a3b8;">As of {last_date.strftime('%b %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Projected Peak Load ({horizon_days}d)</div>
        <div class="metric-value">{int(op_kpis['projected_peak_care_load']):,}</div>
        <div style="font-size:12px; color:#94a3b8;">Model: {primary_key}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    delta_class = "metric-delta-positive" if net_pressure > 0 else "metric-delta-negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Daily Flow (7d)</div>
        <div class="metric-value">{net_pressure:+.1f}/day</div>
        <div class="{delta_class}">Transfers: {recent_transfers:.1f} | Exits: {recent_discharges:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    lead_text = f"{op_kpis['surge_lead_time_days']} Days" if op_kpis['surge_lead_time_days'] < 900 else "Safe (>60d)"
    badge_style = "status-badge-critical" if op_kpis['surge_lead_time_days'] < 14 else ("status-badge-warning" if op_kpis['surge_lead_time_days'] < 30 else "status-badge-normal")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Surge Lead Time</div>
        <div class="metric-value">{lead_text}</div>
        <div><span class="{badge_style}">{op_kpis['stress_level']} RISK</span></div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    util = op_kpis['projected_bed_utilization_pct']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Bed Capacity Utilization</div>
        <div class="metric-value">{util:.1f}%</div>
        <div style="font-size:12px; color:#94a3b8;">Limit: {int(capacity_limit):,} beds</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "📈 Module 1: Care Load Forecast",
    "🚪 Module 2: Placement Demand",
    "🏆 Module 3: Model Comparison",
    "🚨 Module 4: Early Warning Matrix",
    "🧪 Module 5: What-If Scenarios",
    "🔍 Module 6: System Dynamics & EDA",
    "📋 Module 7: Historical Data"
])

# =========================================================================
# TAB 1: FUTURE CARE LOAD FORECAST CHART
# =========================================================================
with tabs[0]:
    st.subheader("Future Care Load Forecast & Confidence Interval Bands")
    st.markdown(
        f"Multi-model predictive trajectory of children under active HHS care over a **{horizon_days}-day horizon**."
    )
    
    # Historical lookback selector
    lookback_days = st.selectbox(
        "Historical Lookback Window:",
        options=[60, 90, 180, 365, "Entire History (2023-2025)"],
        index=1
    )
    
    if lookback_days == "Entire History (2023-2025)":
        hist_df = df_continuous.copy()
    else:
        hist_df = df_continuous.iloc[-int(lookback_days):].copy()
        
    fig = go.Figure()
    
    # Historical Line
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['hhs_in_care'],
        name="Historical Care Load",
        mode='lines',
        line=dict(color='#94a3b8', width=2.5)
    ))
    
    # Color palette for forecasting models
    color_map = {
        'SARIMA': '#38bdf8',
        'Gradient Boosting': '#f59e0b',
        'Random Forest': '#10b981',
        'Naive Persistence': '#ec4899',
        '7-Day Moving Average': '#a855f7',
        'Holt-Winters': '#ef4444'
    }
    
    # Add forecasts
    for m_name, preds in forecast_results_care.items():
        pt, l80, u80, l95, u95 = preds
        c_code = color_map.get(m_name, '#38bdf8')
        
        # Connect to last historical point
        connect_dates = [hist_df['date'].iloc[-1]] + list(fc_dates)
        connect_pt = [hist_df['hhs_in_care'].iloc[-1]] + list(pt)
        
        fig.add_trace(go.Scatter(
            x=connect_dates,
            y=connect_pt,
            name=f"{m_name} Forecast",
            mode='lines',
            line=dict(color=c_code, width=3, dash='solid')
        ))
        
        # If confidence interval enabled for the primary champion model
        if show_confidence and m_name == primary_key:
            if ci_level in ["95% Confidence Interval", "Both"]:
                fig.add_trace(go.Scatter(
                    x=list(fc_dates) + list(fc_dates)[::-1],
                    y=list(u95) + list(np.maximum(0, l95))[::-1],
                    fill='toself',
                    fillcolor='rgba(56, 189, 248, 0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{m_name} 95% Confidence Interval",
                    showlegend=True
                ))
            if ci_level in ["80% Confidence Interval", "Both"]:
                fig.add_trace(go.Scatter(
                    x=list(fc_dates) + list(fc_dates)[::-1],
                    y=list(u80) + list(np.maximum(0, l80))[::-1],
                    fill='toself',
                    fillcolor='rgba(56, 189, 248, 0.22)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{m_name} 80% Confidence Interval",
                    showlegend=True
                ))
                
    # Capacity Limits
    fig.add_hline(
        y=capacity_limit,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Operational Bed Limit ({int(capacity_limit):,})",
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=surge_threshold,
        line_dash="dot",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text=f"Surge Mobilization Threshold ({int(surge_threshold):,})",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Children in HHS Care",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Data Table
    with st.expander("📊 View Detailed Numerical Forecast Projections"):
        st.dataframe(prim_fc_df, use_container_width=True)

# =========================================================================
# TAB 2: DISCHARGE DEMAND FORECAST PANEL
# =========================================================================
with tabs[1]:
    st.subheader("Placement Demand & Exit Velocity Forecasting")
    st.markdown("""
    Predictive estimates of children discharged from HHS Care to vetted sponsors.
    This panel enables child-welfare planners to detect placement backlogs and sponsor matching deficits.
    """)
    
    forecast_results_disch = generate_forecasts('hhs_discharges')
    
    d1, d2 = st.columns([2, 1])
    
    with d1:
        fig_d = go.Figure()
        
        # Historical discharges
        fig_d.add_trace(go.Scatter(
            x=df_continuous['date'].iloc[-90:],
            y=df_continuous['hhs_discharges'].iloc[-90:],
            name="Observed Daily Discharges",
            mode='lines',
            line=dict(color='#64748b', width=1.5)
        ))
        
        for m_name, preds in forecast_results_disch.items():
            pt_d, _, _, _, _ = preds
            c_code = color_map.get(m_name, '#38bdf8')
            fig_d.add_trace(go.Scatter(
                x=fc_dates,
                y=pt_d,
                name=f"{m_name} Placement Demand",
                mode='lines+markers',
                line=dict(color=c_code, width=2.5)
            ))
            
        fig_d.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Date",
            yaxis_title="Discharges / Placements per Day",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_d, use_container_width=True)
        
    with d2:
        st.markdown("### Placement Balance Analytics")
        total_projected_exits = sum([float(x) for x in list(forecast_results_disch.values())[0][0]])
        daily_req_to_stabilize = recent_transfers
        total_projected_entries = recent_transfers * horizon_days
        exit_deficit = total_projected_entries - total_projected_exits
        
        st.metric(
            label=f"Total Projected Placements ({horizon_days}d)",
            value=f"{int(total_projected_exits):,} Children",
            delta=f"{total_projected_exits / horizon_days:.1f}/day avg"
        )
        st.metric(
            label=f"Projected HHS Intake ({horizon_days}d)",
            value=f"{int(total_projected_entries):,} Children",
            delta=f"{recent_transfers:.1f}/day transfer rate"
        )
        st.metric(
            label="Net Shelter Population Shift",
            value=f"{int(exit_deficit):+} Children",
            delta="Intake exceeds exits" if exit_deficit > 0 else "Exits exceed intake",
            delta_color="inverse"
        )
        
        if exit_deficit > 0:
            st.warning(f"⚠️ Placement bottleneck detected: Caseworkers must boost exits by {exit_deficit / horizon_days:.1f} per day to prevent care load accumulation.")
        else:
            st.success("✅ Discharge throughput is currently sufficient to de-escalate active shelter population.")

# =========================================================================
# TAB 3: MODEL SELECTION & COMPARISON BENCHMARK
# =========================================================================
with tabs[2]:
    st.subheader("Model Selection, Validation & Benchmark Comparison")
    st.markdown("""
    Rigorous multi-horizon walk-forward backtesting across **Statistical, Machine Learning, and Baseline** architectures.
    Evaluated with strict temporal splitting (no random sampling) to replicate true operational forecasting conditions.
    """)
    
    if bench_care is not None:
        st.markdown("### 🏆 Care Load Forecasting Benchmark (HHS In Care)")
        st.dataframe(bench_care, use_container_width=True)
        
        # Bar chart comparison
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_mae = px.bar(
                bench_care,
                x='Model',
                y='MAE',
                color='Model',
                title="Mean Absolute Error (MAE) — Lower is Better",
                template="plotly_dark",
                text='MAE'
            )
            fig_mae.update_traces(textposition='outside')
            st.plotly_chart(fig_mae, use_container_width=True)
            
        with col_b2:
            fig_acc = px.bar(
                bench_care,
                x='Model',
                y='Forecast Accuracy (%)',
                color='Model',
                title="Forecast Accuracy (%) — Higher is Better",
                template="plotly_dark",
                text='Forecast Accuracy (%)'
            )
            fig_acc.update_traces(textposition='outside')
            st.plotly_chart(fig_acc, use_container_width=True)
            
    if horiz_care is not None:
        st.markdown("### ⏳ Horizon Error Degradation Curve (7d vs 14d vs 30d)")
        fig_h = px.line(
            horiz_care,
            x='Horizon_Days',
            y='MAE',
            color='Model',
            markers=True,
            title="Forecast Error Growth by Horizon (Multi-Horizon Evaluation)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_h, use_container_width=True)
        
    if feat_imp is not None:
        st.markdown("### 🧠 Machine Learning Feature Importance (Top Predictors)")
        fig_imp = px.bar(
            feat_imp.head(12),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 12 Features Driving Care Load Predictions (Tree Gini Importance)",
            template="plotly_dark",
            color='importance',
            color_continuous_scale='Tealgrn'
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================================
# TAB 4: EARLY WARNING & CAPACITY STRESS MATRIX
# =========================================================================
with tabs[3]:
    st.subheader("Early-Warning Intelligence & Shelter Resource Matrix")
    st.markdown("""
    Actionable healthcare, shelter staffing, and capacity stress indicators for HHS/ORR leadership.
    """)
    
    ew1, ew2, ew3 = st.columns(3)
    
    with ew1:
        st.markdown("### 🚦 Stress Level Indicator")
        if op_kpis['stress_level'] == 'CRITICAL':
            st.error("🚨 **CRITICAL CAPACITY STRESS**  \nHigh probability of exceeding operational limits within forecast window.")
        elif op_kpis['stress_level'] == 'ELEVATED':
            st.warning("⚠️ **ELEVATED READINESS REQUIRED**  \nCare load approaching surge mobilization trigger.")
        else:
            st.success("🟢 **NORMAL OPERATING POSTURE**  \nCare load comfortably within shelter capacity buffer.")
            
        st.metric("Capacity Breach Probability", f"{op_kpis['capacity_breach_prob_pct']}%")
        st.metric("Surge Lead Time", f"{op_kpis['surge_lead_time_days']} Days" if op_kpis['surge_lead_time_days'] < 900 else "Safe (>60d)")
        
    with ew2:
        st.markdown("### 🛏️ Bed Capacity & Buffers")
        curr_beds = current_care_load
        buffer_beds = max(0, capacity_limit - curr_beds)
        st.metric("Available Surplus Beds", f"{int(buffer_beds):,} Beds")
        st.metric("Current Utilization", f"{(curr_beds / capacity_limit)*100:.1f}%")
        st.metric("Projected Peak Utilization", f"{op_kpis['projected_bed_utilization_pct']:.1f}%")
        
    with ew3:
        st.markdown("### 👥 Operational Staffing Calculator")
        peak_projected = op_kpis['projected_peak_care_load']
        # Federal standards
        youth_care_workers = int(np.ceil(peak_projected / 8.0))
        caseworkers = int(np.ceil(peak_projected / 10.0))
        clinicians = int(np.ceil(peak_projected / 25.0))
        
        st.metric("Direct Youth Care Workers (1:8)", f"{youth_care_workers:,} Staff")
        st.metric("Dedicated Caseworkers (1:10)", f"{caseworkers:,} Staff")
        st.metric("Licensed Clinicians (1:25)", f"{clinicians:,} Staff")
        
    st.markdown("---")
    st.markdown("### 📋 4-Tier Operational Readiness Action Protocol")
    readiness_df = pd.DataFrame([
        {
            "Alert Level": "🟢 Stage 1: Normal (<80% Utilization)",
            "Care Load Range": f"< {int(capacity_limit * 0.8):,} Children",
            "Trigger Condition": "Net flow stable, surge lead time > 30 days",
            "Mandated Operational Action": "Routine sponsor vetting, standard 1:8 staffing, scheduled facility maintenance."
        },
        {
            "Alert Level": "🟡 Stage 2: Elevated (80% - 90% Utilization)",
            "Care Load Range": f"{int(capacity_limit * 0.8):,} – {int(capacity_limit * 0.9):,} Children",
            "Trigger Condition": "Net positive intake > 5 children/day for 7 consecutive days",
            "Mandated Operational Action": "Pre-alert standby contractors, expedite Category 1 sponsor releases, review overflow beds."
        },
        {
            "Alert Level": "🟠 Stage 3: Surge Readiness (90% - 98% Utilization)",
            "Care Load Range": f"{int(capacity_limit * 0.9):,} – {int(capacity_limit * 0.98):,} Children",
            "Trigger Condition": "Surge lead time < 14 days, CBP custody backlog > 250",
            "Mandated Operational Action": "Activate Influx Care Facilities (ICFs), authorize staff overtime, deploy mobile case coordinators."
        },
        {
            "Alert Level": "🔴 Stage 4: Critical Emergency (>98% Utilization)",
            "Care Load Range": f"> {int(capacity_limit * 0.98):,} Children",
            "Trigger Condition": "Capacity breach imminent (< 7 days)",
            "Mandated Operational Action": "Execute inter-agency surge agreements (DoD emergency sites), fast-track vetted sponsor handoffs."
        }
    ])
    st.table(readiness_df)

# =========================================================================
# TAB 5: WHAT-IF POLICY SCENARIOS
# =========================================================================
with tabs[4]:
    st.subheader("What-If Policy & Operational Stress Simulator")
    st.markdown("""
    Evaluate how extreme border apprehension surges, policy transitions, or sponsor release bottlenecks
    impact shelter bed demand and staffing thresholds.
    """)
    
    scenarios = get_standard_policy_scenarios()
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("### Scenario Configuration")
        scenario_choice = st.selectbox(
            "Select Pre-configured Policy Shock:",
            options=list(scenarios.keys()),
            index=2
        )
        scen_info = scenarios[scenario_choice]
        st.info(f"**Scenario Rationale:** {scen_info['description']}")
        
        custom_intake_shock = st.slider(
            "Custom Border Intake Shock (%)",
            min_value=-50,
            max_value=150,
            value=int(scen_info['intake_shock_pct']),
            step=5
        )
        custom_discharge_shock = st.slider(
            "Custom Placement Disruption (%)",
            min_value=-80,
            max_value=80,
            value=int(scen_info['discharge_shock_pct']),
            step=5
        )
        
    sim_df, sim_impact = run_scenario_simulation(
        prim_fc_df,
        initial_care_load=current_care_load,
        recent_intake=recent_transfers,
        recent_discharge=recent_discharges,
        horizon_days=horizon_days,
        intake_shock_pct=custom_intake_shock,
        discharge_shock_pct=custom_discharge_shock,
        capacity_limit=capacity_limit
    )
    
    with col_s2:
        st.markdown("### Stress Test Projection Curve")
        fig_sim = go.Figure()
        
        # Baseline forecast
        fig_sim.add_trace(go.Scatter(
            x=prim_fc_df['date'],
            y=prim_fc_df['forecast'],
            name="Baseline Forecast (Status Quo)",
            line=dict(color='#38bdf8', width=2.5, dash='dot')
        ))
        
        # Simulated curve
        fig_sim.add_trace(go.Scatter(
            x=sim_df['date'],
            y=sim_df['simulated_care_load'],
            name=f"Stress Test: {scenario_choice}",
            line=dict(color='#ef4444' if sim_impact['net_load_change'] > 0 else '#10b981', width=3.5)
        ))
        
        fig_sim.add_hline(
            y=capacity_limit,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Capacity Limit ({int(capacity_limit):,})"
        )
        
        fig_sim.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Date",
            yaxis_title="Simulated Care Load",
            hovermode="x unified"
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
    # Scenario Impact Summary
    st.markdown("### 📊 Scenario Operational Impact Analysis")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Peak Simulated Load", f"{int(sim_impact['peak_care_load']):,} Children", f"{sim_impact['net_load_change']:+} Net Change")
    with sc2:
        st.metric("Peak Bed Utilization", f"{sim_impact['peak_bed_utilization_pct']}%", "Capacity Overrun!" if sim_impact['capacity_breached'] else "Within Limits")
    with sc3:
        breach_text = f"Day {sim_impact['days_to_breach']}" if sim_impact['days_to_breach'] else "None"
        st.metric("Days Until Bed Exhaustion", breach_text)
    with sc4:
        st.metric("Surge Beds Needed", f"{int(sim_impact['bed_deficit']):,} Beds")

# =========================================================================
# TAB 6: SYSTEM DYNAMICS & TIME-SERIES DIAGNOSTICS
# =========================================================================
with tabs[5]:
    st.subheader("System Dynamics, Pipeline Transfers & Time-Series Diagnostics")
    st.markdown("""
    Exploration of the upstream pipeline mechanics: Border Apprehensions ➔ CBP Custody ➔ HHS Intake ➔ HHS Discharges.
    """)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### Upstream Pipeline Transfer Relationship")
        fig_pipe = px.scatter(
            df_continuous,
            x='cbp_apprehensions',
            y='transfers_to_hhs',
            trendline='ols',
            title="CBP Daily Apprehensions vs Transfers to HHS (r = 0.904)",
            template="plotly_dark",
            opacity=0.6,
            labels={'cbp_apprehensions': 'CBP Apprehensions', 'transfers_to_hhs': 'Transfers into HHS'}
        )
        st.plotly_chart(fig_pipe, use_container_width=True)
        
    with col_d2:
        st.markdown("### Stock vs Flow Balance Dynamic")
        fig_flow = px.scatter(
            df_continuous,
            x='net_flow',
            y='care_load_diff',
            trendline='ols',
            title="Net Flow (Transfers - Exits) vs Daily Care Load Shift (Stock-Flow Identity)",
            template="plotly_dark",
            opacity=0.6,
            labels={'net_flow': 'Net Flow (Transfers - Discharges)', 'care_load_diff': 'Daily Δ Care Load'}
        )
        st.plotly_chart(fig_flow, use_container_width=True)
        
    if decomp_care is not None:
        st.markdown("### 🧩 Time-Series Decomposition (Trend, Seasonality, Residuals)")
        decomp_care['date'] = pd.to_datetime(decomp_care['date'])
        
        fig_stl = go.Figure()
        fig_stl.add_trace(go.Scatter(x=decomp_care['date'], y=decomp_care['observed'], name="Observed", line=dict(color='#94a3b8')))
        fig_stl.add_trace(go.Scatter(x=decomp_care['date'], y=decomp_care['trend'], name="Trend Component", line=dict(color='#38bdf8', width=2.5)))
        fig_stl.add_trace(go.Scatter(x=decomp_care['date'], y=decomp_care['seasonal'], name="7-Day Seasonality", line=dict(color='#f59e0b', width=1)))
        
        fig_stl.update_layout(
            template="plotly_dark",
            height=400,
            title="Additive Seasonal-Trend Decomposition (STL) of Active Care Load",
            xaxis_title="Date",
            yaxis_title="Care Load Components",
            hovermode="x unified"
        )
        st.plotly_chart(fig_stl, use_container_width=True)

# =========================================================================
# TAB 7: HISTORICAL DATA & DOWNLOADS
# =========================================================================
with tabs[6]:
    st.subheader("Historical UAC Data Repository & CSV Downloads")
    
    st.dataframe(df_continuous, use_container_width=True)
    
    csv_data = df_continuous.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Continuous Daily UAC Dataset (CSV)",
        data=csv_data,
        file_name="uac_daily_continuous_dataset.csv",
        mime="text/csv"
    )
    
    fc_csv = prim_fc_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Forward Forecasts ({primary_key}) (CSV)",
        data=fc_csv,
        file_name="uac_forward_forecasts.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px;">
    <strong>UAC Program Predictive Analytics Engine</strong> | U.S. Department of Health and Human Services (HHS) & CBP Intelligence Unit<br>
    Built for Proactive Child Welfare & Healthcare Operational Planning.
</div>
""", unsafe_allow_html=True)
