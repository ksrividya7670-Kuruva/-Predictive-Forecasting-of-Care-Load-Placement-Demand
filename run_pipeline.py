"""
End-to-End Orchestration Pipeline for UAC Care Load & Placement Demand Forecasting.
Executes data processing, feature engineering, model training, walk-forward backtesting,
and precomputes operational forecast artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd

# Flush prints immediately
sys.stdout.reconfigure(line_buffering=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import (
    load_and_clean_data, prepare_continuous_timeseries,
    perform_stl_decomposition, test_stationarity, compute_exploratory_stats
)
from src.feature_engineering import build_forecasting_features, get_feature_columns
from src.evaluation import evaluate_walk_forward, calculate_operational_kpis
from src.models import MLForecaster


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'uac_daily_raw.csv')
    
    print("=" * 70, flush=True)
    print("UAC PREDICTIVE CARE LOAD & PLACEMENT DEMAND FORECASTING ENGINE", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load and clean
    print("\n[1/5] Loading and cleaning raw dataset...", flush=True)
    df_raw = load_and_clean_data(raw_path)
    print(f"Loaded {len(df_raw)} records spanning {df_raw['date'].min().strftime('%Y-%m-%d')} to {df_raw['date'].max().strftime('%Y-%m-%d')}", flush=True)
    
    # 2. Continuous time-series preparation
    print("\n[2/5] Constructing continuous daily time-series with interpolation...", flush=True)
    df_continuous = prepare_continuous_timeseries(df_raw, method='time')
    df_continuous.to_csv(os.path.join(base_dir, 'data', 'uac_daily_continuous.csv'), index=False)
    print(f"Constructed continuous grid with {len(df_continuous)} calendar days.", flush=True)
    
    # STL Decomposition
    print("Decomposing HHS Care Load into Trend, Seasonality, Residuals...", flush=True)
    decomp_care = perform_stl_decomposition(df_continuous, target_col='hhs_in_care', period=7)
    decomp_care.to_csv(os.path.join(base_dir, 'data', 'stl_decomposition_care.csv'), index=False)
    
    decomp_disch = perform_stl_decomposition(df_continuous, target_col='hhs_discharges', period=7)
    decomp_disch.to_csv(os.path.join(base_dir, 'data', 'stl_decomposition_discharges.csv'), index=False)
    
    # 3. Feature Engineering
    print("\n[3/5] Engineering predictive features (Lags, Rolling stats, Pipeline flows, Calendar)...", flush=True)
    feat_df = build_forecasting_features(df_continuous, target_col='hhs_in_care')
    feat_df.to_csv(os.path.join(base_dir, 'data', 'uac_features.csv'), index=False)
    feature_cols = get_feature_columns(feat_df)
    print(f"Constructed {len(feature_cols)} engineered features.", flush=True)
    
    # 4. Multi-Horizon Walk-Forward Backtesting for Care Load
    print("\n[4/5] Executing Walk-Forward Backtesting for Care Load (Horizons: 7d, 14d, 30d)...", flush=True)
    df_summary_care, df_horizons_care, _ = evaluate_walk_forward(
        df_continuous, feat_df, target_col='hhs_in_care',
        horizons=[7, 14, 30], n_splits=4
    )
    df_summary_care.to_csv(os.path.join(base_dir, 'data', 'benchmark_summary_care.csv'), index=False)
    df_horizons_care.to_csv(os.path.join(base_dir, 'data', 'benchmark_horizons_care.csv'), index=False)
    print("\n=== Model Benchmark Summary: Care Load ===", flush=True)
    print(df_summary_care.to_string(index=False), flush=True)
    
    # Multi-Horizon Walk-Forward Backtesting for Discharges
    print("\nExecuting Walk-Forward Backtesting for Placement Demand (Discharges)...", flush=True)
    df_summary_disch, df_horizons_disch, _ = evaluate_walk_forward(
        df_continuous, feat_df, target_col='hhs_discharges',
        horizons=[7, 14, 30], n_splits=4
    )
    df_summary_disch.to_csv(os.path.join(base_dir, 'data', 'benchmark_summary_discharges.csv'), index=False)
    df_horizons_disch.to_csv(os.path.join(base_dir, 'data', 'benchmark_horizons_discharges.csv'), index=False)
    print("\n=== Model Benchmark Summary: Placement Demand ===", flush=True)
    print(df_summary_disch.to_string(index=False), flush=True)
    
    # 5. Fit Champion Models and produce forward forecasts
    print("\n[5/5] Generating operational forward forecasts (30-day horizon)...", flush=True)
    last_date = df_continuous['date'].iloc[-1]
    fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    # Care Load Champion: Random Forest / Gradient Boosting
    gbr_care = MLForecaster(model_type='rf', target_col='hhs_in_care').fit(feat_df, feature_cols)
    pt, l80, u80, l95, u95 = gbr_care.predict_recursive(feat_df, horizon=30, last_date=last_date)
    forecast_care_df = pd.DataFrame({
        'date': fc_dates,
        'forecast': np.round(pt, 1),
        'lower_80': np.round(np.maximum(0, l80), 1),
        'upper_80': np.round(u80, 1),
        'lower_95': np.round(np.maximum(0, l95), 1),
        'upper_95': np.round(u95, 1)
    })
    forecast_care_df.to_csv(os.path.join(base_dir, 'data', 'forward_forecast_care_gbr.csv'), index=False)
    
    # Discharges Champion: Random Forest
    rf_disch = MLForecaster(model_type='rf', target_col='hhs_discharges').fit(feat_df, feature_cols)
    pt_d, l80_d, u80_d, l95_d, u95_d = rf_disch.predict_recursive(feat_df, horizon=30, last_date=last_date)
    forecast_disch_df = pd.DataFrame({
        'date': fc_dates,
        'forecast': np.round(pt_d, 1),
        'lower_80': np.round(np.maximum(0, l80_d), 1),
        'upper_80': np.round(u80_d, 1),
        'lower_95': np.round(np.maximum(0, l95_d), 1),
        'upper_95': np.round(u95_d, 1)
    })
    forecast_disch_df.to_csv(os.path.join(base_dir, 'data', 'forward_forecast_discharges_gbr.csv'), index=False)
    
    # Feature importances
    if gbr_care.feature_importances_ is not None:
        gbr_care.feature_importances_.to_csv(os.path.join(base_dir, 'data', 'feature_importances_care.csv'))
    if rf_disch.feature_importances_ is not None:
        rf_disch.feature_importances_.to_csv(os.path.join(base_dir, 'data', 'feature_importances_discharges.csv'))
        
    # Operational KPIs
    kpis = calculate_operational_kpis(df_continuous, forecast_care_df, capacity_limit=3000.0, surge_threshold=2700.0)
    print("\nOperational KPIs:", flush=True)
    for k, v in kpis.items():
        print(f"  {k}: {v}", flush=True)
        
    print("\nPipeline execution complete successfully!", flush=True)


if __name__ == '__main__':
    main()
