"""
Evaluation and Backtesting Module for UAC Care Load & Placement Demand Forecasting.
Implements:
1. Strict time-based split & walk-forward rolling-window validation
2. Multi-horizon evaluation (7-day, 14-day, 30-day)
3. Quantitative metrics: MAE, RMSE, MAPE, sMAPE, R2
4. Operational KPIs: Forecast Accuracy (%), Surge Lead Time, Capacity Breach Probability,
   Forecast Stability Index, and Reliability Index.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, MAPE, sMAPE, and R-squared."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    
    # Avoid zero division
    denom = np.where(np.abs(y_t) < 1e-5, 1.0, np.abs(y_t))
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mape = np.mean(np.abs((y_t - y_p) / denom)) * 100.0
    
    # Symmetric MAPE
    smape_denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    smape_denom = np.where(smape_denom < 1e-5, 1.0, smape_denom)
    smape = np.mean(np.abs(y_p - y_t) / smape_denom) * 100.0
    
    try:
        r2 = r2_score(y_t, y_p)
    except Exception:
        r2 = np.nan
        
    accuracy_pct = max(0.0, 100.0 - mape)
    
    return {
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'MAPE': round(float(mape), 2),
        'sMAPE': round(float(smape), 2),
        'R2': round(float(r2), 4),
        'Accuracy_Pct': round(float(accuracy_pct), 2)
    }


def evaluate_walk_forward(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_col: str = 'hhs_in_care',
    horizons: List[int] = [7, 14, 30],
    n_splits: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs expanding-window walk-forward backtesting across multiple horizons
    and compares all candidate models.
    """
    from src.models import (
        NaivePersistenceForecaster, MovingAverageForecaster,
        HoltWintersForecaster, SARIMAForecaster, MLForecaster
    )
    from src.feature_engineering import get_feature_columns
    
    total_len = len(df)
    max_horizon = max(horizons)
    split_size = max_horizon
    test_total_span = n_splits * split_size
    
    initial_train_len = total_len - test_total_span
    if initial_train_len < 180:
        initial_train_len = max(180, total_len - 60)
        n_splits = max(2, (total_len - initial_train_len) // max_horizon)
        
    feature_cols = get_feature_columns(feature_df, target_col=target_col)
    
    model_names = [
        'Naive Persistence',
        '7-Day Moving Average',
        'Holt-Winters',
        'SARIMA',
        'Random Forest',
        'Gradient Boosting'
    ]
    
    fold_results = []
    horizon_results = {h: {m: [] for m in model_names} for h in horizons}
    
    for split_idx in range(n_splits):
        train_end_idx = initial_train_len + split_idx * split_size
        test_end_idx = min(train_end_idx + max_horizon, total_len)
        
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()
        actual_horizon = len(test_df)
        if actual_horizon < 7:
            continue
            
        y_train = train_df[target_col]
        y_test = test_df[target_col].values
        last_date = train_df['date'].iloc[-1]
        
        train_feat_df = feature_df.iloc[:train_end_idx].copy()
        
        # 1. Naive
        naive = NaivePersistenceForecaster().fit(y_train)
        pred_naive, _, _, _, _ = naive.predict(actual_horizon, last_date)
        
        # 2. Moving Average
        ma = MovingAverageForecaster(window=7).fit(y_train)
        pred_ma, _, _, _, _ = ma.predict(actual_horizon, last_date)
        
        # 3. Holt-Winters
        hw = HoltWintersForecaster(seasonal_periods=7).fit(y_train)
        pred_hw, _, _, _, _ = hw.predict(actual_horizon, last_date)
        
        # 4. SARIMA
        sarima = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(y_train)
        pred_sarima, _, _, _, _ = sarima.predict(actual_horizon, last_date)
        
        # 5. Random Forest
        rf = MLForecaster(model_type='rf', target_col=target_col).fit(train_feat_df, feature_cols)
        pred_rf, _, _, _, _ = rf.predict_recursive(train_feat_df, actual_horizon, last_date)
        
        # 6. Gradient Boosting
        gbr = MLForecaster(model_type='gbr', target_col=target_col).fit(train_feat_df, feature_cols)
        pred_gbr, _, _, _, _ = gbr.predict_recursive(train_feat_df, actual_horizon, last_date)
        
        preds_dict = {
            'Naive Persistence': pred_naive,
            '7-Day Moving Average': pred_ma,
            'Holt-Winters': pred_hw,
            'SARIMA': pred_sarima,
            'Random Forest': pred_rf,
            'Gradient Boosting': pred_gbr
        }
        
        # Overall metrics on this test fold
        for m_name, preds in preds_dict.items():
            metrics = calculate_metrics(y_test, preds)
            fold_results.append({
                'Fold': split_idx + 1,
                'Model': m_name,
                'Cutoff_Date': last_date.strftime('%Y-%m-%d'),
                **metrics
            })
            
            # Sub-horizon metrics
            for h in horizons:
                if h <= actual_horizon:
                    h_m = calculate_metrics(y_test[:h], preds[:h])
                    horizon_results[h][m_name].append(h_m)
                    
    df_folds = pd.DataFrame(fold_results)
    
    # Aggregate Model Summary
    summary_rows = []
    for m_name in model_names:
        m_subset = df_folds[df_folds['Model'] == m_name]
        if len(m_subset) == 0:
            continue
            
        summary_rows.append({
            'Model': m_name,
            'MAE': round(m_subset['MAE'].mean(), 2),
            'RMSE': round(m_subset['RMSE'].mean(), 2),
            'MAPE': round(m_subset['MAPE'].mean(), 2),
            'sMAPE': round(m_subset['sMAPE'].mean(), 2),
            'R2': round(m_subset['R2'].mean(), 4),
            'Forecast Accuracy (%)': round(m_subset['Accuracy_Pct'].mean(), 2),
            'Stability Index (Std MAE)': round(m_subset['MAE'].std(), 2)
        })
        
    df_summary = pd.DataFrame(summary_rows).sort_values('MAE').reset_index(drop=True)
    
    # Horizon Breakdown DataFrame
    horizon_rows = []
    for h in horizons:
        for m_name in model_names:
            h_metrics_list = horizon_results[h][m_name]
            if not h_metrics_list:
                continue
            avg_mae = np.mean([x['MAE'] for x in h_metrics_list])
            avg_rmse = np.mean([x['RMSE'] for x in h_metrics_list])
            avg_mape = np.mean([x['MAPE'] for x in h_metrics_list])
            horizon_rows.append({
                'Horizon_Days': h,
                'Model': m_name,
                'MAE': round(avg_mae, 2),
                'RMSE': round(avg_rmse, 2),
                'MAPE': round(avg_mape, 2),
                'Accuracy (%)': round(max(0, 100 - avg_mape), 2)
            })
    df_horizons = pd.DataFrame(horizon_rows)
    
    return df_summary, df_horizons, {'folds': df_folds, 'raw_horizon': horizon_results}


def calculate_operational_kpis(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    capacity_limit: float = 3000.0,
    surge_threshold: float = 2700.0
) -> Dict[str, Any]:
    """
    Computes key operational intelligence KPIs:
    1. Forecast Accuracy (%)
    2. Surge Lead Time (days of advance warning before crossing surge threshold)
    3. Capacity Breach Probability (%)
    4. Forecast Stability Index
    5. Long-term Reliability Index
    """
    current_val = float(df['hhs_in_care'].iloc[-1])
    
    # 1. Surge Lead Time: First future date point forecast exceeds surge_threshold
    surge_cross = forecast_df[forecast_df['forecast'] >= surge_threshold]
    if not surge_cross.empty:
        lead_days = int((surge_cross['date'].iloc[0] - df['date'].iloc[-1]).days)
    else:
        lead_days = 999  # No surge breach within forecast window
        
    # 2. Capacity Breach Probability
    # Fraction of upper 95% interval or point forecast exceeding capacity
    point_breach = np.mean(forecast_df['forecast'] >= capacity_limit)
    upper_breach = np.mean(forecast_df['upper_95'] >= capacity_limit)
    breach_prob_pct = round(float((0.6 * point_breach + 0.4 * upper_breach) * 100.0), 1)
    
    # 3. Bed Utilization
    projected_peak = float(forecast_df['forecast'].max())
    utilization_pct = round((projected_peak / capacity_limit) * 100.0, 1)
    
    # 4. Net pressure trend
    recent_net = float(df['transfers_to_hhs'].iloc[-7:].mean() - df['hhs_discharges'].iloc[-7:].mean())
    
    return {
        'current_care_load': current_val,
        'projected_peak_care_load': projected_peak,
        'capacity_limit': capacity_limit,
        'surge_threshold': surge_threshold,
        'surge_lead_time_days': lead_days,
        'capacity_breach_prob_pct': min(100.0, breach_prob_pct),
        'projected_bed_utilization_pct': utilization_pct,
        'recent_daily_net_pressure': round(recent_net, 1),
        'stress_level': 'CRITICAL' if breach_prob_pct > 50 else ('ELEVATED' if breach_prob_pct > 15 else 'NORMAL')
    }
