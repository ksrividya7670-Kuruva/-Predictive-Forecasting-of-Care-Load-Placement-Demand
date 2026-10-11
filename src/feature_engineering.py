"""
Feature Engineering Module for UAC Care Load & Placement Demand Forecasting.
Constructs lag features, rolling statistics, flow-based signals, and calendar effects.
"""

import pandas as pd
import numpy as np


def build_forecasting_features(df: pd.DataFrame, target_col: str = 'hhs_in_care') -> pd.DataFrame:
    """
    Generates rich predictive features for both statistical feature-based models
    and machine learning regressors (Random Forest, Gradient Boosting).
    
    Includes:
    1. Lag features (t-1, t-2, t-3, t-7, t-14, t-21)
    2. Rolling window statistics (7d, 14d mean, std, min, max)
    3. Flow-based pipeline signals (net pressure, cumulative flow, custody ratio)
    4. Calendar & temporal effects (day of week, month, weekend, cyclical encoding)
    5. Velocity & acceleration indicators
    """
    data = df.copy().sort_values('date').reset_index(drop=True)
    
    # 1. Flow Signals
    if 'net_flow' not in data.columns:
        data['net_flow'] = data['transfers_to_hhs'] - data['hhs_discharges']
        
    data['flow_ratio'] = data['transfers_to_hhs'] / (data['hhs_discharges'] + 1.0)
    data['cbp_turnover_rate'] = data['transfers_to_hhs'] / (data['cbp_in_custody'] + 1.0)
    data['hhs_discharge_rate'] = data['hhs_discharges'] / (data['hhs_in_care'] + 1.0)
    
    # Cumulative flow pressure
    data['net_flow_cum7'] = data['net_flow'].rolling(window=7, min_periods=1).sum()
    data['net_flow_cum14'] = data['net_flow'].rolling(window=14, min_periods=1).sum()
    
    # 2. Target and Flow Lag Features (t-1, t-2, t-3, t-7, t-14, t-21)
    lags = [1, 2, 3, 7, 14, 21]
    
    # Lags for HHS In Care
    for lag in lags:
        data[f'hhs_in_care_lag_{lag}'] = data['hhs_in_care'].shift(lag)
        
    # Lags for HHS Discharges
    for lag in [1, 2, 3, 7, 14]:
        data[f'hhs_discharges_lag_{lag}'] = data['hhs_discharges'].shift(lag)
        
    # Lags for Transfers to HHS
    for lag in [1, 2, 3, 7, 14]:
        data[f'transfers_lag_{lag}'] = data['transfers_to_hhs'].shift(lag)
        
    # Lags for CBP Apprehensions & Custody (Leading upstream pipeline indicators!)
    for lag in [1, 2, 3, 7]:
        data[f'cbp_apprehensions_lag_{lag}'] = data['cbp_apprehensions'].shift(lag)
        data[f'cbp_in_custody_lag_{lag}'] = data['cbp_in_custody'].shift(lag)
        
    # Lags for Net Flow
    for lag in [1, 2, 7, 14]:
        data[f'net_flow_lag_{lag}'] = data['net_flow'].shift(lag)
        
    # 3. Rolling Averages & Volatility (7-day and 14-day rolling mean and variance/std)
    for w in [7, 14]:
        # Care load rolling stats (using lag 1 to avoid lookahead leakage)
        care_lag1 = data['hhs_in_care'].shift(1)
        data[f'care_roll_mean_{w}'] = care_lag1.rolling(window=w, min_periods=1).mean()
        data[f'care_roll_std_{w}'] = care_lag1.rolling(window=w, min_periods=1).std().fillna(0)
        data[f'care_roll_var_{w}'] = care_lag1.rolling(window=w, min_periods=1).var().fillna(0)
        
        # Discharges rolling stats
        dis_lag1 = data['hhs_discharges'].shift(1)
        data[f'discharges_roll_mean_{w}'] = dis_lag1.rolling(window=w, min_periods=1).mean()
        data[f'discharges_roll_std_{w}'] = dis_lag1.rolling(window=w, min_periods=1).std().fillna(0)
        
        # Transfers rolling stats
        tr_lag1 = data['transfers_to_hhs'].shift(1)
        data[f'transfers_roll_mean_{w}'] = tr_lag1.rolling(window=w, min_periods=1).mean()
        
        # Apprehensions rolling stats (upstream pipeline pressure)
        app_lag1 = data['cbp_apprehensions'].shift(1)
        data[f'apprehensions_roll_mean_{w}'] = app_lag1.rolling(window=w, min_periods=1).mean()
        data[f'apprehensions_roll_std_{w}'] = app_lag1.rolling(window=w, min_periods=1).std().fillna(0)
        
    # 4. Momentum, Velocity & Rate of Change
    data['care_velocity_7d'] = (data['hhs_in_care'].shift(1) - data['hhs_in_care'].shift(8)) / 7.0
    data['care_velocity_14d'] = (data['hhs_in_care'].shift(1) - data['hhs_in_care'].shift(15)) / 14.0
    data['apprehension_velocity_7d'] = (data['cbp_apprehensions'].shift(1) - data['cbp_apprehensions'].shift(8)) / 7.0
    
    # Upstream surge z-score (Lead warning signal)
    mean_app = data['cbp_apprehensions'].shift(1).rolling(14, min_periods=1).mean()
    std_app = data['cbp_apprehensions'].shift(1).rolling(14, min_periods=1).std().replace(0, 1).fillna(1)
    data['apprehension_surge_zscore'] = (data['cbp_apprehensions'].shift(1) - mean_app) / std_app
    
    # 5. Calendar Effects & Cyclical Encodings
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['day_of_month'] = data['date'].dt.day
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical day of week encoding (period = 7)
    data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7.0)
    data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7.0)
    
    # Cyclical month encoding (period = 12)
    data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12.0)
    
    return data


def get_feature_columns(df: pd.DataFrame, target_col: str = 'hhs_in_care') -> list:
    """
    Returns list of engineered feature column names, excluding non-feature metadata,
    raw unshifted targets, and dates.
    """
    excluded = {
        'date', 'cbp_apprehensions', 'cbp_in_custody', 'transfers_to_hhs',
        'hhs_in_care', 'hhs_discharges', 'net_flow', 'is_interpolated',
        'care_load_diff', 'flow_accounting_discrepancy'
    }
    
    feature_cols = [c for c in df.columns if c not in excluded and not c.startswith('target_')]
    return feature_cols
