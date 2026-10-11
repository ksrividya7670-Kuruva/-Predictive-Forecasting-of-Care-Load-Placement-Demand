"""
Data Processing and Time-Series Preparation Module
for UAC Care Load & Placement Demand Predictive Analytics.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw UAC daily reporting dataset, cleans formatted numbers,
    normalizes column names, and sorts chronologically.
    """
    df = pd.read_csv(file_path).dropna(how='all')
    
    # Standardize column names
    col_mapping = {
        'Date': 'date',
        'Children apprehended and placed in CBP custody*': 'cbp_apprehensions',
        'Children in CBP custody': 'cbp_in_custody',
        'Children transferred out of CBP custody': 'transfers_to_hhs',
        'Children in HHS Care': 'hhs_in_care',
        'Children discharged from HHS Care': 'hhs_discharges'
    }
    
    # Match columns flexibly
    new_cols = {}
    for col in df.columns:
        clean_c = col.strip()
        if 'date' in clean_c.lower():
            new_cols[col] = 'date'
        elif 'apprehended' in clean_c.lower():
            new_cols[col] = 'cbp_apprehensions'
        elif 'in cbp custody' in clean_c.lower():
            new_cols[col] = 'cbp_in_custody'
        elif 'transferred out of cbp' in clean_c.lower():
            new_cols[col] = 'transfers_to_hhs'
        elif 'in hhs care' in clean_c.lower():
            new_cols[col] = 'hhs_in_care'
        elif 'discharged' in clean_c.lower():
            new_cols[col] = 'hhs_discharges'
        else:
            new_cols[col] = col
            
    df = df.rename(columns=new_cols)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean numeric fields
    for col in ['cbp_apprehensions', 'cbp_in_custody', 'transfers_to_hhs', 'hhs_in_care', 'hhs_discharges']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_continuous_timeseries(df: pd.DataFrame, method: str = 'time') -> pd.DataFrame:
    """
    Ensures daily continuity of the time-series by creating a complete
    date range index and handling non-reporting days via interpolation.
    
    Stock variables (cbp_in_custody, hhs_in_care) represent state levels.
    Flow variables (cbp_apprehensions, transfers_to_hhs, hhs_discharges) represent daily rates.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=['date']).sort_values('date')
    df = df.set_index('date')
    
    # Full daily calendar range
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # Reindex
    full_df = df.reindex(full_idx)
    full_df.index.name = 'date'
    
    # Mark which rows were original reports vs interpolated
    full_df['is_interpolated'] = full_df['hhs_in_care'].isna()
    
    # Interpolate
    numeric_cols = ['cbp_apprehensions', 'cbp_in_custody', 'transfers_to_hhs', 'hhs_in_care', 'hhs_discharges']
    if method == 'time':
        full_df[numeric_cols] = full_df[numeric_cols].interpolate(method='time')
    elif method == 'linear':
        full_df[numeric_cols] = full_df[numeric_cols].interpolate(method='linear')
    elif method == 'spline':
        full_df[numeric_cols] = full_df[numeric_cols].interpolate(method='spline', order=2)
    elif method == 'ffill':
        full_df[numeric_cols] = full_df[numeric_cols].ffill()
    else:
        full_df[numeric_cols] = full_df[numeric_cols].interpolate(method='time')
        
    # Forward-fill / backward-fill any boundary edge cases
    full_df[numeric_cols] = full_df[numeric_cols].bfill().ffill()
    
    # Round integer-like counts for flows and stocks
    full_df['cbp_apprehensions'] = full_df['cbp_apprehensions'].round(1)
    full_df['cbp_in_custody'] = full_df['cbp_in_custody'].round(1)
    full_df['transfers_to_hhs'] = full_df['transfers_to_hhs'].round(1)
    full_df['hhs_in_care'] = full_df['hhs_in_care'].round(1)
    full_df['hhs_discharges'] = full_df['hhs_discharges'].round(1)
    
    # Flow balance & dynamic pressure metrics
    full_df['net_flow'] = full_df['transfers_to_hhs'] - full_df['discharges' if 'discharges' in full_df.columns else 'hhs_discharges']
    full_df['care_load_diff'] = full_df['hhs_in_care'].diff()
    
    # Flow-stock accounting discrepancy (reconciles operational reporting divergence)
    full_df['flow_accounting_discrepancy'] = full_df['care_load_diff'] - full_df['net_flow']
    
    return full_df.reset_index()


def perform_stl_decomposition(df: pd.DataFrame, target_col: str = 'hhs_in_care', period: int = 7):
    """
    Decomposes the time-series into Trend, Seasonality, and Residual components.
    """
    ts_df = df.copy().set_index('date')
    series = ts_df[target_col].dropna()
    
    # Use additive model for care load
    decomp = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
    
    decomp_df = pd.DataFrame({
        'date': series.index,
        'observed': decomp.observed,
        'trend': decomp.trend,
        'seasonal': decomp.seasonal,
        'resid': decomp.resid
    }).reset_index(drop=True)
    
    return decomp_df


def test_stationarity(series: pd.Series) -> dict:
    """
    Performs Augmented Dickey-Fuller (ADF) and KPSS tests on a series.
    """
    clean_s = series.dropna()
    adf_res = adfuller(clean_s)
    
    try:
        kpss_res = kpss(clean_s, regression='c', nlags="auto")
        kpss_stat = kpss_res[0]
        kpss_pval = kpss_res[1]
    except Exception:
        kpss_stat = np.nan
        kpss_pval = np.nan
        
    return {
        'adf_stat': adf_res[0],
        'adf_pvalue': adf_res[1],
        'adf_usedlag': adf_res[2],
        'is_stationary_adf': adf_res[1] < 0.05,
        'kpss_stat': kpss_stat,
        'kpss_pvalue': kpss_pval,
        'is_stationary_kpss': kpss_pval >= 0.05 if not np.isnan(kpss_pval) else False
    }


def compute_exploratory_stats(df: pd.DataFrame) -> dict:
    """
    Generates summary statistics, correlation matrices, and pipeline transfer ratios.
    """
    cols = ['cbp_apprehensions', 'cbp_in_custody', 'transfers_to_hhs', 'hhs_in_care', 'hhs_discharges', 'net_flow']
    present_cols = [c for c in cols if c in df.columns]
    
    summary = df[present_cols].describe().T
    corr = df[present_cols].corr()
    
    # Key pipeline metrics
    total_apprehensions = df['cbp_apprehensions'].sum()
    total_transfers = df['transfers_to_hhs'].sum()
    total_discharges = df['hhs_discharges'].sum()
    
    mean_daily_net_pressure = df['net_flow'].mean()
    std_daily_net_pressure = df['net_flow'].std()
    
    return {
        'summary': summary,
        'correlation': corr,
        'total_apprehensions': total_apprehensions,
        'total_transfers': total_transfers,
        'total_discharges': total_discharges,
        'mean_net_pressure': mean_daily_net_pressure,
        'std_net_pressure': std_daily_net_pressure,
        'peak_care_load': df['hhs_in_care'].max(),
        'peak_care_load_date': df.loc[df['hhs_in_care'].idxmax(), 'date'],
        'trough_care_load': df['hhs_in_care'].min(),
        'trough_care_load_date': df.loc[df['hhs_in_care'].idxmin(), 'date']
    }
