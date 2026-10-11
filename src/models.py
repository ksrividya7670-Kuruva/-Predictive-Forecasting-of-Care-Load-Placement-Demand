"""
Forecasting Models Architecture for UAC Care Load & Placement Demand.
Implements:
1. Baseline: Naïve Persistence & Moving Average
2. Statistical: Holt-Winters Exponential Smoothing & SARIMA
3. Machine Learning: Random Forest & Gradient Boosting Regressors
Includes multi-horizon multi-step forecasting and empirical/conformal confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ForecastResult:
    """Encapsulates forecast point predictions and uncertainty intervals."""
    def __init__(self, model_name: str, target_name: str, dates: pd.DatetimeIndex,
                 point_forecast: np.ndarray, lower_80: np.ndarray, upper_80: np.ndarray,
                 lower_95: np.ndarray, upper_95: np.ndarray, metadata: Optional[Dict] = None):
        self.model_name = model_name
        self.target_name = target_name
        self.dates = dates
        self.point_forecast = point_forecast
        self.lower_80 = lower_80
        self.upper_80 = upper_80
        self.lower_95 = lower_95
        self.upper_95 = upper_95
        self.metadata = metadata or {}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': self.dates,
            'forecast': np.round(self.point_forecast, 1),
            'lower_80': np.round(np.maximum(0, self.lower_80), 1),
            'upper_80': np.round(self.upper_80, 1),
            'lower_95': np.round(np.maximum(0, self.lower_95), 1),
            'upper_95': np.round(self.upper_95, 1)
        })


class NaivePersistenceForecaster:
    """Naïve persistence baseline: assumes future values equal the last observed value."""
    def __init__(self):
        self.last_value = None
        self.residual_std = None

    def fit(self, y: pd.Series):
        self.last_value = float(y.iloc[-1])
        residuals = y.diff().dropna()
        self.residual_std = float(residuals.std()) if len(residuals) > 1 else 10.0
        return self

    def predict(self, horizon: int, last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point = np.full(horizon, self.last_value)
        step_factor = np.sqrt(np.arange(1, horizon + 1))
        se = self.residual_std * step_factor
        l80 = point - 1.282 * se
        u80 = point + 1.282 * se
        l95 = point - 1.960 * se
        u95 = point + 1.960 * se
        return point, l80, u80, l95, u95


class MovingAverageForecaster:
    """Moving Average baseline: persists the mean of the last W observations."""
    def __init__(self, window: int = 7):
        self.window = window
        self.ma_value = None
        self.residual_std = None

    def fit(self, y: pd.Series):
        self.ma_value = float(y.iloc[-self.window:].mean())
        residuals = y - y.rolling(self.window).mean()
        self.residual_std = float(residuals.dropna().std()) if len(residuals.dropna()) > 1 else 15.0
        return self

    def predict(self, horizon: int, last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point = np.full(horizon, self.ma_value)
        step_factor = np.sqrt(np.arange(1, horizon + 1))
        se = self.residual_std * step_factor
        l80 = point - 1.282 * se
        u80 = point + 1.282 * se
        l95 = point - 1.960 * se
        u95 = point + 1.960 * se
        return point, l80, u80, l95, u95


class HoltWintersForecaster:
    """Holt's Linear / Holt-Winters Exponential Smoothing statistical model."""
    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted = None
        self.residual_std = None

    def fit(self, y: pd.Series):
        try:
            if len(y) >= 2 * self.seasonal_periods:
                self.model = ExponentialSmoothing(
                    y,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.seasonal_periods,
                    initialization_method='estimated'
                )
            else:
                self.model = ExponentialSmoothing(y, trend='add', initialization_method='estimated')
            self.fitted = self.model.fit(optimized=True)
            residuals = y - self.fitted.fittedvalues
            self.residual_std = float(residuals.std())
        except Exception:
            self.model = ExponentialSmoothing(y, trend=None)
            self.fitted = self.model.fit()
            self.residual_std = float((y - self.fitted.fittedvalues).std())
        return self

    def predict(self, horizon: int, last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point = np.array(self.fitted.forecast(horizon))
        step_factor = np.sqrt(np.arange(1, horizon + 1))
        se = (self.residual_std if self.residual_std else 20.0) * step_factor
        l80 = point - 1.282 * se
        u80 = point + 1.282 * se
        l95 = point - 1.960 * se
        u95 = point + 1.960 * se
        return point, l80, u80, l95, u95


class SARIMAForecaster:
    """SARIMA (Seasonal Autoregressive Integrated Moving Average) statistical model."""
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted = None
        self.residual_std = None

    def fit(self, y: pd.Series):
        try:
            # Use concise recent history for high-speed estimation if long
            y_fit = y.iloc[-300:] if len(y) > 300 else y
            model = SARIMAX(
                y_fit,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted = model.fit(disp=False, maxiter=20, method='lbfgs')
            self.residual_std = float(self.fitted.resid.std())
        except Exception:
            model = SARIMAX(y.iloc[-150:], order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
            self.fitted = model.fit(disp=False, maxiter=15)
            self.residual_std = float(self.fitted.resid.std())
        return self

    def predict(self, horizon: int, last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            fc = self.fitted.get_forecast(steps=horizon)
            point = np.array(fc.predicted_mean)
            ci80 = fc.conf_int(alpha=0.20)
            ci95 = fc.conf_int(alpha=0.05)
            l80 = np.array(ci80.iloc[:, 0])
            u80 = np.array(ci80.iloc[:, 1])
            l95 = np.array(ci95.iloc[:, 0])
            u95 = np.array(ci95.iloc[:, 1])
        except Exception:
            point = np.array(self.fitted.forecast(steps=horizon))
            se = (self.residual_std if self.residual_std else 25.0) * np.sqrt(np.arange(1, horizon + 1))
            l80 = point - 1.282 * se
            u80 = point + 1.282 * se
            l95 = point - 1.960 * se
            u95 = point + 1.960 * se
        return point, l80, u80, l95, u95


class MLForecaster:
    """
    Machine Learning Forecaster supporting Random Forest and Gradient Boosting.
    Implements fast recursive multi-step forecasting with tail feature reconstruction.
    """
    def __init__(self, model_type: str = 'rf', target_col: str = 'hhs_in_care', random_state: int = 42):
        self.model_type = model_type
        self.target_col = target_col
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.residual_std = None
        self.feature_importances_ = None

        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(
                n_estimators=80,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, df_features: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        clean_df = df_features.dropna(subset=self.feature_cols + [self.target_col]).copy()
        
        X = clean_df[self.feature_cols]
        y = clean_df[self.target_col]
        
        self.model.fit(X, y)
        
        preds = self.model.predict(X)
        self.residual_std = float(np.std(y - preds))
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
            
        return self

    def predict_recursive(self, df_features: pd.DataFrame, horizon: int, last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from src.feature_engineering import build_forecasting_features
        
        core_cols = ['date', 'cbp_apprehensions', 'cbp_in_custody', 'transfers_to_hhs', 'hhs_in_care', 'hhs_discharges', 'net_flow']
        # Keep only essential columns and tail to make feature generation lightning-fast
        current_data = df_features[[c for c in core_cols if c in df_features.columns]].copy()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        preds = []
        for step_date in forecast_dates:
            # Slicing the tail (last 45 rows) is sufficient for all 21-day lags and 14-day rolling windows
            tail_data = current_data.iloc[-45:].copy()
            feat_df = build_forecasting_features(tail_data, target_col=self.target_col)
            last_row = feat_df.iloc[[-1]][self.feature_cols]
            
            step_pred = float(self.model.predict(last_row)[0])
            preds.append(step_pred)
            
            # Form next row
            new_row = {col: current_data.iloc[-1][col] for col in current_data.columns}
            new_row['date'] = step_date
            new_row[self.target_col] = step_pred
            
            if self.target_col == 'hhs_in_care':
                new_row['transfers_to_hhs'] = current_data['transfers_to_hhs'].iloc[-7:].mean()
                new_row['hhs_discharges'] = current_data['hhs_discharges'].iloc[-7:].mean()
                new_row['cbp_apprehensions'] = current_data['cbp_apprehensions'].iloc[-7:].mean()
                new_row['cbp_in_custody'] = current_data['cbp_in_custody'].iloc[-1]
                new_row['net_flow'] = new_row['transfers_to_hhs'] - new_row['hhs_discharges']
            elif self.target_col == 'hhs_discharges':
                new_row['hhs_discharges'] = max(0.0, step_pred)
                new_row['transfers_to_hhs'] = current_data['transfers_to_hhs'].iloc[-7:].mean()
                new_row['net_flow'] = new_row['transfers_to_hhs'] - new_row['hhs_discharges']
                new_row['hhs_in_care'] = max(0.0, current_data['hhs_in_care'].iloc[-1] + new_row['net_flow'])
            
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            
        point = np.array(preds)
        step_factor = np.sqrt(np.arange(1, horizon + 1))
        se = (self.residual_std if self.residual_std else 18.0) * step_factor
        l80 = point - 1.282 * se
        u80 = point + 1.282 * se
        l95 = point - 1.960 * se
        u95 = point + 1.960 * se
        return point, l80, u80, l95, u95
