# Predictive Forecasting of Care Load & Placement Demand

### Unaccompanied Alien Children (UAC) Program — HHS & CBP Decision Intelligence System

This repository provides an enterprise-grade predictive modeling architecture and interactive web dashboard for forecasting daily care load and sponsor placement demand for the UAC Program, operated by the U.S. Department of Health and Human Services (HHS / ORR) and U.S. Customs and Border Protection (CBP).

---

## 📁 Project Structure

```text
uac_care_load_forecasting/
├── app.py                      # Interactive Streamlit Web Application
├── run_pipeline.py             # End-to-end data processing, feature engineering, and backtesting runner
├── README.md                   # Project overview & execution instructions
├── data/                       # Datasets, benchmarks, and model artifacts
│   ├── uac_daily_raw.csv               # Raw historical reporting data (2023 - 2025)
│   ├── uac_daily_continuous.csv        # Continuous daily reindexed dataset with time interpolation
│   ├── uac_features.csv                # 61 engineered predictive features
│   ├── benchmark_summary_care.csv      # Walk-forward backtest metrics (Care Load)
│   ├── benchmark_horizons_care.csv     # Horizon breakdown (7d, 14d, 30d) (Care Load)
│   ├── benchmark_summary_discharges.csv# Walk-forward backtest metrics (Discharges)
│   ├── benchmark_horizons_discharges.csv# Horizon breakdown (Discharges)
│   ├── stl_decomposition_care.csv      # STL Decomposition components
│   ├── forward_forecast_care_gbr.csv   # 30-day forward forecasts & confidence intervals
│   └── feature_importances_care.csv    # Tree feature importances
├── src/                        # Modular source code
│   ├── data_processor.py       # Data cleaning, interpolation, ADF/KPSS tests, STL decomposition
│   ├── feature_engineering.py  # Autoregressive lags, rolling stats, flow pressure, calendar encodings
│   ├── models.py               # Naïve, Moving Average, Holt-Winters, SARIMA, RF, Gradient Boosting
│   ├── evaluation.py           # Walk-forward backtesting, metrics (MAE, RMSE, MAPE, sMAPE, R2), KPIs
│   └── scenario_simulator.py   # What-if policy stress tests and staffing translations
└── deliverables/               # Final client & stakeholder deliverables
    ├── RESEARCH_PAPER.md       # Peer-reviewed academic & operational research monograph
    └── EXECUTIVE_SUMMARY.md    # High-level strategic briefing & 4-tier operational playbook
```

---

## 🚀 How to Run the Application

### 1. Launch the Streamlit Dashboard
To launch the interactive dashboard, open a terminal in this directory and run:

```bash
streamlit run app.py
```

The application will launch on `http://localhost:8501` featuring:
- **Module 1:** Future Care Load Forecast Chart with 80% & 95% Confidence Intervals
- **Module 2:** Placement Demand Forecast Panel with Exit Velocity Tracking
- **Module 3:** Multi-Horizon Model Selection & Comparative Benchmark
- **Module 4:** Early Warning Matrix & Bed Capacity Stress Gauges
- **Module 5:** "What-If" Policy Stress Simulator (Border Surges, Vetting Bottlenecks, Staffing Needs)
- **Module 6:** System Dynamics, Cross-Correlation Lag Analyses, and STL Decomposition
- **Module 7:** Historical Data Browser & CSV Export Feeds

### 2. Re-run or Retrain the Pipeline
To re-run the end-to-end data processing, feature engineering, and model validation pipeline:

```bash
python run_pipeline.py
```

---

## 📊 Summary of Benchmark Performance

Evaluated via strict temporal expanding-window walk-forward backtesting (4 folds, 30-day horizons):

| Model | Care Load MAE | RMSE | MAPE (%) | Forecast Accuracy (%) | Stability Index |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SARIMA(1,1,1)(1,0,1)$_7$** | **40.56** | **49.46** | **1.83%** | **98.17%** | 23.77 |
| **Gradient Boosting Regressor** | **46.46** | **57.19** | **2.02%** | **97.98%** | **22.74** |
| **Random Forest Regressor** | 55.79 | 67.12 | 2.46% | 97.54% | 23.66 |
| **Naïve Persistence Baseline** | 57.16 | 68.61 | 2.52% | 97.48% | 17.36 |
| **7-Day Moving Average Baseline**| 70.44 | 80.34 | 3.11% | 96.89% | 14.86 |
| **Holt-Winters Exponential** | 177.76 | 200.05 | 8.10% | 91.90% | 126.91 |

---

## 📑 Deliverables Included
- **[Research Paper](file:///C:/Users/user/.gemini/antigravity/scratch/uac_care_load_forecasting/deliverables/RESEARCH_PAPER.md)**: Full 12-section technical study with literature context, mathematical formulations, stationarity proofs, empirical backtest tables, and policy implications.
- **[Executive Summary](file:///C:/Users/user/.gemini/antigravity/scratch/uac_care_load_forecasting/deliverables/EXECUTIVE_SUMMARY.md)**: High-level briefing for federal agency executives featuring the 4-Tier Operational Readiness Matrix and 30-60-90 day deployment roadmap.
- **[Interactive Streamlit App](file:///C:/Users/user/.gemini/antigravity/scratch/uac_care_load_forecasting/app.py)**: Web application meeting all dashboard and analytical requirements.
