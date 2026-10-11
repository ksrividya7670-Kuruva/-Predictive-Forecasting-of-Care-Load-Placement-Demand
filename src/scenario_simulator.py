"""
Scenario Simulation and Policy Stress-Testing Module.
Simulates border apprehension surges, sponsor vetting bottlenecks, and policy interventions.
Translates care load forecasts into operational resource requirements (beds, caseworkers, clinicians).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


# Federal ORR Staffing Standard Ratios
STAFF_TO_CHILD_RATIO = 1 / 8.0     # 1 direct care youth care worker per 8 children
CASEWORKER_TO_CHILD_RATIO = 1 / 10.0 # 1 caseworker per 10 children
CLINICIAN_TO_CHILD_RATIO = 1 / 25.0  # 1 clinician/counselor per 25 children


def run_scenario_simulation(
    base_forecast_df: pd.DataFrame,
    initial_care_load: float,
    recent_intake: float,
    recent_discharge: float,
    horizon_days: int,
    intake_shock_pct: float = 0.0,
    discharge_shock_pct: float = 0.0,
    capacity_limit: float = 3000.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates dynamic stock-flow trajectory under counterfactual intake/discharge shocks:
    CareLoad_{t+1} = CareLoad_t + Intake_{eff} - Discharge_{eff}
    
    intake_shock_pct: percentage change in daily transfers into HHS (e.g., +50% for border surge)
    discharge_shock_pct: percentage change in daily sponsor placements (e.g., -30% for vetting slowdown)
    """
    dates = base_forecast_df['date'].iloc[:horizon_days]
    
    sim_intake = max(0.0, recent_intake * (1.0 + intake_shock_pct / 100.0))
    sim_discharge = max(1.0, recent_discharge * (1.0 + discharge_shock_pct / 100.0))
    net_daily_flow = sim_intake - sim_discharge
    
    sim_care_load = []
    current = initial_care_load
    
    for i in range(len(dates)):
        # Daily flow update with realistic weekend dips in discharges
        dow = dates.iloc[i].dayofweek
        dow_factor = 0.4 if dow in [5, 6] else 1.2
        effective_disch = max(0.0, sim_discharge * dow_factor)
        current = max(0.0, current + sim_intake - effective_disch)
        sim_care_load.append(current)
        
    sim_care_load = np.array(sim_care_load)
    
    # Calculate operational staffing and capacity
    peak_load = float(np.max(sim_care_load))
    final_load = float(sim_care_load[-1])
    bed_deficit = max(0.0, peak_load - capacity_limit)
    
    # Days until capacity breach
    breach_indices = np.where(sim_care_load >= capacity_limit)[0]
    days_to_breach = int(breach_indices[0] + 1) if len(breach_indices) > 0 else None
    
    # Staffing requirements at peak load
    direct_care_staff_needed = int(np.ceil(peak_load * STAFF_TO_CHILD_RATIO))
    caseworkers_needed = int(np.ceil(peak_load * CASEWORKER_TO_CHILD_RATIO))
    clinicians_needed = int(np.ceil(peak_load * CLINICIAN_TO_CHILD_RATIO))
    
    scenario_df = pd.DataFrame({
        'date': dates,
        'simulated_care_load': np.round(sim_care_load, 1),
        'simulated_daily_intake': round(sim_intake, 1),
        'simulated_daily_discharge': round(sim_discharge, 1),
        'capacity_limit': capacity_limit
    })
    
    impact_metrics = {
        'initial_care_load': initial_care_load,
        'peak_care_load': round(peak_load, 1),
        'final_care_load': round(final_load, 1),
        'net_load_change': round(final_load - initial_care_load, 1),
        'capacity_limit': capacity_limit,
        'bed_deficit': round(bed_deficit, 1),
        'capacity_breached': days_to_breach is not None,
        'days_to_breach': days_to_breach,
        'peak_bed_utilization_pct': round((peak_load / capacity_limit) * 100.0, 1),
        'direct_care_staff_needed': direct_care_staff_needed,
        'caseworkers_needed': caseworkers_needed,
        'clinicians_needed': clinicians_needed,
        'simulated_intake_rate': round(sim_intake, 1),
        'simulated_discharge_rate': round(sim_discharge, 1),
        'net_daily_pressure': round(net_daily_flow, 1)
    }
    
    return scenario_df, impact_metrics


def get_standard_policy_scenarios() -> Dict[str, Dict[str, float]]:
    """Returns predefined operational scenarios for stress testing."""
    return {
        'Status Quo (Baseline)': {
            'intake_shock_pct': 0.0,
            'discharge_shock_pct': 0.0,
            'description': 'Current intake and discharge operational velocity maintained without exogenous shocks.'
        },
        'Mild Border Surge (+25% Intake)': {
            'intake_shock_pct': 25.0,
            'discharge_shock_pct': 0.0,
            'description': 'Seasonal spring/summer border migration uptick or policy transition at southwestern border.'
        },
        'Severe Border Surge (+50% Intake)': {
            'intake_shock_pct': 50.0,
            'discharge_shock_pct': 0.0,
            'description': 'Major humanitarian corridor surge or policy transition causing sudden influx of transfers.'
        },
        'Sponsor Vetting Bottleneck (-30% Discharges)': {
            'intake_shock_pct': 0.0,
            'discharge_shock_pct': -30.0,
            'description': 'Administrative delays, enhanced biometric vetting, or caseworker shortages decelerating sponsor releases.'
        },
        'Compound Crisis (+40% Intake, -25% Discharges)': {
            'intake_shock_pct': 40.0,
            'discharge_shock_pct': -25.0,
            'description': 'Simultaneous border crossing surge combined with systemic placement bottleneck.'
        },
        'Fast-Track Placement Initiative (+35% Discharges)': {
            'intake_shock_pct': 0.0,
            'discharge_shock_pct': 35.0,
            'description': 'Emergency deployment of surge case coordinators and streamlined Category 1 sponsor releases.'
        }
    }
