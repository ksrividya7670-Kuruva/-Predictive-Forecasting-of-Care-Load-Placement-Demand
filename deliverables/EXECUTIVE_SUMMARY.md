# EXECUTIVE BRIEFING: Predictive Forecasting of Care Load & Placement Demand

**To:** Leadership & Operational Directors, U.S. Department of Health and Human Services (HHS), Administration for Children and Families (ACF), Office of Refugee Resettlement (ORR), and U.S. Customs and Border Protection (CBP)  
**From:** Predictive Analytics & Child-Welfare Operations Taskforce  
**Subject:** Implementation of Predictive Forecasting Architecture for the Unaccompanied Alien Children (UAC) Program  
**Date:** October 2026  
**Security / Operational Classification:** Internal Government Working Paper / Operational Foresight  

---

## 1. Executive Summary & Strategic Context

The UAC Program operates under stringent legal and humanitarian mandates, most notably the Trafficking Victims Protection Reauthorization Act (TVPRA) requirement that unaccompanied children encountered by CBP be transferred to HHS/ORR custody within **72 hours**. Historically, the program has operated in a **reactive posture**, relying on descriptive dashboards that report past occupancy. 

In a volatile operational environment shaped by seasonal migration waves and rapid policy shifts, reactive resource allocation leads to severe operational bottlenecks:
- **Shelter Overcrowding Risk:** Shelters rapidly breach 90%+ occupancy, forcing emergency activation of costly Influx Care Facilities (ICFs).
- **Statutory Transfer Delays:** CBP holding stations face overflow when HHS intake slows down, compromising 72-hour legal compliance.
- **Child Welfare Staff Burnout:** Unplanned surges distort staff-to-child ratios, prolonging length of stay (LOS) and clinical backlogs.

To solve this, we have developed and validated an enterprise-grade **Predictive Care Load & Placement Demand Forecasting Engine**. Operating across 7-, 14-, and 30-day planning horizons, this system provides HHS leadership with actionable advance warning to scale shelter bed capacity, dynamically allocate caseworkers, and de-risk the care pipeline.

---

## 2. Key Empirical Findings

Our quantitative evaluation of 1,075 days of empirical reporting (January 2023 to December 2025) across 720 federal reporting cycles establishes four operational realities:

1. **Upstream Pipeline Signals Provide 24–48 Hours of Pure Lead Time:**
   Border apprehensions and CBP custody levels exhibit a **0.9256 correlation at Lag +1** with HHS intake transfers. Tracking CBP border encounters provides HHS with an automated 24- to 48-hour advance signal of incoming shelter admissions.
2. **Systemic Integration and Differencing ($I(1)$):**
   Active care load and daily flows are non-stationary in levels but strictly stationary under first differencing ($p < 0.0001$). Models that ignore differencing fail to handle regime shifts, whereas properly differenced models maintain **over 98% accuracy**.
3. **Weekly Placement Deceleration:**
   Discharges to vetted sponsors plunge sharply on weekends (<10/day vs. >180/day on peak weekdays) due to administrative and sponsor travel schedules, creating structural intra-week care load accumulation.
4. **Current Operational Posture (December 2025):**
   Active care load currently sits at **2,484 children**, well below the historic peak of 11,516 (December 2023). Current net daily flow is roughly neutral to slightly contracting (-2.4 children/day), representing a **Tier 1 (Normal)** operating posture against a 3,000-bed baseline capacity.

---

## 3. Champion Model Architecture & Benchmark Performance

We conducted rigorous multi-horizon walk-forward validation across baseline, statistical, and machine learning models under strict temporal backtesting (no data leakage):

| Model Paradigm | Architecture | Care Load MAE | Care Load Accuracy (%) | 30-Day Stability | Operational Recommendation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Statistical Champion** | **SARIMA(1,1,1)(1,0,1)$_7$** | **40.56** | **98.17%** | **High (23.77)** | **Primary Automated Engine:** Delivers highest accuracy, best calibration, and optimal 7-day administrative rhythm tracking. |
| **Machine Learning Champion** | **Gradient Boosting Regressor** | **46.46** | **97.98%** | **High (22.74)** | **Scenario & Pipeline Engine:** Leverages non-linear upstream CBP features to simulate policy and influx shocks. |
| Machine Learning | Random Forest | 55.79 | 97.54% | Moderate (23.66) | Robust secondary ensemble component. |
| Baseline | Naïve Persistence | 57.16 | 97.48% | Moderate (17.36) | Strong short-term (1–3 day) reference check. |
| Baseline | 7-Day Moving Average | 70.44 | 96.89% | High (14.86) | Effective for placement demand smoothing. |
| Statistical | Holt-Winters | 177.76 | 91.90% | Poor (126.91) | Unsuitable: over-extrapolates linear trends. |

**Operational Takeaway:** **SARIMA** is selected as the primary operational engine for automated daily care load forecasting, while **Gradient Boosting** serves as the decision engine for "What-If" policy stress tests and multi-feature scenario planning.

---

## 4. Operational Playbook: 4-Tier Early Warning Readiness Matrix

We translate model predictions into a standardized 4-Tier Early Warning Protocol to replace subjective decision-making with objective, data-driven operational triggers:

```
┌────────────────────────────────────────────────────────────────────────┐
│             4-TIER OPERATIONAL READINESS ACTION MATRIX                 │
└────────────────────────────────────────────────────────────────────────┘

 🟢 TIER 1: NORMAL (< 80% Capacity | Surge Lead Time > 30 Days)
    • Routine operations across state-licensed shelter networks.
    • Standard 1:8 youth care worker and 1:10 caseworker staffing ratios.
    • Standard Category 1, 2, and 3 sponsor background checks.

 🟡 TIER 2: ELEVATED (80% - 89% Capacity | Surge Lead Time 15 - 30 Days)
    • Issue Pre-Alert to contracted standby shelter facilities.
    • Fast-track Category 1 (parent/legal guardian) sponsor releases.
    • Authorize voluntary overtime for case managers and clinicians.
    • Daily inter-agency synchronization between CBP and ORR intake teams.

 🟠 TIER 3: SURGE READINESS (90% - 97% Capacity | Surge Lead Time 7 - 14 Days)
    • Place emergency Influx Care Facilities (ICFs) on warm standby.
    • Deploy Rapid Placement Teams to top 10 highest-occupancy shelters.
    • Streamline sponsor fingerprint appointments and home-study processing.
    • Coordinate standby medical personnel through federal health partners.

 🔴 TIER 4: CRITICAL EMERGENCY (≥ 98% Capacity | Surge Lead Time < 7 Days)
    • Activate emergency overflow capacity (federal emergency sites).
    • Enact expedited sponsor placement protocols for low-risk cases.
    • Meter regional CBP transfers in coordination with Border Patrol leadership.
    • Request inter-agency emergency personnel details.
```

---

## 5. What-If Scenario Stress Testing & Staffing Translation

Using our scenario simulation engine, we stress-tested the federal shelter network against potential border influx and placement bottlenecks from the December 2025 baseline (2,484 children, 3,000-bed baseline threshold):

| Operational Scenario | Intake Shock | Discharge Shock | 30-Day Peak Load | Bed Capacity Status | Days to Breach | Direct Care Staff Needed (1:8) | Caseworkers Needed (1:10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Status Quo (Baseline)** | 0% | 0% | 2,480 | 82.7% Utilization | Safe | 310 Staff | 248 Staff |
| **Mild Spring Surge** | +25% | 0% | 2,674 | 89.1% Utilization | Safe | 335 Staff | 268 Staff |
| **Severe Border Surge** | +50% | 0% | 2,868 | 95.6% Utilization | Safe | 359 Staff | 287 Staff |
| **Vetting Deceleration** | 0% | -30% | 2,658 | 88.6% Utilization | Safe | 333 Staff | 266 Staff |
| **Compound Crisis** | **+40%** | **-25%** | **3,012** | **100.4% (Breach)** | **Day 28** | **377 Staff (+67)** | **302 Staff (+54)** |
| **Extreme Surge (2023 Scale)** | **+100%** | **-30%** | **3,720** | **124.0% (Breach)** | **Day 11** | **465 Staff (+155)** | **372 Staff (+124)** |

**Strategic Implication:** Under a compound crisis (+40% intake surge coupled with a -25% discharge deceleration), HHS has exactly **28 days of advance lead time** before operational capacity is exhausted. This provides sufficient lead time to mobilize standby facilities and staff provided action is initiated at Tier 2 (Day 7).

---

## 6. Actionable 30-60-90 Day Implementation Roadmap

To operationalize this predictive intelligence system across HHS/ORR and CBP, we recommend the following phased deployment plan:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   DAYS 1 - 30   │ ──> │   DAYS 31 - 60  │ ──> │   DAYS 61 - 90  │
│ Dashboard & API │     │ Policy SOPs &   │     │ Full Enterprise │
│ Integration     │     │ Drills          │     │ Integration     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Days 1 – 30: System Deployment & Inter-Agency Integration
- **Deploy Streamlit Analytics Dashboard:** Host the validated application on secure HHS internal infrastructure for daily use by ORR Division of Unaccompanied Children Operations (DUCO).
- **Automate Pipeline Ingestion:** Establish an automated daily data pipeline linking CBP border encounter reporting directly to the ORR feature store.
- **Set Up Real-Time Alert Feeds:** Configure automated email/SMS alerts triggered when projected Surge Lead Time drops below 14 days or Capacity Breach Probability exceeds 20%.

### Days 31 – 60: Operational Standard Operating Procedures (SOPs)
- **Codify 4-Tier Readiness Matrix:** Formalize the Tier 1–4 action matrix into official ORR operational policies, establishing clear lines of authority for contractor pre-alerts and ICF mobilization.
- **Conduct Desktop Surge Drills:** Execute simulated surge tabletop exercises with regional shelter coordinators using the What-If Scenario Simulator.
- **Refine Weekend Placement Protocol:** Test weekend caseworker skeleton shifts to mitigate Monday placement backlogs.

### Days 61 – 90: Enterprise Scaling & Continuous Retraining
- **Establish Automated Continuous Model Retraining:** Implement automated weekly retraining pipelines with drift monitoring to ensure SARIMA and Gradient Boosting parameters remain tuned to evolving migration dynamics.
- **Expand Regional Stratification:** Extend the modeling architecture from national aggregates to regional shelter hubs (e.g., Texas, Arizona, California, Florida).
- **Executive Review & Reporting:** Institutionalize bi-weekly predictive briefings for the HHS Secretary, ACF Assistant Secretary, and CBP Leadership.

---

## 7. Conclusion

By shifting from retrospective reporting to forward-looking predictive foresight, HHS and CBP leadership can transition from reactive crisis management to proactive operational excellence. With over 98% forecasting accuracy up to 30 days in advance, the predictive tools and operational frameworks presented here ensure that the U.S. government maintains uninterrupted legal compliance, minimizes operational costs, and safeguards the health and welfare of vulnerable children in federal care.
