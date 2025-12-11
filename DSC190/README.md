# Evaluating the Influence of Fuel Costs and Charging Infrastructure on National and State-Level Electric Vehicle Adoption

This repository contains my version of the DSC 190 project. It builds a state–year panel that combines EV registrations, public charging infrastructure, population, and gasoline prices to explore how fuel costs and charging access relate to electric vehicle (EV) adoption over time in the United States.

---

## Project Overview

**Goal**

Quantify how EV adoption differs across states and over time, and evaluate the role of:

- Real gasoline prices
- Public charging ports per capita  
- Population and state-specific differences

The project also produces simple forecasts of EV adoption under different charging infrastructure growth scenarios.

### Main Research Questions

1. **RQ1 → Gas prices and EV adoption**  
   How are real gasoline prices associated with EV adoption (EVs per 1,000 residents) in the United States at the national and state level from 2020–2023?

2. **RQ2 → Charging ports and EV adoption**  
   How strongly is the availability of public charging infrastructure (ports per 100,000 residents) associated with EV adoption (EVs per 1,000 residents) across U.S. states from 2020–2023?

3. **RQ3 → Forecasting under alternative charging scenarios**  
   Under baseline versus accelerated growth in public charging infrastructure, how is EV adoption projected to evolve in the next few years?

### Guiding Questions 

These are the more intuitive questions that motivated the analysis:

1. How have EVs per 1,000 people evolved nationally and across states since 2020?  
2. How strongly are gasoline prices associated with changes in EV adoption?  
3. How does public charging infrastructure per capita relate to EV penetration?  
4. How might EV adoption evolve over the next five years under baseline vs. accelerated charging build-out?

---

## Approach

1. **Build a unified state–year dataset**
   - Parse and clean raw data on:
     - EV registrations by state and year
     - Public charging ports
     - State population
     - Gasoline prices (inflation-adjusted to 2023 dollars)
   - Merge these sources into a single panel dataset by state and year.
   - Create per-capita metrics such as:
     - EVs per 1,000 people
     - Ports per 100,000 people

2. **Calculate descriptive statistics and trends**
   - Summarize EV adoption and charging infrastructure by state and year.
   - Identify top and bottom states in EVs per 1,000 residents.
   - Examine how charging ports per capita have changed over time.

3. **Relate charging infrastructure and EV adoption**
   - Fit a panel regression with state fixed effects:
     - Outcome: EVs per 1,000 people
     - Predictors: chargers per 100k, year trend, and (optionally) gasoline prices
   - Interpret how changes in charging density are associated with changes in EV adoption, after accounting for state-specific differences.

4. **Forecast EV adoption**
   - Use the panel model to generate 5 year forecasts under:
     - A **baseline** scenario where charging ports grow at their historical rate.
     - An **accelerated** scenario where charging ports grow faster than in the past.
   - For comparison, fit simple ARIMA time-series models to EV adoption for each state and produce separate 5-year forecasts.

---

## Repository Structure

```text
.
│
├── Datasets/                 # Raw datasets (CSV/XLSX): EV registrations, ports, population, gas prices
│   ├── 2020_ports.csv
│   ├── 2021_ports.csv
│   ├── 2022_ports.csv
│   ├── 2023_ports.csv
│   ├── ev_registrations_2020.csv
│   ├── ev_registrations_2021.csv
│   ├── ev_registrations_2022.csv
│   ├── ev_registrations_2023.csv
│   ├── population_estimate.csv
│   ├── 10567_pev_sales_2-28-20.xlsx
│   └── 10641_gasoline_prices_by_year_1-26-24.xlsx
│
├── data/
│   └── processed/            # Cleaned data, merged panel, forecasts, and generated figures
│       ├── Cleaned tables
│       ├── Forecast outputs
│       ├── Text summaries
│       └── Figures
│
├── notebooks/                # Jupyter notebooks for exploratory analysis (optional)
│
├── reports/                  # Plots and figures for write-ups or slides
│
├── src/
│   ├── parsing/              # Scripts to read and clean each raw dataset
│   │   ├── parse_ev_registrations.py
│   │   ├── parse_ports.py
│   │   ├── parse_population.py
│   │   └── parse_gas_prices.py
│   │
│   ├── cleaning/             # Data merging / feature construction
│   │   └── build_panel.py    # Builds state–year panel with per-capita metrics
│   │
│   ├── analysis/             # Summary stats and modeling / forecasting
│   │   ├── descriptives.py          # Basic summaries of the panel dataset
│   │   ├── gas_vs_ev.py             # RQ1: national EV adoption vs gas prices
│   │   ├── logspec_fe.py            # RQ2: log–log ports per capita FE model (state FE)
│   │   ├── state_gas_fe.py          # Robustness: state-level EV vs gas with state FE
│   │   └── forecast_ev_panel.py     # Panel + ARIMA forecasts for EV adoption
│   │
│   └── visualization/        # Plotting utilities
│       └── plots.py          # Line charts, scatter plots, and other figures
│
├── src/config.py             # Central paths for raw/processed data files
│
├── README.md                 # This file
└── requirements.txt          # Python dependencies for the project
```
