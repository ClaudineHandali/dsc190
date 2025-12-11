import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

from src.config import PANEL_FILE, FORECAST_DIR


# =========================================================================================================
# 1. Load panel
# =========================================================================================================
panel = pd.read_csv(PANEL_FILE)
col_map = {
    "state": "State",
    "year": "Year",
    "ev_per_1000": "EVs_per_1000",
    "ports_per_100k": "Outlets_per_100k",
}
panel = panel.rename(columns=col_map)

required = ["State", "Year", "EVs_per_1000", "Outlets_per_100k"]
missing = [c for c in required if c not in panel.columns]
if missing:
    raise ValueError(f"Panel is missing columns: {missing}")

# Use only the years
merged = panel[required].dropna().copy()


# ===========================================================================================================
# 2. Panel regression with FE
# ===========================================================================================================
merged["Year_trend"] = merged["Year"] - merged["Year"].min()

panel_formula = "EVs_per_1000 ~ Outlets_per_100k + Year_trend + C(State)"
panel_model = smf.ols(
    panel_formula,
    data=merged
).fit(cov_type="cluster", cov_kwds={"groups": merged["State"]})

print("=== Panel regression summary ===")
print(panel_model.summary())


# ===========================================================================================================
# 3. Panel forecasts: baseline & accelerated
# ===========================================================================================================

last_year = merged["Year"].max()
first_year = merged["Year"].min()
forecast_years = list(range(last_year + 1, last_year + 6))  # 5 years ahead

# Average percentage growth in outlets per 100k from first year to last year
outlets_last = merged[merged["Year"] == last_year].set_index("State")["Outlets_per_100k"]
outlets_first = merged[merged["Year"] == first_year].set_index("State")["Outlets_per_100k"]

pct_growth = outlets_last / outlets_first - 1.0
avg_pct_growth = pct_growth.replace([np.inf, -np.inf], np.nan).dropna().mean()

print(f"\nAverage annualized outlet growth (first→last year): {avg_pct_growth:.3f}")

# State baseline at last observed year
state_base = (
    merged[merged["Year"] == last_year]
    .set_index("State")[["Outlets_per_100k", "EVs_per_1000"]]
)

# --- Scenario 1: Baseline outlet growth ---
forecast_rows = []
for state, row in state_base.iterrows():
    outlets0 = row["Outlets_per_100k"]
    for i, y in enumerate(forecast_years, start=1):
        outlets_proj = outlets0 * ((1 + avg_pct_growth) ** i)
        year_trend = y - first_year
        pred_df = pd.DataFrame(
            {
                "Outlets_per_100k": [outlets_proj],
                "Year_trend": [year_trend],
                "State": [state],
            }
        )
        pred = float(panel_model.predict(pred_df)[0])
        forecast_rows.append(
            {
                "State": state,
                "Year": y,
                "EVs_per_1000_forecast_panel_baseline": pred,
                "Outlets_per_100k_proj": outlets_proj,
            }
        )

forecast_panel = pd.DataFrame(forecast_rows)
print("\n=== Panel forecasts (baseline) – mean EVs_per_1000 by year ===")
print(forecast_panel.groupby("Year")["EVs_per_1000_forecast_panel_baseline"].mean())

# --- Scenario 2: Accelerated outlet rollout (+10 percentage points) ---
acc = 0.10  # extra 10% growth per year
forecast_rows_acc = []
for state, row in state_base.iterrows():
    outlets0 = row["Outlets_per_100k"]
    for i, y in enumerate(forecast_years, start=1):
        outlets_proj = outlets0 * ((1 + avg_pct_growth + acc) ** i)
        year_trend = y - first_year
        pred_df = pd.DataFrame(
            {
                "Outlets_per_100k": [outlets_proj],
                "Year_trend": [year_trend],
                "State": [state],
            }
        )
        pred = float(panel_model.predict(pred_df)[0])
        forecast_rows_acc.append(
            {
                "State": state,
                "Year": y,
                "EVs_per_1000_forecast_panel_acc": pred,
                "Outlets_per_100k_proj": outlets_proj,
            }
        )

forecast_panel_acc = pd.DataFrame(forecast_rows_acc)
print("\n=== Panel forecasts (accelerated) – mean EVs_per_1000 by year ===")
print(forecast_panel_acc.groupby("Year")["EVs_per_1000_forecast_panel_acc"].mean())


# =====================================================================================================
# 4. Per-state ARIMA time-series EVs
# =====================================================================================================

arima_forecasts = []
for state, g in merged.groupby("State"):
    g_sorted = g.sort_values("Year")
    if len(g_sorted) >= 3:
        ts = g_sorted.set_index("Year")["EVs_per_1000"]
        try:
            model = ARIMA(ts, order=(1, 1, 0)).fit()
            fc = model.get_forecast(steps=len(forecast_years))
            mean_fc = fc.predicted_mean
            for i, y in enumerate(forecast_years):
                arima_forecasts.append(
                    {
                        "State": state,
                        "Year": y,
                        "EVs_per_1000_arima": float(mean_fc.iloc[i]),
                    }
                )
        except Exception:
            continue

arima_df = pd.DataFrame(arima_forecasts)

print("\n=== ARIMA forecasts – mean EVs_per_1000 by year (across states) ===")
if not arima_df.empty:
    print(arima_df.groupby("Year")["EVs_per_1000_arima"].mean())
else:
    print("Not enough data for ARIMA forecasts.")


# =========================================================================================================
# 5. Save to CSV 
# =========================================================================================================

out_baseline = FORECAST_DIR / "forecast_panel_baseline.csv"
out_acc = FORECAST_DIR / "forecast_panel_accelerated.csv"
out_arima = FORECAST_DIR / "forecast_arima.csv"

forecast_panel.to_csv(out_baseline, index=False)
forecast_panel_acc.to_csv(out_acc, index=False)
arima_df.to_csv(out_arima, index=False)

print("\nSaved forecast CSVs:")
print(" -", out_baseline)
print(" -", out_acc)
print(" -", out_arima)
