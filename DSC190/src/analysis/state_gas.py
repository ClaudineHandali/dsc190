import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.config import PANEL_FILE, TEXT_SUMMARIES_DIR

def run_state_gas_fe():
    """
    State-level panel regression of EV adoption on gas prices with:
      log(ev_per_1000) ~ log(gas_real_2023) + year_centered + state FE
    Clustered SEs at the state level.
    """
    # Load panel
    panel = pd.read_csv(PANEL_FILE)
    df = panel.copy()

    # Restrict to years where you trust both EV and gas data
    df = df[df["year"].between(2020, 2023)]

    # Keep needed columns and drop missing
    cols = ["state", "year", "ev_per_1000", "gas_real_2023"]
    df = df[cols].dropna()

    # Require strictly positive values for logs
    df = df[(df["ev_per_1000"] > 0) & (df["gas_real_2023"] > 0)].copy()

    # Logs + centered year
    df["log_ev_per_1000"] = np.log(df["ev_per_1000"])
    df["log_gas_real_2023"] = np.log(df["gas_real_2023"])
    df["year_centered"] = df["year"] - df["year"].mean()

    formula = "log_ev_per_1000 ~ log_gas_real_2023 + year_centered + C(state)"

    model = smf.ols(formula, data=df)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["state"]},
    )

    # Elasticity + 95% CI (log-log means coeff is elasticity)
    coef = results.params["log_gas_real_2023"]
    se = results.bse["log_gas_real_2023"]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    lines = []
    lines.append("State-level FE regression of EV adoption on gas prices\n")
    lines.append(f"Formula: {formula}\n\n")
    lines.append(results.summary().as_text())
    lines.append("\n\nElasticity interpretation (log-log):\n")
    lines.append(
        f"  coef(log_gas_real_2023) = {coef:.3f}\n"
        f"  95% CI = [{ci_low:.3f}, {ci_high:.3f}]\n"
        "Read as: a 1% increase in real gas prices is associated with an "
        f"approximate {coef*100:.1f}% change in EVs per 1,000 residents,\n"
        "holding state fixed effects and the overall time trend constant.\n"
    )
    lines.append(f"\nNumber of states: {df['state'].nunique()}\n")
    lines.append(f"Number of state-years: {len(df)}\n")

    out_path = TEXT_SUMMARIES_DIR / "state_gas_summary.txt"
    with out_path.open("w") as f:
        for line in lines:
            f.write(line)
    
    print(f"Regression summary to {out_path}")


if __name__ == "__main__":
    run_state_gas_fe()