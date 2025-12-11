import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.config import PANEL_FILE, GAS_CLEAN_FILE, TEXT_SUMMARIES_DIR


def main():
    # ===========================================================================================================
    # 1. Load panel + gas data
    # ===========================================================================================================
    panel = pd.read_csv(PANEL_FILE)
    gas = pd.read_csv(GAS_CLEAN_FILE)

    # Aggregate to national totals by year
    national = (
        panel.groupby("year", as_index=False)
        .agg(
            ev_total=("ev_count", "sum"),
            population_total=("population", "sum"),
        )
    )
    national["ev_per_1000"] = (
        national["ev_total"] / national["population_total"] * 1000.0
    )

    # Keep only the columns we need from gas (year + real price)
    gas = gas[["year", "gas_real_2023"]].copy()

    # Merge EV + gas on year
    merged = (
        national.merge(gas, on="year", how="inner")
        .dropna(subset=["ev_per_1000", "gas_real_2023"])
        .sort_values("year")
    )

    print("Years in merged national EV + gas series:", merged["year"].tolist())
    if len(merged) < 3:
        print(
            "Warning: very few years of overlap; correlations and regressions "
            "will be noisy."
        )


    # ===========================================================================================================
    # 2. Correlations in levels
    # ===========================================================================================================
    pearson_r = merged["gas_real_2023"].corr(merged["ev_per_1000"], method="pearson")
    spearman_r = merged["gas_real_2023"].corr(merged["ev_per_1000"], method="spearman")

    print("\n=== Gas vs EV (levels) ===")
    print(f"Pearson r(ev_per_1000, gas_real_2023)  = {pearson_r:.3f}")
    print(f"Spearman r(ev_per_1000, gas_real_2023) = {spearman_r:.3f}")


    # ===========================================================================================================
    # 3. Correlations in growth rates (year-to-year % change)
    # ===========================================================================================================
    merged["ev_growth"] = merged["ev_per_1000"].pct_change()
    merged["gas_growth"] = merged["gas_real_2023"].pct_change()

    growth = merged.dropna(subset=["ev_growth", "gas_growth"]).copy()

    print("\n=== Gas vs EV (growth rates) ===")
    if len(growth) >= 2:
        pearson_g = growth["gas_growth"].corr(growth["ev_growth"], method="pearson")
        spearman_g = growth["gas_growth"].corr(growth["ev_growth"], method="spearman")

        print(f"Pearson r(ev_growth, gas_growth)  = {pearson_g:.3f}")
        print(f"Spearman r(ev_growth, gas_growth) = {spearman_g:.3f}")
    else:
        print("Not enough years to compute growth-rate correlations.")


    # ===========================================================================================================
    # 4. Simple OLS: EV adoption vs gas price + time trend
    # ===========================================================================================================
    # Keep only rows with complete data
    reg_data = merged.dropna(subset=["ev_per_1000", "gas_real_2023"]).copy()

    # Center year so intercept is more interpretable
    reg_data["year_centered"] = reg_data["year"] - reg_data["year"].mean()

    formula = "ev_per_1000 ~ gas_real_2023 + year_centered"
    model = smf.ols(formula, data=reg_data).fit()

    print("\n=== OLS: ev_per_1000 ~ gas_real_2023 + year_centered ===")
    print(model.summary())


    # ===========================================================================================================
    # 5. Save summary to text file for the report
    # ===========================================================================================================
    out_path = TEXT_SUMMARIES_DIR / "gas_vs_ev_summary.txt"
    with open(out_path, "w") as f:
        f.write("=== Gas vs EV adoption (national series) ===\n\n")
        f.write("Years used: " + ", ".join(map(str, merged["year"].tolist())) + "\n\n")

        f.write("Correlations in levels:\n")
        f.write(f"  Pearson r = {pearson_r:.3f}\n")
        f.write(f"  Spearman r = {spearman_r:.3f}\n\n")

        if len(growth) >= 2:
            f.write("Correlations in growth rates (year-to-year % change):\n")
            f.write(f"  Pearson r = {pearson_g:.3f}\n")
            f.write(f"  Spearman r = {spearman_g:.3f}\n\n")
        else:
            f.write("Not enough years for growth-rate correlations.\n\n")

        f.write("OLS regression: ev_per_1000 ~ gas_real_2023 + year_centered\n\n")
        f.write(model.summary().as_text())

    print(f"\nSaved gas vs ev summary to {out_path}")


if __name__ == "__main__":
    main()
