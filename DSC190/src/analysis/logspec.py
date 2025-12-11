import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.config import PANEL_FILE, TEXT_SUMMARIES_DIR


def main():
    # ===========================================================================================================
    # 1. Load panel and prepare log variables
    # ===========================================================================================================
    panel = pd.read_csv(PANEL_FILE).copy()

    # Drop rows with missing key variables
    panel = panel.dropna(subset=["ev_per_1000", "ports_per_100k"])

    # Avoid log(0) by clipping at a small positive number
    eps = 1e-3
    panel["ev_per_1000_adj"] = panel["ev_per_1000"].clip(lower=eps)
    panel["ports_per_100k_adj"] = panel["ports_per_100k"].clip(lower=eps)

    panel["log_ev_per_1000"] = np.log(panel["ev_per_1000_adj"])
    panel["log_ports_per_100k"] = np.log(panel["ports_per_100k_adj"])

    # Rename 
    df = panel.rename(columns={"state": "State", "year": "Year"})
    df["Year_trend"] = df["Year"] - df["Year"].min()

    print("Years in panel:", sorted(df["Year"].unique()))
    print("Number of states:", df["State"].nunique())


    # ===========================================================================================================
    # 2. Log–log fixed-effects regression with clustered SEs
    # ===========================================================================================================
    formula = "log_ev_per_1000 ~ log_ports_per_100k + Year_trend + C(State)"
    ols_model = smf.ols(formula, data=df)
    fe_log = ols_model.fit(cov_type="cluster", cov_kwds={"groups": df["State"]})

    print("\n=== Robustness: log–log FE model ===")
    print("Formula:", formula)
    print(fe_log.summary())

    # Extract elasticity and 95% CI for log_ports_per_100k
    coef_name = "log_ports_per_100k"
    if coef_name in fe_log.params.index:
        beta = fe_log.params[coef_name]
        ci_low, ci_high = fe_log.conf_int().loc[coef_name]

        print("\nElasticity interpretation:")
        print(
            f"  A 1% increase in ports_per_100k is associated with "
            f"approximately {beta:.3f}% change in ev_per_1000."
        )
        print(
            f"  95% CI for elasticity: [{ci_low:.3f}, {ci_high:.3f}] "
            "(clustered by state)"
        )
    else:
        beta = np.nan
        ci_low = ci_high = np.nan
        print("\nWarning: log_ports_per_100k coefficient not found in the model.")


    # ===========================================================================================================
    # 3. Save to txt
    # ===========================================================================================================
    out_path = TEXT_SUMMARIES_DIR / "logspec_summary.txt"
    with open(out_path, "w") as f:
        f.write("=== Robustness: log–log FE regression ===\n\n")
        f.write(f"Formula: {formula}\n\n")
        f.write(fe_log.summary().as_text())

        f.write("\n\nKey elasticity result:\n")
        if not np.isnan(beta):
            f.write(
                f"  Elasticity of ev_per_1000 with respect to ports_per_100k: "
                f"{beta:.3f}\n"
            )
            f.write(
                f"  95% CI (clustered by state): [{ci_low:.3f}, {ci_high:.3f}]\n"
            )
        else:
            f.write("  Coefficient for log_ports_per_100k not found.\n")

    print(f"\nSaved log-spec summary to {out_path}")


if __name__ == "__main__":
    main()
