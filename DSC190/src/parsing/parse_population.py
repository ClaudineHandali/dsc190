import pandas as pd
from src.config import POP_FILE, POP_CLEAN_FILE

def main():
    df = pd.read_csv(POP_FILE)

    # SUMLEV 40 = states; keep only those
    states = df[df["SUMLEV"] == 40].copy()

    # Keep name + population estimates for 2020–2023
    pop_cols = ["POPESTIMATE2020", "POPESTIMATE2021", "POPESTIMATE2022", "POPESTIMATE2023"]
    keep = ["NAME"] + pop_cols
    states = states[keep]

    # Wide to long
    pop_long = states.melt(
        id_vars="NAME",
        value_vars=pop_cols,
        var_name="pop_year",
        value_name="population"
    )
    pop_long["year"] = pop_long["pop_year"].str.extract(r"(\d{4})").astype(int)
    pop_long = pop_long.drop(columns=["pop_year"])
    pop_long = pop_long.rename(columns={"NAME": "state"})

    pop_long.to_csv(POP_CLEAN_FILE, index=False)
    print(f"Saved clean population data to {POP_CLEAN_FILE}")

if __name__ == "__main__":
    main()
