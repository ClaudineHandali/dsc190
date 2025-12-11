import pandas as pd
from src.config import GAS_FILE, GAS_CLEAN_FILE

def main():
    df = pd.read_excel(GAS_FILE, sheet_name="Gas Prices", header=2)

    print("Columns from gas price file:")
    print(df.columns.tolist())

    # Find year column
    year_col = None
    for c in df.columns:
        if "year" in str(c).lower():
            year_col = c
            break
    if year_col is None:
        raise ValueError(f"Could not find year column in gas file: {df.columns.tolist()}")

    # Find 2023$ gas price column
    gas_col = None
    for c in df.columns:
        text = str(c)
        if "2023" in text and "gallon" in text:
            gas_col = c
            break
    if gas_col is None:
        gas_col = df.columns[-1]

    gas = df[[year_col, gas_col]].copy()
    gas = gas.rename(columns={year_col: "year", gas_col: "gas_real_2023"})

    # New: coerce non-numeric years to NaN, then drop them
    gas["year"] = pd.to_numeric(gas["year"], errors="coerce")
    gas = gas[gas["year"].notna()]
    gas["year"] = gas["year"].astype(int)

    gas["gas_real_2023"] = (
        gas["gas_real_2023"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    gas = gas[gas["year"].between(2020, 2023)]

    gas.to_csv(GAS_CLEAN_FILE, index=False)
    print(f"Saved clean gas price data to {GAS_CLEAN_FILE}")

if __name__ == "__main__":
    main()
