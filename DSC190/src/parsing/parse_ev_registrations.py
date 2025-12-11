import pandas as pd
from src.config import RAW_DIR, EV_REG_CLEAN_FILE

YEAR_FILES = {
    2020: "ev_registrations_2020.csv",
    2021: "ev_registrations_2021.csv",
    2022: "ev_registrations_2022.csv",
    2023: "ev_registrations_2023.csv",
}

def load_one_year(year, filename):
    path = RAW_DIR / filename
    df = pd.read_csv(path)

    # Print columns
    print(f"\n=== {year} columns in {filename} ===")
    print(df.columns.tolist())

    # Add Year column
    if "Year" not in df.columns and "year" not in df.columns:
        df["Year"] = year

    # Standardize state/year column names
    rename_map = {}
    for col in df.columns:
        if col.lower() == "state":
            rename_map[col] = "state"
        if col.lower() == "year":
            rename_map[col] = "year"
    df = df.rename(columns=rename_map)

    # Find the EV column:
    #  - prefer an exact match
    #  - otherwise look for something containing "Electric" but not "Hybrid"
    ev_col = None
    if "ev_count" in df.columns:
        ev_col = "ev_count"
    elif "Electric (EV)" in df.columns:
        ev_col = "Electric (EV)"
    elif "Registration Count" in df.columns:
        ev_col = "Registration Count"
    else:
        candidates = [
            c for c in df.columns
            if "electric" in c.lower() and "hybrid" not in c.lower()
        ]
        if candidates:
            ev_col = candidates[0]

    if ev_col is None:
        raise ValueError(
            f"Could not find EV count column in {filename}. "
            f"Columns: {df.columns.tolist()}"
        )

    # Clean the EV count: remove commas, convert to int
    ev_series = (
        df[ev_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    out = pd.DataFrame({
        "state": df["state"],
        "year": df["year"],
        "ev_count": ev_series,
    })

    return out

def main():
    frames = []
    for year, fname in YEAR_FILES.items():
        print(f"\nLoading EV registrations for {year} from {fname}...")
        frames.append(load_one_year(year, fname))

    ev_all = pd.concat(frames, ignore_index=True)
    ev_all.to_csv(EV_REG_CLEAN_FILE, index=False)
    print(f"\nSaved combined EV registrations to {EV_REG_CLEAN_FILE}")

if __name__ == "__main__":
    main()