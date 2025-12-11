import pandas as pd
import numpy as np
from src.config import PORT_FILES, PORTS_CLEAN_FILE

def extract_outlets(cell):
    """Extract total charging outlets from 'stations | outlets' string."""
    if isinstance(cell, str) and "|" in cell:
        parts = [p.strip().replace(",", "") for p in cell.split("|")]
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return np.nan
    return np.nan

def parse_ports_for_year(path, year):
    df = pd.read_csv(path)

    # Rename columns
    df = df.rename(columns={
        "Station Counts by State and Fuel Type": "state",
        "Unnamed: 4": "electric"
    })

    # Keep only rows that actually have a state name
    state_rows = df[df["state"].notna() & (df["state"] != "State")].copy()

    # Extract total outlets from 'stations | outlets'
    state_rows["ports_total"] = state_rows["electric"].apply(extract_outlets)
    out = state_rows[["state", "ports_total"]].copy()
    out["year"] = year
    return out

def main():
    frames = []
    for year, path in PORT_FILES.items():
        print(f"Parsing ports for {year} from {path}...")
        frames.append(parse_ports_for_year(path, year))

    ports = pd.concat(frames, ignore_index=True)

    # Drop rows that are regions, not states
    bad_names = [
        "United States", "U.S. Total", "District of Columbia",
        "U.S. Territories", "Other"
    ]
    ports = ports[~ports["state"].isin(bad_names)]

    ports.to_csv(PORTS_CLEAN_FILE, index=False)
    print(f"Saved clean ports data to {PORTS_CLEAN_FILE}")

if __name__ == "__main__":
    main()
