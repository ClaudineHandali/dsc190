import pandas as pd
from src.config import PANEL_FILE

def main():
    panel = pd.read_csv(PANEL_FILE)

    print("Panel shape:", panel.shape)
    print("\nColumns:\n", panel.columns.tolist())

    print("\nYears:", sorted(panel["year"].unique()))
    print("\nExample states:", panel["state"].unique()[:10])

    print("\nSummary of EV per 1000 people:")
    print(panel["ev_per_1000"].describe())

    print("\nSummary of ports per 100k people:")
    print(panel["ports_per_100k"].describe())

if __name__ == "__main__":
    main()
