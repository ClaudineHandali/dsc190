import pandas as pd
from src.config import (
    PORTS_CLEAN_FILE,
    POP_CLEAN_FILE,
    GAS_CLEAN_FILE,
    EV_REG_CLEAN_FILE,
    PANEL_FILE,
)

def main():
    ports = pd.read_csv(PORTS_CLEAN_FILE)
    pop = pd.read_csv(POP_CLEAN_FILE)
    gas = pd.read_csv(GAS_CLEAN_FILE)
    ev = pd.read_csv(EV_REG_CLEAN_FILE)

    # Merge ports + population + EVs on state + year
    panel = (
        ev.merge(ports, on=["state", "year"], how="left")
          .merge(pop,  on=["state", "year"], how="left")
    )

    # Merge gas price on year only
    panel = panel.merge(gas, on="year", how="left")

    # Per-capita metrics
    panel["ev_per_1000"] = panel["ev_count"] / panel["population"] * 1000
    panel["ports_per_100k"] = panel["ports_total"] / panel["population"] * 100_000

    panel.to_csv(PANEL_FILE, index=False)
    print(f"Saved panel dataset to {PANEL_FILE}")

if __name__ == "__main__":
    main()
