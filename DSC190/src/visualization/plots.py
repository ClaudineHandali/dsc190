import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PANEL_FILE, PROCESSED_DIR, GAS_CLEAN_FILE


def build_national_ev_gas(panel: pd.DataFrame, gas: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the state panel to a national EV-per-1000 series and merge
    with real gas prices. Returns columns:
      year, ev_per_1000, gas_real_2023, ev_growth, gas_growth
    """
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

    gas = gas[["year", "gas_real_2023"]].copy()

    merged = (
        national.merge(gas, on="year", how="inner")
        .dropna(subset=["ev_per_1000", "gas_real_2023"])
        .sort_values("year")
    )

    merged["ev_growth"] = merged["ev_per_1000"].pct_change()
    merged["gas_growth"] = merged["gas_real_2023"].pct_change()

    return merged


def plot_ev_gas_timeseries(merged: pd.DataFrame) -> None:
    """Time-series plot of national EV adoption and real gas prices."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # EV per 1,000 on left y-axis
    ax1.plot(
        merged["year"],
        merged["ev_per_1000"],
        marker="o",
        label="EVs per 1,000 (national)",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("EVs per 1,000 people")
    ax1.tick_params(axis="y")

    # Gas price on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        merged["year"],
        merged["gas_real_2023"],
        marker="s",
        linestyle="--",
        label="Gas price (real 2023 $/gal)",
    )
    ax2.set_ylabel("Gas price (2023 dollars per gallon)")
    ax2.tick_params(axis="y")

    plt.title("National EV Adoption and Real Gas Prices Over Time")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    out_path = PROCESSED_DIR / "ev_gas_timeseries.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_ev_vs_gas_scatter_levels(merged: pd.DataFrame) -> None:
    """Scatter of national EV per 1,000 vs real gas prices (levels)."""
    plt.figure(figsize=(6, 5))
    sns.regplot(
        data=merged,
        x="gas_real_2023",
        y="ev_per_1000",
        marker="o",
    )

    for _, row in merged.iterrows():
        plt.text(
            row["gas_real_2023"],
            row["ev_per_1000"],
            str(int(row["year"])),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Gas price (2023 dollars per gallon)")
    plt.ylabel("EVs per 1,000 people (national)")
    plt.title("National EV Adoption vs Gas Price (Levels)")
    plt.tight_layout()

    out_path = PROCESSED_DIR / "ev_vs_gas_scatter_levels.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def plot_ev_vs_gas_scatter_growth(merged: pd.DataFrame) -> None:
    """Scatter of national EV growth vs gas price growth (year-to-year % change)."""
    growth = merged.dropna(subset=["ev_growth", "gas_growth"]).copy()
    if growth.empty:
        print("Not enough years for growth-rate scatter.")
        return

    plt.figure(figsize=(6, 5))
    sns.regplot(
        data=growth,
        x="gas_growth",
        y="ev_growth",
        marker="o",
    )

    for _, row in growth.iterrows():
        plt.text(
            row["gas_growth"],
            row["ev_growth"],
            str(int(row["year"])),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Gas price growth (year-to-year % change)")
    plt.ylabel("EV adoption growth (year-to-year % change)")
    plt.title("EV Adoption vs Gas Price (Growth Rates)")
    plt.tight_layout()

    out_path = PROCESSED_DIR / "ev_vs_gas_scatter_growth.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def lineplot_top_states_ev(panel: pd.DataFrame, top_n: int = 5) -> None:
    """Line plot of EV adoption over time for the top N states."""
    latest_year = panel["year"].max()
    latest = panel[panel["year"] == latest_year].copy()
    top_states = (
        latest.sort_values("ev_per_1000", ascending=False)
        .head(top_n)["state"]
        .tolist()
    )

    subset = panel[panel["state"].isin(top_states)].copy()

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=subset,
        x="year",
        y="ev_per_1000",
        hue="state",
        marker="o",
    )
    plt.title(f"EVs per 1,000 People in Top {top_n} States")
    plt.ylabel("EVs per 1,000 people")
    plt.xlabel("Year")
    plt.tight_layout()

    out_path = PROCESSED_DIR / "ev_per_1000_top_states.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def scatter_ports_vs_ev(panel: pd.DataFrame) -> None:
    """Scatter of ports per 100k vs EVs per 1,000 by state-year."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=panel,
        x="ports_per_100k",
        y="ev_per_1000",
        hue="year",
    )
    plt.xlabel("Public charging ports per 100,000 people")
    plt.ylabel("EVs per 1,000 people")
    plt.title("EV Adoption vs Public Charging Ports (State-Year)")
    plt.tight_layout()

    out_path = PROCESSED_DIR / "ports_vs_ev_scatter.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def main() -> None:
    panel = pd.read_csv(PANEL_FILE)
    gas = pd.read_csv(GAS_CLEAN_FILE)

    # State-level EV vs ports plots
    lineplot_top_states_ev(panel, top_n=5)
    scatter_ports_vs_ev(panel)

    # National EV + gas plots
    merged = build_national_ev_gas(panel, gas)
    print("Years in national EV + gas series:", merged["year"].tolist())
    plot_ev_gas_timeseries(merged)
    plot_ev_vs_gas_scatter_levels(merged)
    plot_ev_vs_gas_scatter_growth(merged)


if __name__ == "__main__":
    main()
