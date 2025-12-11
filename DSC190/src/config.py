from pathlib import Path

# Project root is the folder containing this src/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw + processed roots
RAW_DIR = PROJECT_ROOT / "Datasets"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Subfolders inside processed/
CLEANED_DIR = PROCESSED_DIR / "cleaned table"
FIGURES_DIR = PROCESSED_DIR / "figures"
FORECAST_DIR = PROCESSED_DIR / "forecast output"
TEXT_SUMMARIES_DIR = PROCESSED_DIR / "text summaries"

for d in (CLEANED_DIR, FIGURES_DIR, FORECAST_DIR, TEXT_SUMMARIES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Central place for raw file names
PORT_FILES = {
    2020: RAW_DIR / "2020_ports.csv",
    2021: RAW_DIR / "2021_ports.csv",
    2022: RAW_DIR / "2022_ports.csv",
    2023: RAW_DIR / "2023_ports.csv",
}

POP_FILE = RAW_DIR / "population_estimate.csv"
GAS_FILE = RAW_DIR / "10641_gasoline_prices_by_year_1-26-24.xlsx"
EV_REG_FILE = RAW_DIR / "ev_registrations.csv"  

# Output files 
PORTS_CLEAN_FILE = CLEANED_DIR / "ports_clean.csv"
POP_CLEAN_FILE = CLEANED_DIR / "population_states.csv"
GAS_CLEAN_FILE = CLEANED_DIR / "gas_prices_clean.csv"
EV_REG_CLEAN_FILE = CLEANED_DIR / "ev_registrations_clean.csv"
PANEL_FILE = CLEANED_DIR / "panel.csv"
