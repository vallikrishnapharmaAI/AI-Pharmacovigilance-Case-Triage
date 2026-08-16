import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "cases.csv"

# Load pharmacovigilance case data
df = pd.read_csv(DATA_PATH)

print("Pharmacovigilance Case Triage Project")
print("--------------------------------------")

print(f"Total cases: {len(df)}")

print("\nDataset:")
print(df)

print("\nSeriousness distribution:")
print(df["seriousness"].value_counts())

print("\nEvent distribution:")
print(df["event"].value_counts())

print("\nData loading completed successfully.")
