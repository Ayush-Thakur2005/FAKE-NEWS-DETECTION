# eda.py
import pandas as pd

# Load CSV
csv_path = 'dataset/FakeNewsNet.csv'  # make sure this path is correct
df = pd.read_csv(csv_path)

# Basic info
print("=== Dataset Info ===")
print("Shape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Quick look at top rows
print("\nFirst 5 rows:")
print(df.head())

# Label distribution
if 'real' in df.columns:
    print("\nLabel distribution (real column):")
    print(df['real'].value_counts())
else:
    print("\nNo 'real' column found in dataset.")
