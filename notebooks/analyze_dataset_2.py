import pandas as pd
import os

# Define file path
DATA_DIR = "../data"
DATA_FILE = os.path.join(DATA_DIR, "lung_cancer_dataset_2_nancy.csv")

print(f"Analyzing dataset: {DATA_FILE}\n")

try:
    df = pd.read_csv(DATA_FILE)

    print("--- Dataset Info ---")
    df.info()

    print("\n--- Dataset Description (Numerical Columns) ---")
    print(df.describe())

    print("\n--- Value Counts for Each Column ---")
    for col in df.columns:
        print(f"\nValue counts for column: {col}")
        # Show more values if cardinality is high, but limit for clarity
        if df[col].nunique() > 50:
             print(df[col].value_counts().head(10)) # Show top 10 for high cardinality
             print("... (truncated)")
        else:
            print(df[col].value_counts())

    # Check for missing values
    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())

except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_FILE}")
except Exception as e:
    print(f"An error occurred during analysis: {e}")

print("\n--- Analysis Complete ---")

