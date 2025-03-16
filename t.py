# ========== 1. Load and Preprocess Dataset ==========
import pandas as pd

# Load CSV file
df = pd.read_csv("Phishing_Email.csv", index_col=0)

# Display the first few rows to verify structure
print(df.head())
# Check column names and structure
print(df.columns)