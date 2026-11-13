import pandas as pd

# Read the parquet file
df = pd.read_parquet(r"C:\Users\ABHISHEK SHARMA\Desktop\DIP\data\processed\ML_Dataset\combined\raichur_features.parquet")
# View the first 5 rows to check the data
print(df.head())