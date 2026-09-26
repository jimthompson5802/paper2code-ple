from pathlib import Path

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

df = housing.frame

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
output_path = data_dir / "ca-housing.csv"
df.to_csv(output_path, index=False)

print(df.head())
print(df.shape)
print(f"Saved dataset to {output_path}")