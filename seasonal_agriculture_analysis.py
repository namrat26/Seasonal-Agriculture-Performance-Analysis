import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "seasonal_agriculture_performance_dataset.csv"
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# Data quality
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Clean selected numeric columns
for col in ["Rainfall_mm", "Soil_Moisture_pct", "Yield_Tonnes_Ha"]:
    df[col] = df[col].fillna(df[col].median())

# Seasonal analysis
season_order = ["Kharif", "Rabi", "Zaid"]
seasonal = df.groupby("Season").agg(
    Farms=("Farm_ID","count"),
    Avg_Yield=("Yield_Tonnes_Ha","mean"),
    Avg_Production=("Production_Tonnes","mean"),
    Avg_Revenue=("Revenue_INR","mean"),
    Avg_Profit=("Profit_INR","mean"),
    Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3","mean"),
    Avg_Disease_Pest_Risk=("Disease_Pest_Risk_pct","mean")
).reindex(season_order)
print("\nSeasonal summary:\n", seasonal.round(2))
seasonal.to_csv(OUTPUT/"seasonal_summary.csv")

# Crop analysis
crop = df.groupby("Crop").agg(
    Farms=("Farm_ID","count"),
    Avg_Yield=("Yield_Tonnes_Ha","mean"),
    Avg_Profit=("Profit_INR","mean")
).sort_values("Avg_Profit", ascending=False)
print("\nCrop summary:\n", crop.round(2))
crop.to_csv(OUTPUT/"crop_summary.csv")

# Irrigation analysis
irrig = df.groupby("Irrigation_Method").agg(
    Avg_Yield=("Yield_Tonnes_Ha","mean"),
    Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3","mean"),
    Avg_Profit=("Profit_INR","mean")
).sort_values("Avg_Yield", ascending=False)
print("\nIrrigation summary:\n", irrig.round(2))
irrig.to_csv(OUTPUT/"irrigation_summary.csv")

# Correlation
cols=["Yield_Tonnes_Ha","Profit_INR","Rainfall_mm","Water_Efficiency_t_per_1000m3","Disease_Pest_Risk_pct"]
corr=df[cols].corr()
print("\nCorrelation:\n", corr.round(2))
corr.to_csv(OUTPUT/"correlation_matrix.csv")

# Charts
seasonal["Avg_Yield"].plot(kind="bar", figsize=(8,5), title="Average Yield by Season")
plt.ylabel("Yield (tonnes/ha)"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUTPUT/"average_yield_by_season.png", dpi=200); plt.close()

(seasonal["Avg_Profit"]/1000).plot(kind="bar", figsize=(8,5), title="Average Profit by Season")
plt.ylabel("Profit (₹ thousand/farm)"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUTPUT/"average_profit_by_season.png", dpi=200); plt.close()

irrig["Avg_Yield"].plot(kind="bar", figsize=(8,5), title="Average Yield by Irrigation Method")
plt.ylabel("Yield (tonnes/ha)"); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(OUTPUT/"yield_by_irrigation.png", dpi=200); plt.close()

(crop["Avg_Profit"]/1000).plot(kind="bar", figsize=(9,5), title="Average Profit by Crop")
plt.ylabel("Profit (₹ thousand/farm)"); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(OUTPUT/"profit_by_crop.png", dpi=200); plt.close()

print("\nAnalysis completed. Results saved in outputs/.")
