import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns

#======================================================
# Cleaning and First Look
#======================================================
# Load the Excel file
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')

# Parse a specific sheet
df_lcf = xls.parse('LCF summary') 

print(df_lcf.head(12))
print(df_lcf.info())

# Dataset Overview and Statistical Summary

# === STEP 1: Dataset Overview ===
print("Shape of dataset:", df_lcf.shape)
print("\nColumn names:", df_lcf.columns.tolist())
print("\nData types:\n", df_lcf.dtypes)
print("\nMissing values per column:\n", df_lcf.isnull().sum())

# === STEP 2: Statistical Summary ===
print("\nStatistical Summary:\n", df_lcf.describe())

# === STEP 3: Quick Visuals for Distributions === #NOTE Great resource for first raw look at column distributions
numeric_cols = df_lcf.select_dtypes(include='number').columns

plt.figure(figsize=(14, 8))
df_lcf[numeric_cols].hist(bins=20, figsize=(14, 8))
plt.suptitle("Histograms of Numeric Columns", fontsize=16)
plt.show()

# === STEP 4: Correlation Heatmap === #NOTE Also a great initial resource
plt.figure(figsize=(10, 8))
sns.heatmap(df_lcf.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

