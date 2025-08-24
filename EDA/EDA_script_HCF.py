import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# Explore the HCF Dataset and Feature Engineering for HCF
#============================================================

#-------------------------------
# Load and parse 
#------------------------------
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')
df1 = xls.parse('HCF summary') 
df2 = xls.parse('HCF individual dataset')

print('HCF summary info:', df1.info())
print('\nHCF individual dataset', df2.info())

# Step 1: Inspect and Merge HCF Data
# ----------------------------------

# Assume df1 = HCF summary dataset, df2 = HCF individual dataset
# Merge on ID
df_hcf = pd.merge(df2, df1, on='ID', how='left')

# print('\nHCF Info:', df_hcf.info())

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_hcf.columns = df_hcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Drop rows with missing UTS or Stress amplitude (critical for modeling)
df_hcf_clean = df_hcf.dropna(subset=['Stress amplitude (MPa)', 'UTS (MPa)', 'Number of cycles'], how= 'all')

# Step 2: Feature Engineering
# ---------------------------

# 1. Normalized stress (w.r.t. UTS)
df_hcf_clean['Normalized_Stress'] = df_hcf_clean['Stress amplitude (MPa)'] / df_hcf_clean['UTS (MPa)']

# 2. Log cycles to failure
df_hcf_clean['log_cycles'] = np.log10(df_hcf_clean['Number of cycles'])

# 3. Stress Ratio (R) from summary
df_hcf_clean['Stress_Ratio'] = df_hcf_clean['R']

# 4. Goodman-adjusted stress (optional if you want to account for mean stress effects)
# Requires Max. stress and UTS (skip rows without Max stress)
df_hcf_clean['Goodman_Adjusted_Stress'] = np.where(
    pd.notna(df_hcf_clean['Max. stress (MPa)']),
    df_hcf_clean['Stress amplitude (MPa)'] / (1 - (df_hcf_clean['Max. stress (MPa)'] / df_hcf_clean['UTS (MPa)'])),
    np.nan
)

# 5. Frequency binning (optional feature)
# df_hcf_clean['Frequency_Binned'] = pd.cut(df_hcf_clean['Frequency (Hz)'], bins=[0, 10, 50, 1000], labels=['Low', 'Medium', 'High']) # Maybe later if neccessary

# 6. Fatigue ratio (optional for analysis)
df_hcf_clean['Fatigue_Ratio'] = df_hcf_clean['Fatigue ratio']

# -------------------
# View sample results
# -------------------
print(df_hcf_clean[['ID', 'Stress amplitude (MPa)', 'UTS (MPa)', 'Normalized_Stress', 'log_cycles', 'Stress_Ratio']].head())
print(df_hcf_clean.info())

# save_df = df_hcf_clean.to_csv('hcf_clean.csv')

# ---------------------------------------------
# STEP 3: EDA - Correlation Matrix + Scatterplots
# ---------------------------------------------

# Convert Frequency to numeric in case of bad entries
df_hcf_clean["Frequency (Hz)"] = pd.to_numeric(df_hcf_clean["Frequency (Hz)"], errors="coerce")

# Drop rows with missing frequencies
df_hcf_clean = df_hcf_clean.dropna(subset=["Frequency (Hz)"])

# Create frequency bins
df_hcf_clean['Frequency_Binned'] = pd.cut(
    df_hcf_clean['Frequency (Hz)'],
    bins=[0, 10, 100, 1000, 10000],
    labels=['Low (<10Hz)', 'Medium (10-100Hz)', 'High (100-1kHz)', 'Very High (>1kHz)']
)

# A. Correlation Matrix
numeric_cols = df_hcf_clean.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix - HCF Features")
plt.tight_layout()
plt.show()

# B. Scatter Plot: Stress Amplitude vs Log Cycles
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_hcf_clean, x="Stress amplitude (MPa)", y="log_cycles", hue="Stress_Ratio", palette="viridis")
plt.title("Stress Amplitude vs. Log Cycles to Failure")
plt.xlabel("Stress Amplitude (MPa)")
plt.ylabel("log10(Cycles to Failure)")
plt.tight_layout()
plt.show()

# C. Scatter Plot: Normalized Stress vs Log Cycles
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_hcf_clean, x="Normalized_Stress", y="log_cycles", hue="Frequency_Binned", palette="plasma")
plt.title("Normalized Stress vs. Log Cycles to Failure")
plt.xlabel("Normalized Stress (σ/UTS)")
plt.ylabel("log10(Cycles to Failure)")
plt.tight_layout()
plt.show()

# D. Grouped Stats by Frequency Bins
grouped_freq_stats = df_hcf_clean.groupby("Frequency_Binned")["Number of cycles"].agg(["count", "mean", "std", "min", "max"])
print("\nGrouped Fatigue Life by Frequency Bins:\n")
print(grouped_freq_stats)

# ---------------------------------------------
# STEP 4: Fit a Basquin-like Model (log-log linear fit)
# ---------------------------------------------

# Drop missing values in required columns
fit_data = df_hcf_clean.dropna(subset=["Normalized_Stress", "log_cycles"])

# Create log stress
fit_data["log_stress"] = np.log10(fit_data["Normalized_Stress"])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(fit_data["log_stress"], fit_data["log_cycles"])

# Line for plotting
x_vals = np.linspace(fit_data["log_stress"].min(), fit_data["log_stress"].max(), 100)
y_vals = intercept + slope * x_vals

# Plot Basquin model fit
plt.figure(figsize=(8, 6))
sns.scatterplot(x=fit_data["log_stress"], y=fit_data["log_cycles"], alpha=0.6, label="Data")
plt.plot(x_vals, y_vals, color='red', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
plt.xlabel("log10(Normalized Stress)")
plt.ylabel("log10(Cycles to Failure)")
plt.title("Basquin Model: Stress-Life Fit")
plt.legend()
plt.tight_layout()
plt.show()

# Display fit summary
print("\nBasquin Model Fit Summary:")
print(f"  Slope (b): {slope:.4f}")
print(f"  Intercept (log σ_f'): {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  p-value: {p_value:.4e}")
print(f"  Standard Error: {std_err:.4f}")


