import pandas as pd
import numpy as np

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

print('\nHCF Info:', df_hcf.info())

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