import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import itertools
from scipy.stats import linregress


#============================================================================================
# LCF Analysis ('cleaning_LCF.py', 'EDA_script_LCF_EDA.py'  and `EDA_script_LCF_Insights.py`)
#======================================================================================================

# From `EDA_script_LCF_Insights.py`

# Load the Excel file
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')

# Parse a specific sheet
df_lcf = xls.parse('LCF individual dataset') 

# Drop NaN and column named rows
print(df_lcf.info())
df_lcf = df_lcf.dropna()
print(df_lcf.head())
print(df_lcf.info())

# Must ensure that column names are exact, errors were showing up. There was a space at the end of 'Total strain amplitude '
# Check exact column names
print(df_lcf.columns.tolist())

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Verify again
print(df_lcf.columns.tolist())

# Dataset Overview and Statistical Summary

# === STEP 1: Dataset Overview ===
print("\nShape of dataset:", df_lcf.shape)
print("\nColumn names:", df_lcf.columns.tolist())
print("\nData types:\n", df_lcf.dtypes)
print("\nMissing values per column:\n", df_lcf.isnull().sum())

# === STEP 2: Statistical Summary ===
print("\nStatistical Summary:\n", df_lcf.describe())

# === STEP 3: Quick Visuals for Distributions ===
numeric_cols = df_lcf.select_dtypes(include='number').columns

plt.figure(figsize=(14, 8))
df_lcf[numeric_cols].hist(bins=20, figsize=(14, 8))
plt.suptitle("Histograms of Numeric Columns", fontsize=16)
plt.show()

# === Group Analysis (Optional) ===
# If 'ID' or 'Material' column exists, group by and see average fatigue life
if 'ID' in df_lcf.columns:
    avg_cycles = df_lcf.groupby('ID')["Number of cycles"].mean().sort_values(ascending=False)
    print("\nAverage Number of Cycles per ID:\n", avg_cycles)

# Save cleaned DF as CSV
df_lcf.to_csv('cleaned_student_data.csv')


###########################################
# Exploratory Data Analysis Pipeline
# =========================
# From 'EDA_script_LCF_EDA.py'

# Optional: for statistical tests
from scipy.stats import pearsonr, spearmanr

# Configure plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 2. Load the cleaned dataset
# Replace with your cleaned file
df = pd.read_csv("cleaned_student_data.csv")

# Drop column
df.drop('Unnamed: 0', axis= 1, inplace= True)

# 1. Categorical Feature Counts
categorical_cols = df.select_dtypes(exclude=np.number).columns
numeric_cols = df.select_dtypes(include=np.number).columns
for col in categorical_cols:
    plt.figure()
    sns.countplot(data=df, x=col, palette="Set2")
    plt.title(f"Count of {col}")
    plt.xticks(rotation=45)
    plt.show()
    # There are no categorical columns so this block of code will return None

# 2. Bivariate Analysis (Numerical vs Target)
# Replace 'Target' with your target column name (e.g., 'FinalScore' or 'Grade')
target = 'Number of cycles'  # Change as needed
for col in numeric_cols:
    if col != target:
        plt.figure()
        sns.scatterplot(data=df, x=col, y=target, hue=target, palette="coolwarm")
        plt.title(f"{col} vs {target}")
        plt.show()

# 3. Categorical vs Target (Boxplots)
for col in categorical_cols:
    plt.figure()
    sns.boxplot(data=df, x=col, y=target, palette="Set3")
    plt.title(f"{col} vs {target}")
    plt.xticks(rotation=45)
    plt.show()
    # Again no categorical datapoints

# 4. Correlation Matrix (Numerical)
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 5. Pairplot (for top correlated features)
top_corr = corr_matrix[target].abs().sort_values(ascending=False)[1:6].index
plt.figure(figsize=(18, 14))
sns.pairplot(df[top_corr.tolist() + [target]], diag_kind='kde', corner=True)
plt.tight_layout()
plt.show()

# 6. Statistical Relationships (Optional)
for col in numeric_cols:
    if col != target:
        pearson_corr, _ = pearsonr(df[col], df[target])
        spearman_corr, _ = spearmanr(df[col], df[target])
        print(f"{col}: Pearson = {pearson_corr:.3f}, Spearman = {spearman_corr:.3f}")


#================================================================
# LCF Deeper insights and Feature engineering
#===================================================================

# From `EDA_script_LCF_Insights.py`
# -------------------------------
# Load datasets
# -------------------------------
# Load the Excel file
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')

# Parse a specific sheet
df1 = xls.parse('LCF summary') 
df2 = xls.parse('LCF individual dataset')

# Merge datasets on ID
df_lcf = pd.merge(df1, df2, on="ID", how="left")

# Add an elastic modulus column to the DF
modulus = []
for row in df_lcf['ID']:
    if row in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]: # 3 is a different alloy but has the same modulus
        modulus.append(205e9)
    elif row in [11, 12]:
        modulus.append(180e9) 
    elif row == 13:
        modulus.append(165e9)
df_lcf['Elastic Modulus'] = pd.DataFrame({'Elastic Modulus': modulus})

print('Summary core info:')
print(df_lcf.head())
print(df_lcf.info())

# === Dataset Overview ===
print("\nShape of dataset:", df_lcf.shape)
print("\nColumn names:", df_lcf.columns.tolist())
print("\nData types:\n", df_lcf.dtypes)
print("\nMissing values per column:\n", df_lcf.isnull().sum())

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Drop NaN columns and rows
df_lcf.drop(columns= ['Frequency'], inplace= True)
df_lcf.dropna(inplace= True)

# =========================================================
# PART 1: FEATURE ENGINEERING
# =========================================================

# 1. Calculate total strain ratio (Elastic / Total)
df_lcf["Elastic_to_Total_Strain"] = (
    df_lcf["Elastic strain amplitude"] / df_lcf["Total strain amplitude"]
)

# 2. Calculate plastic to total strain ratio
df_lcf["Plastic_to_Total_Strain"] = (
    df_lcf["Plastic strain amplitude"] / df_lcf["Total strain amplitude"]
)

# 3. Estimate stress amplitude using Elastic strain amplitude and Tensile YS (if available)
#    This assumes elastic modulus is approximated from Tensile YS / Yield strain.
#    If Tensile YS missing, we leave as NaN.
df_lcf["Estimated_Stress_Amplitude"] = np.where(
    pd.notna(df_lcf["Tensile YS (MPa)"]),
    df_lcf["Elastic strain amplitude"] * df_lcf["Elastic Modulus"], 
    np.nan
)

# 4. Fatigue strength coefficient (approximation)
#    Uses UTS if Tensile YS is missing
df_lcf["Fatigue_Strength_Coeff"] = df_lcf[["Tensile YS (MPa)", "UTS (MPa)"]].max(axis=1) # Not sure how accurate this is, how sound the logic is

# 5. Log-transform cycles for regression analysis
df_lcf["log_cycles"] = np.log10(df_lcf["Number of cycles"])

# 6. Create strain energy density approximation
#    Strain energy density ≈ 0.5 × Stress amplitude × Total strain amplitude
df_lcf["Strain_Energy_Density"] = 0.5 * df_lcf["Estimated_Stress_Amplitude"] * df_lcf["Total strain amplitude"]

# Overview after cleaning and feature engineering
print("\nShape of dataset:", df_lcf.shape)
print("\nColumn names:", df_lcf.columns.tolist())
print("\nData types:\n", df_lcf.dtypes)
print("\nMissing values per column:\n", df_lcf.isnull().sum())

# Save as csv
df_lcf.to_csv('lcf_clean.csv') # Not sure if this will result is simalar performance of the model (Lost about half the rows)


# =========================================================
# PART 2: DEEPER STATISTICAL ANALYSIS
# =========================================================

# 1. Correlation matrix for numeric columns
num_cols = df_lcf.select_dtypes(include=[np.number])
corr_matrix = num_cols.corr()

# 2. Grouped fatigue life stats by testing temperature
group_temp_stats = df_lcf.groupby("Tesing temperature (K)")["Number of cycles"].agg(["mean", "std", "min", "max"])

# 3. Relationship between strain amplitudes and fatigue life
strain_life_corr = df_lcf[["Total strain amplitude", "Elastic strain amplitude", "Plastic strain amplitude", "Number of cycles"]].corr()

# 4. Effect of tensile properties on fatigue life
mechanical_life_corr = df_lcf[["Tensile YS (MPa)", "UTS (MPa)", "Uniform elongation", "Number of cycles"]].corr()

# -------------------------------
# Output key results
# -------------------------------
print("Correlation Matrix (Numeric Features):\n", corr_matrix, "\n")
print("Fatigue Life Stats by Testing Temperature:\n", group_temp_stats, "\n")
print("Strain-Life Correlation:\n", strain_life_corr, "\n")
print("Mechanical Properties vs Life Correlation:\n", mechanical_life_corr, "\n")

# 1. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_lcf.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

# Put grain sizes in categories
df_lcf['Grain size (um)'] = df_lcf['Grain size (um)'].astype(int)
# Grain size category
if 'Grain size (um)' in df_lcf.columns:
    q1, q2 = df_lcf['Grain size (um)'].quantile([0.33, 0.66])
    def grain_category(gs):
        if gs <= q1:
            return 'Small'
        elif gs <= q2:
            return 'Medium'
        else:
            return 'Large'
    df_lcf['Grain_Category'] = df_lcf['Grain size (um)'].apply(grain_category)

# ANOVA: Does grain size category affect fatigue life?
if 'Grain_Category' in df_lcf.columns:
    groups = [group['Number of cycles'].values for name, group in df_lcf.groupby('Grain_Category')]
    anova_result = stats.f_oneway(*groups)
    print("\nANOVA Result (Grain Size vs Fatigue Life):")
    print(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4e}")

# Pairwise t-tests between grain categories (if 3 groups exist)
if 'Grain_Category' in df_lcf.columns:
    categories = df_lcf['Grain_Category'].unique()
    for cat1, cat2 in itertools.combinations(categories, 2):
        group1 = df_lcf[df_lcf['Grain_Category'] == cat1]['Number of cycles']
        group2 = df_lcf[df_lcf['Grain_Category'] == cat2]['Number of cycles']
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        print(f"T-test {cat1} vs {cat2}: t-stat = {t_stat:.4f}, p-value = {p_val:.4e}")

# Change 'Number of cycles' column values to log10_Number_of_cycles
df_lcf['Log10_Number of cycles'] = np.log10(df_lcf['Number of cycles'])

# Visualization: Boxplot + Swarmplot for Grain Size vs Fatigue Life
if 'Grain_Category' in df_lcf.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Grain_Category', y='Log10_Number of cycles', data=df_lcf, palette='Set2')
    sns.swarmplot(x='Grain_Category', y='Log10_Number of cycles', data=df_lcf, color='black', alpha=0.6)
    plt.title("Fatigue Life by Grain Size Category")
    plt.show()

# Visualization: Pairplot for key features      # Later maybe add this chart
# key_features = ['Number of cycles', 'Stress', 'Grain_Size', 'Norm_Fatigue_Life', 'Energy_Param']
# available_features = [col for col in key_features if col in df_lcf.columns]
# sns.pairplot(df_lcf[available_features], diag_kind='kde')
# plt.suptitle("Pairwise Relationships", y=1.02)
# plt.show()

# =========================
# 3) FATIGUE CURVE MODELING (LCF)
#    Coffin–Manson split:
#    ε_p,a = ε_f' * (2N)^c      (plastic term)
#    ε_e,a = (σ_f'/E) * (2N)^b  (elastic/Basquin-like term)
#    ε_t,a ≈ ε_p,a + ε_e,a
# =========================

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# --- Map your column names here if needed ---  Not needed
# colmap = {
#     "Number of cycles" : "N",
#     "Total strain amplitude" : "eps_total",
#     "Plastic strain amplitude" : "eps_plastic",
#     "Elastic strain amplitude" : "eps_elastic",
#     # optional if present
#     "Elastic_modulus" "E": "E",
#     "Grain size (um)" : "grain"
# }
# Create a normalized view with safe names
# df_fit = df_lcf.rename(columns=colmap).copy()
# print(df_fit.info())

# Keep only positive, finite rows needed for log fits
df_lcf = df_lcf.replace([np.inf, -np.inf], np.nan)
df_lcf = df_lcf.dropna(subset=["Number of cycles", "Total strain amplitude", "Plastic strain amplitude", "Elastic strain amplitude"])
df_lcf = df_lcf[(df_lcf["Number of cycles"] > 0) & (df_lcf["Plastic strain amplitude"] > 0) & (df_lcf["Elastic strain amplitude"] > 0)]

# Helper: linear fit on log10(y) = a + b*log10(x)
def loglog_fit(x, y):
    xlog = np.log10(x)
    ylog = np.log10(y)
    # polyfit degree 1 -> slope, intercept
    slope, intercept = np.polyfit(xlog, ylog, 1)
    # R^2 on log space
    yhat = intercept + slope * xlog
    ss_res = np.sum((ylog - yhat)**2)
    ss_tot = np.sum((ylog - np.mean(ylog))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2

# Fit plastic term: ε_p,a = ε_f' * (2N)^c
c_slope, c_inter, c_r2 = loglog_fit(2*df_lcf["Number of cycles"].values, df_lcf["Plastic strain amplitude"].values)
# Fit elastic term: ε_e,a = K * (2N)^b  where K = σ_f'/E (lumped constant)
b_slope, b_inter, b_r2 = loglog_fit(2*df_lcf["Number of cycles"].values, df_lcf["Elastic strain amplitude"].values)

# Extract parameters
# In log10 space: log10(ε_p,a) = log10(ε_f') + c * log10(2N)
eps_f_prime = 10**(c_inter)        # plastic fatigue ductility coefficient (approx)
c_exp = c_slope                    # plastic fatigue ductility exponent

# Elastic/Basquin-like term: log10(ε_e,a) = log10(K) + b*log10(2N)
K_elastic = 10**(b_inter)          # K = σ_f'/E (lumped)
b_exp = b_slope

print("=== Coffin–Manson Split Fits (LCF) ===")
print(f"Plastic term:   ε_plastic,a = {eps_f_prime:.3e} * (2N)^{c_exp:.3f}   | R^2_log = {c_r2:.3f}") # Plastic and elastic amplitude
print(f"Elastic term:   ε_elastic,a = {K_elastic:.3e} * (2N)^{b_exp:.3f}     | R^2_log = {b_r2:.3f}")

# Plot ε_p,a and ε_e,a vs 2N (log-log) with fit lines
N_grid = np.logspace(np.log10((2*df_lcf["Number of cycles"]).min()),
                     np.log10((2*df_lcf["Number of cycles"]).max()), 200)

eps_p_pred = eps_f_prime * (N_grid**c_exp)
eps_e_pred = K_elastic   * (N_grid**b_exp)
eps_t_pred = eps_p_pred + eps_e_pred

plt.figure(figsize=(10,7))
plt.scatter(2*df_lcf["Number of cycles"], df_lcf["Plastic strain amplitude"], s=30, alpha=0.7, label="ε_plastic,a data", edgecolor='none')
plt.scatter(2*df_lcf["Number of cycles"], df_lcf["Elastic strain amplitude"], s=30, alpha=0.7, label="ε_elastic,a data", edgecolor='none')
plt.plot(N_grid, eps_p_pred, linewidth=2, label="ε_plastic,a fit")
plt.plot(N_grid, eps_e_pred, linewidth=2, label="ε_elastic,a fit")
plt.xscale('log'); plt.yscale('log')
plt.xlabel("2N (reversals)"); plt.ylabel("Strain amplitude")
plt.title("LCF Coffin–Manson Split Fits")
plt.legend()
plt.tight_layout()
plt.show()

# Compare total strain amplitude to the sum of fitted components
plt.figure(figsize=(10,7))
plt.scatter(2*df_lcf["Number of cycles"], df_lcf["Total strain amplitude"], s=30, alpha=0.7, label="ε_total,a data", edgecolor='none') # a is amplitude
plt.plot(N_grid, eps_t_pred, linewidth=2, label="ε_total,a model = ε_plastic,a + ε_elastic,a")
plt.xscale('log'); plt.yscale('log')
plt.xlabel("2N (reversals)"); plt.ylabel("Total strain amplitude (ε_total,a)")
plt.title("Total Strain vs Reversals with Coffin–Manson Composite")
plt.legend()
plt.tight_layout()
plt.show()


# save_df = df_lcf.to_csv('lcf_clean.csv')

# LCF cleaning and EDA done

#==============================================================
# HCF Analysis (`EDA_script_HCF.py`)
#=================================================

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

# === Dataset Overview ===
print("\nShape of dataset:", df_hcf_clean.shape)
print("\nColumn names:", df_hcf_clean.columns.tolist())
print("\nData types:\n", df_hcf_clean.dtypes)
print("\nMissing values per column:\n", df_hcf_clean.isnull().sum())

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

# -------------------
# View sample results
# -------------------
print(df_hcf_clean[['ID', 'Stress amplitude (MPa)', 'UTS (MPa)', 'Normalized_Stress', 'log_cycles', 'Stress_Ratio']].head())
print(df_hcf_clean.info())

# === Dataset Overview === After cleaning
print("\nShape of dataset:", df_hcf_clean.shape)
print("\nColumn names:", df_hcf_clean.columns.tolist())
print("\nData types:\n", df_hcf_clean.dtypes)
print("\nMissing values per column:\n", df_hcf_clean.isnull().sum())

save_df = df_hcf_clean.to_csv('hcf_clean.csv')

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

# HCF cleaning and EDA done
# Up to end of EDA
# =============================================================================================

