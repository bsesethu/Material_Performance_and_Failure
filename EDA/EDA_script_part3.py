import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools

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

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

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
plt.show()

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

# Change 'Number of cycles' column values to log10
df_lcf['Log10_Number of cycles'] = np.log10(df_lcf['Number of cycles'])

# Visualization: Boxplot + Swarmplot for Grain Size vs Fatigue Life
if 'Grain_Category' in df_lcf.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Grain_Category', y='Log10_Number of cycles', data=df_lcf, palette='Set2') # Need to tinker, the outliers are makeing the chart look bad
    sns.swarmplot(x='Grain_Category', y='Log10_Number of cycles', data=df_lcf, color='black', alpha=0.6)
    plt.title("Fatigue Life by Grain Size Category")
    plt.show()

# Visualization: Pairplot for key features      # Later maybe add this chart
# key_features = ['Number of cycles', 'Stress', 'Grain_Size', 'Norm_Fatigue_Life', 'Energy_Param']
# available_features = [col for col in key_features if col in df_lcf.columns]
# sns.pairplot(df_lcf[available_features], diag_kind='kde')
# plt.suptitle("Pairwise Relationships", y=1.02)
# plt.show()

