import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the cleaned datasets
df_lcf_clean = pd.read_csv("lcf_clean.csv")
df_hcf_clean = pd.read_csv("hcf_clean.csv")

# Prepare LCF (strain-based)
df_lcf_plot = df_lcf_clean[["eps_total", "log_cycles"]].dropna().copy()
df_lcf_plot["Metric_Type"] = "Strain Amplitude (LCF)"
df_lcf_plot.rename(columns={"eps_total": "x_value"}, inplace=True)

# Prepare HCF (stress-based)
df_hcf_plot = df_hcf_clean[["Normalized_Stress", "log_cycles"]].dropna().copy()
df_hcf_plot["Metric_Type"] = "Normalized Stress (HCF)"
df_hcf_plot.rename(columns={"Normalized_Stress": "x_value"}, inplace=True)

# Combine
df_compare = pd.concat([df_lcf_plot, df_hcf_plot], ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_compare, x="x_value", y="log_cycles", hue="Metric_Type", alpha=0.6)
plt.title("Fatigue Behavior: LCF (Strain) vs HCF (Stress)")
plt.xlabel("Strain Amplitude (LCF) or Normalized Stress (HCF)")
plt.ylabel("log10(Cycles to Failure)")
plt.legend(title="Metric Type")
plt.tight_layout()
plt.show() # Is this a useful plot, I'm not so sure


print('Info')
print(df_lcf_clean.info(), df_hcf_clean.info())
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot LCF data (Strain vs. log_cycles)
sns.regplot(x='log_cycles', y='eps_total', data=df_lcf_clean, ax=ax1,
            scatter_kws={'alpha':0.6, 's':80, 'color':'#0077b6'},
            line_kws={'color':'#d62728', 'linewidth':3})
ax1.set_title('LCF: Strain-Based Behavior (Inconel X-750)', fontsize=16, pad=20) # Inconel is NiCr+ alloy
ax1.set_xlabel('Log (Cycles to Failure)', fontsize=12)
ax1.set_ylabel('Strain Amplitude', fontsize=12)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot HCF data (Stress vs. log_cycles)
sns.regplot(x='log_cycles', y='Stress amplitude (MPa)', data=df_hcf_clean, ax=ax2,
            scatter_kws={'alpha':0.6, 's':80, 'color':'#2ca02c'},
            line_kws={'color':'#ff7f0e', 'linewidth':3})
ax2.set_title('HCF: Stress-Based Behavior (Two Ni-Superalloys)', fontsize=16, pad=20)
ax2.set_xlabel('Log (Cycles to Failure)', fontsize=12)
ax2.set_ylabel('Stress Amplitude (MPa)', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()


#--------------------------------
# Model comparisons
#----------------------------------
# ========== Load Cleaned Data ==========
df_hcf = pd.read_csv("hcf_clean.csv")
df_lcf = pd.read_csv("lcf_clean.csv")

# ========== Feature Engineering ==========
df_hcf['Frequency (Hz)'] = df_hcf['Frequency (Hz)'].astype(float) # Convert object type to float
df_model_clean = df_hcf[[ "Stress amplitude (MPa)",
                                "Tensile YS (MPa)",
                                "UTS (MPa)",
                                "R",
                                "Frequency (Hz)",
                                "Normalized_Stress",
                                "Stress_Ratio",
                                "Goodman_Adjusted_Stress", # Why is it included, doesn't have a strong correlation. Also has the most missing values. Maybe un-include it
                                "Fatigue_Ratio", # Same
                                "log_cycles"]] 
df_model_clean.dropna(inplace= True)
print(df_model_clean.info())

# split your data into X_train, X_test, y_train, y_test
# Define features (X) and target (y)
X_hcf = df_model_clean.drop(columns=["log_cycles"])
y_hcf = df_model_clean["log_cycles"]

# Change some column names
df_lcf['strain_total'] = df_lcf['eps_total']
df_lcf['strain_plastic'] = df_lcf['eps_plastic']
df_lcf['strain_elastic'] = df_lcf['eps_elastic']

# LCF
features = [
    "Max. strain", "strain_total", "strain_plastic", "strain_elastic", "Elastic_to_Total_Strain", 
    "Plastic_to_Total_Strain" # "Strain×Temp", "Strain/Temp", "Strain Ratio" # Add later mb
]
target = "log_cycles"

X_lcf = df_lcf[features].copy()
y_lcf = df_lcf[target].copy()

# ========== Scale + Split ==========
# scaler_hcf = StandardScaler()
# X_hcf_scaled = scaler_hcf.fit_transform(X_hcf)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hcf, y_hcf, test_size=0.2, random_state=42) 

scaler_lcf = StandardScaler()
X_lcf_scaled = scaler_lcf.fit_transform(X_lcf)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lcf_scaled, y_lcf, test_size=0.2, random_state=42)

# ========== Train Random Forests ==========
rf_hcf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_hcf.fit(X_train_h, y_train_h)
y_pred_h = rf_hcf.predict(X_test_h)

rf_lcf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lcf.fit(X_train_l, y_train_l)
y_pred_l = rf_lcf.predict(X_test_l)

# ========== Evaluate ==========
def evaluate(y_true, y_pred, name):
    return {
        "Dataset": name,
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

results = [
    evaluate(y_test_h, y_pred_h, "HCF"),
    evaluate(y_test_l, y_pred_l, "LCF")
]
results_df = pd.DataFrame(results)
print("\n🔍 Model Performance Comparison:\n")
print(results_df)

# ========== Compare Feature Importances ==========
importances_hcf = pd.Series(rf_hcf.feature_importances_, index=X_hcf.columns).sort_values(ascending=False)
importances_lcf = pd.Series(rf_lcf.feature_importances_, index=X_lcf.columns).sort_values(ascending=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
importances_hcf.plot(kind='barh', ax=axes[0], title="HCF Feature Importance")
importances_lcf.plot(kind='barh', ax=axes[1], title="LCF Feature Importance")
axes[0].invert_yaxis()
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()
