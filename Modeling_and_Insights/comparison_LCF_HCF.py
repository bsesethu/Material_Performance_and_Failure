import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


