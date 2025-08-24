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


