import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
xls = pd.ExcelFile('Fatigue database of High-entropy alloys.xlsx')

# Parse a specific sheet
df_lcf = xls.parse('LCF individual dataset') # Parsing is act of breaking down data to useful components

# Drop NaN and column named rows
df_lcf = df_lcf.dropna()
print(df_lcf.head(20))
print(df_lcf.info())

# Must ensure that column names a exact, errors were showing up. There was a space at the end of 'Total strain amplitude '
# Check exact column names
print(df_lcf.columns.tolist())

# Clean up column names: strip spaces and replace multiple spaces with single space NOTE Really important function
df_lcf.columns = df_lcf.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Verify again
print(df_lcf.columns.tolist())

# Dataset Overview and Statistical Summary

# === STEP 1: Dataset Overview ===
print("Shape of dataset:", df_lcf.shape)
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

# === STEP 4: Correlation Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df_lcf.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# === STEP 5: Scatterplots for Fatigue Analysis ===

# Define pairs we care about
pairs_to_plot = [
    ("Total strain amplitude", "Number of cycles"),
    ("Max. strain", "Number of cycles"),
    ("Plastic strain amplitude", "Number of cycles"),
    ("Elastic strain amplitude", "Number of cycles")
]

plt.figure(figsize=(16, 12)) #NOTE Great 'for' loop example
for i, (x_col, y_col) in enumerate(pairs_to_plot, 1):
    # plt.subplot(2, 2, i)
    plt.scatter(df_lcf[x_col], df_lcf[y_col], alpha=0.6, edgecolor='k')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.yscale('log')  # Log scale is common for fatigue cycles
    plt.tight_layout()
    plt.show()

# === STEP 6: Group Analysis (Optional) ===
# If 'ID' or 'Material' column exists, group by and see average fatigue life
if 'ID' in df_lcf.columns:
    avg_cycles = df_lcf.groupby('ID')["Number of cycles"].mean().sort_values(ascending=False)
    print("\nAverage Number of Cycles per ID:\n", avg_cycles)

# Save cleaned DF as CSV
df_lcf.to_csv('cleaned_student_data.csv')