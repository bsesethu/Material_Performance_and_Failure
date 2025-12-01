# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools

###########################################
# Exploratory Data Analysis Pipeline
# =========================


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

# 6. Univariate Analysis (Distribution of numerical features) #NOTE already done in cleaning_LCF.py Here it's just more focused
# numeric_cols = df.select_dtypes(include=np.number).columns
# for col in numeric_cols:
#     plt.figure()
#     sns.histplot(df[col], kde=True, bins=30, color="skyblue")
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()

# 7. Categorical Feature Counts
categorical_cols = df.select_dtypes(exclude=np.number).columns
numeric_cols = df.select_dtypes(include=np.number).columns
for col in categorical_cols:
    plt.figure()
    sns.countplot(data=df, x=col, palette="Set2")
    plt.title(f"Count of {col}")
    plt.xticks(rotation=45)
    plt.show()

# 8. Bivariate Analysis (Numerical vs Target)
# Replace 'Target' with your target column name (e.g., 'FinalScore' or 'Grade')
target = 'Number of cycles'  # Change as needed
for col in numeric_cols:
    if col != target:
        plt.figure()
        sns.scatterplot(data=df, x=col, y=target, hue=target, palette="coolwarm")
        plt.title(f"{col} vs {target}")
        plt.show()

# 9. Categorical vs Target (Boxplots)
for col in categorical_cols:
    plt.figure()
    sns.boxplot(data=df, x=col, y=target, palette="Set3")
    plt.title(f"{col} vs {target}")
    plt.xticks(rotation=45)
    plt.show()

# 10. Correlation Matrix (Numerical)
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 11. Pairplot (for top correlated features)
top_corr = corr_matrix[target].abs().sort_values(ascending=False)[1:6].index
sns.pairplot(df[top_corr.tolist() + [target]], diag_kind='kde', corner=True)
plt.show()

# 12. Statistical Relationships (Optional)
for col in numeric_cols:
    if col != target:
        pearson_corr, _ = pearsonr(df[col], df[target])
        spearman_corr, _ = spearmanr(df[col], df[target])
        print(f"{col}: Pearson = {pearson_corr:.3f}, Spearman = {spearman_corr:.3f}")


