from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the cleaned HCF dataset
df_hcf_clean = pd.read_csv("hcf_clean.csv")

# Clean DF
df_hcf_clean['Frequency (Hz)'] = df_hcf_clean['Frequency (Hz)'].astype(float) # Convert object type to float
df_model_clean = df_hcf_clean[[ "Stress amplitude (MPa)",
                                "Tensile YS (MPa)",
                                "UTS (MPa)",
                                "R",
                                "Frequency (Hz)",
                                "Normalized_Stress",
                                "Stress_Ratio",
                                "Goodman_Adjusted_Stress", # Why is it included, doesn't have a strong correlation. Also has the most missing values. Maybe un-include it
                                "Fatigue_Ratio", # Same
                                "log_cycles"]] 
print(df_model_clean.info())
df_model_clean.dropna(inplace= True)
print(df_model_clean.info())

# split your data into X_train, X_test, y_train, y_test
# Define features (X) and target (y)
X = df_model_clean.drop(columns=["log_cycles"])
y = df_model_clean["log_cycles"]

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Confirm the shape
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred), 
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

# Evaluate models
lr_metrics = evaluate_model(y_test, y_pred_lr)
rf_metrics = evaluate_model(y_test, y_pred_rf)

# Print results
results_df = pd.DataFrame([lr_metrics, rf_metrics], index=["Linear Regression", "Random Forest"])
print(results_df)


# Residuals
# Plot residuals
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(12, 5))

# Linear Regression residuals
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_lr, y=residuals_lr)
plt.axhline(0, color='red', linestyle='--')
plt.title("Linear Regression Residuals")
plt.xlabel("Predicted log(Cycles)")
plt.ylabel("Residuals")

# Random Forest residuals
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_rf, y=residuals_rf)
plt.axhline(0, color='red', linestyle='--')
plt.title("Random Forest Residuals")
plt.xlabel("Predicted log(Cycles)")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()


# Retrain Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importances
importances = rf_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)

# Random forest feature importance chart
# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# Look through
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"Model": model_name, "R²": r2, "MAE": mae, "RMSE": rmse}

results = []
results.append(evaluate_model(y_test_poly, y_pred_lr_poly, "Poly Linear Regression"))
results.append(evaluate_model(y_test_rf, y_pred_rf_improved, "Improved Random Forest"))

results_df = pd.DataFrame(results)
print(results_df)
