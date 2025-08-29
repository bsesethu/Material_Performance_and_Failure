from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures # For adding polynomial features to linear regression model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #NOTE These are all ACTUAL values

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


# Feature importances (Random Forest) plot
importances = rf_model.feature_importances_ # I don't know what this is
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

# Model improvements
# 2. Define Features (X) and Target (y) [1. Was loading dataset]
# Based on your setup, we select the relevant features
features = ['Stress amplitude (MPa)', 'Frequency (Hz)', 'R']
X = df_hcf_clean[features]
y = df_hcf_clean['log_cycles']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Polynomial Regression model using a pipeline
# The pipeline automates the workflow:
# a. StandardScaler: Scales the data (important for polynomial features).
# b. PolynomialFeatures: Creates polynomial and interaction features.
# c. LinearRegression: The final regression model.

# We'll start with degree=2, which is a common choice.
degree = 5

# Create the pipeline
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression()
)
# Train the model on the training data
poly_model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred_poly = poly_model.predict(X_test)

# 6. (Optional) Evaluate the model's performance
lr_metrics = evaluate_model(y_test, y_pred_poly)
# Print results
results_df = pd.DataFrame([lr_metrics], index=["Linear Regression"])
print('\nPolynomial infused Linear Regression model')
print(f"Polynomial Regression Model (degree={degree})")
print(results_df)
print("This is the best result the Linear regression model can have, and it's still not as good as the Random Forest model")


    # Improving Random Forrest model using Scaled Features
X = df_model_clean.drop(columns=["log_cycles"])
y = df_model_clean["log_cycles"]

# Normalize and Add Domain Features
X["Stress×Freq"] = X["Stress amplitude (MPa)"] * X["Frequency (Hz)"]
X["Stress/UTS"] = X["Stress amplitude (MPa)"] / (X["UTS (MPa)"] + 1e-6)
X["Fatigue×NormStress"] = X["Fatigue_Ratio"] * X["Normalized_Stress"]

# Normalize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest on improved features
rf_model_scaled = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_scaled.fit(X_train_scaled, y_train_scaled)
y_pred_rf_scaled = rf_model_scaled.predict(X_test_scaled)

# Evaluate
results = [evaluate_model(y_test_scaled, y_pred_rf_scaled)]
results_df = pd.DataFrame(results, index= ['Random Forest'])
print('\nAfter using Scaled Features')
print(results_df)
print("Error values remain the same, in fact they're very slightly worse")

    # Different appraoach to improve Random Forrest
# Define and train the model
best_rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)
best_rf_model.fit(X_train_scaled, y_train_scaled)
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test_scaled, y_pred_best_rf)
mae = mean_absolute_error(y_test_scaled, y_pred_best_rf)
rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_best_rf))

print('\nAfter using Hyperparameter Tuning')
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print('Again results are ever so slightly worse. Use the original model.')
    # If your evaluation metrics (R², MAE, RMSE) aren’t budging much, it suggests:
    # The model has already captured most of the predictive patterns in the data.
    # Random Forest has likely maxed out its benefit on this dataset.

