import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load LCF data
df_lcf = pd.read_csv("lcf_clean.csv")
print(df_lcf.info())

# ========== Define Features & Target ==========
features = [
    "Max. strain", "eps_total", "eps_plastic", "eps_elastic", "Elastic_to_Total_Strain", 
    "Plastic_to_Total_Strain" # "Strain×Temp", "Strain/Temp", "Strain Ratio" # Add later mb
]
target = "log_cycles"

X = df_lcf[features].copy()
y = df_lcf[target].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== Step 3a: Linear Regression ==========
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ========== Step 3b: Random Forest ==========
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ========== Step 4: Evaluation Function ==========
def evaluate(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

results = [
    evaluate(y_test, y_pred_lr, "Linear Regression"),
    evaluate(y_test, y_pred_rf, "Random Forest")
]

# ========== Step 5: Display Results ==========
results_df = pd.DataFrame(results)
print("\n🔍 Model Performance on LCF Dataset:\n")
print(results_df)
print('Random Forest is the better model')
