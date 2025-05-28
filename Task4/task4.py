# ==========================================================================
# Task 4: Credit Risk Analysis - Default Prediction
# ==========================================================================
# Objective: Train models to identify high-risk customers for banks.
# Models: Random Forest & XGBoost (comparison)
# Dataset: Give Me Some Credit (Kaggle)
# ==========================================================================

# --------------------------------------------------------------------------
# 1. Install dependencies (first time only)
# --------------------------------------------------------------------------
# pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

# --------------------------------------------------------------------------
# 2. Imports
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --------------------------------------------------------------------------
# 3. Load Dataset
# --------------------------------------------------------------------------
df = pd.read_csv("cs-training.csv", index_col=0)  # from Kaggle: Give Me Some Credit
print("[INFO] Dataset loaded:", df.shape)

# --------------------------------------------------------------------------
# 4. Handle Missing & Extreme Values
# --------------------------------------------------------------------------
# Fill missing MonthlyIncome with median
df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)

# Fill missing Dependents with 0
df["NumberOfDependents"].fillna(0, inplace=True)

# Cap outliers in age
df = df[df["age"] > 0]

# --------------------------------------------------------------------------
# 5. Feature & Target split
# --------------------------------------------------------------------------
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# --------------------------------------------------------------------------
# 6. Train-Test Split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=42)

# --------------------------------------------------------------------------
# 7. Handle Class Imbalance using SMOTE
# --------------------------------------------------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("[INFO] After SMOTE:", np.bincount(y_train_res))

# --------------------------------------------------------------------------
# 8. Feature Scaling
# --------------------------------------------------------------------------
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 9. Train Models
# --------------------------------------------------------------------------
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res_scaled, y_train_res)
rf_preds = rf.predict(X_test_scaled)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train_res_scaled, y_train_res)
xgb_preds = xgb.predict(X_test_scaled)

# --------------------------------------------------------------------------
# 10. Evaluation Function
# --------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} Report ===")
    print(classification_report(y_true, y_pred))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate both models
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

# --------------------------------------------------------------------------
# 11. Save Results
# --------------------------------------------------------------------------
results_df = X_test.copy()
results_df["Actual"] = y_test
results_df["RF_Predicted"] = rf_preds
results_df["XGB_Predicted"] = xgb_preds
results_df.to_csv("credit_risk_predictions.csv", index=False)
print("[INFO] Predictions saved to credit_risk_predictions.csv")
