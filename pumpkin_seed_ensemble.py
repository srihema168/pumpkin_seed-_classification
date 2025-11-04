"""
An Optimized Ensemble Machine Learning Framework for Pumpkin Seed Classification
-------------------------------------------------------------------------------
Author: Hema Sri
Description:
This project proposes a hybrid ensemble approach integrating Bagging, Boosting,
RUSBoosting, and Optimized Ensemble techniques for binary classification of
pumpkin seed types using MRMR feature selection.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
import pymrmr
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Load Dataset
# ===============================
# Replace with your dataset file name
data = pd.read_csv("pumpkin_seed_data.csv")

# Assume last column is the target class
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset Shape:", data.shape)
print("Class Distribution:\n", y.value_counts())

# ===============================
# 3. MRMR Feature Selection
# ===============================
# Combine target + features (as required by pymrmr)
df = pd.concat([y, X], axis=1)

# Select top 10 features using MRMR
selected_features = pymrmr.mRMR(df, 'MIQ', 10)
print("\nSelected Features using MRMR:\n", selected_features)

# Keep only the selected features
X_selected = X[selected_features]

# ===============================
# 4. Data Splitting & Scaling
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Define Ensemble Models
# ===============================

# Base learner
base_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)

# 1️⃣ Bagging Classifier
bag_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

# 2️⃣ AdaBoost Classifier
boost_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=0.8, random_state=42)

# 3️⃣ RUSBoost Classifier (for imbalanced datasets)
rusboost_model = RUSBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

# 4️⃣ Optimized Ensemble (Voting of all)
optimized_ensemble = VotingClassifier(
    estimators=[
        ('Bagging', bag_model),
        ('Boosting', boost_model),
        ('RUSBoost', rusboost_model),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'
)

# ===============================
# 6. Train and Evaluate Models
# ===============================
models = {
    "Bagging": bag_model,
    "Boosting": boost_model,
    "RUSBoost": rusboost_model,
    "Optimized Ensemble": optimized_ensemble
}

results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name} Model...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    results[name] = acc

# ===============================
# 7. Display Overall Results
# ===============================
print("\n📊 Final Accuracy Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Performing Model: {best_model} with Accuracy = {results[best_model]:.4f}")

# ===============================
# 8. Save Selected Features & Model Summary
# ===============================
pd.DataFrame({
    'Selected_Features': selected_features
}).to_csv('selected_mrmr_features.csv', index=False)

print("\n✅ Feature selection results saved to 'selected_mrmr_features.csv'")
