#
# models.py
# Reads through the cleaned CSV and trains multiple ML models on the data.
# Code by Max Cheezic, Nicholas Demetrio, Hayden Ward
#
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load and prepare target
df = pd.read_csv('data_cleaned.csv')
df['Loneliness_Level'] = (df['UCLA Loneliness Total (Label)'] >= 28).astype(int)

# Features vs target
X = df.drop(columns=['Participant', 'UCLA Loneliness Total (Label)', 'Loneliness_Level'])
y = df['Loneliness_Level']

# Impute NaNs and split
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print("Class distribution in the full dataset:")
print(df['Loneliness_Level'].value_counts())

# Check the test set specifically
print("\nClass distribution in the test set (8 samples):")
print(y_test.value_counts())

# Scaling for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Print Comparison
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2%}")
print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=50,       # Very few trees
    learning_rate=0.1, 
    max_depth=1,            # "Decision Stumps" - only one split per tree
    colsample_bytree=0.8,
    subsample=0.8,
    reg_lambda=10,          # Heavy penalty on complex weights
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Comparison
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.2%}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# List of models and their names for easy looping
models = [
    ("SVM", y_pred_svm),
    ("Random Forest", y_pred_rf),
    ("XGBoost", y_pred_xgb),
]

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, pred) in enumerate(models):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Lonely', 'Lonely'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"{name} Confusion Matrix")
    
    # Calculate specific counts
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- {name} Stats ---")
    print(f"True Positives: {tp} | True Negatives: {tn}")
    print(f"False Positives: {fp} | False Negatives: {fn}")

plt.tight_layout()
plt.show()

# Random forest feature importance
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
plt.title("Top 10 Predictors of Loneliness (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

#quit()