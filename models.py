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

# Check the test set 
print("\nClass distribution in the test set (8 samples):")
print(y_test.value_counts())

# Scaling for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_train_pred_svm = svm.predict(X_train_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_train_pred_rf = rf.predict(X_train)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=25,    
    learning_rate=0.1, 
    max_depth=3,            
    colsample_bytree=0.8,
    subsample=0.8,
    reg_lambda=1,        
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_train_pred_xgb = xgb_model.predict(X_train_scaled)

# Neural Network (MLP)
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8), 
    max_iter=1000, 
    activation='relu', 
    solver='adam', 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
y_train_pred_mlp = mlp.predict(X_train_scaled)

# Print out important metrics
model_data = [
    ("SVM", y_train_pred_svm, y_pred_svm),
    ("Random Forest", y_train_pred_rf, y_pred_rf),
    ("XGBoost", y_train_pred_xgb, y_pred_xgb),
    ("Neural Network", y_train_pred_mlp, y_pred_mlp)
]

print(f"\n{'Model':<15} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 40)

for name, train_p, test_p in model_data:
    tr_acc = accuracy_score(y_train, train_p)
    te_acc = accuracy_score(y_test, test_p)
    print(f"{name:<15} | {tr_acc:<10.2%} | {te_acc:<10.2%}")
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, test_p, target_names=['Not Lonely', 'Lonely']))
    print("-" * 40)

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models_viz = [
    ("SVM", y_pred_svm),
    ("Random Forest", y_pred_rf),
    ("XGBoost", y_pred_xgb),
    ("Neural Network", y_pred_mlp)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, pred) in enumerate(models_viz):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Lonely', 'Lonely'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"{name} Confusion Matrix")

plt.tight_layout()
plt.show()