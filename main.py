import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle

# Import custom functions
from modules import fixDataType, transformTargetLabel, balanceData, dropInfinateNull, dropColumns, normalizeData
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
raw_data = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)

# Data type fix
raw_data = fixDataType(raw_data)

# Display class distribution
print("Class distribution before processing:")
print(raw_data['Label'].value_counts())

# Replace infinity values and drop nulls
processed_data = raw_data.replace(["Infinity", "infinity"], np.inf)
processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
processed_data.dropna(inplace=True)

# Drop unnecessary columns
if "Timestamp" in processed_data.columns:
    processed_data.drop(columns="Timestamp", inplace=True)

# Transform target label
processed_data = transformTargetLabel(processed_data)

# Display class distribution after transformation
print("Class distribution after label transformation:")
print(processed_data['Label'].value_counts())

# Feature-target separation
X = processed_data.drop(columns=["Label"])
y = processed_data["Label"]

# Handle class imbalance using SMOTE
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Display new class distribution
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Grid Search with Cross-Validation for Random Forest
print("Optimizing Random Forest using Grid Search...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv, scoring='f1', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
print("Best Random Forest Parameters:", rf_grid_search.best_params_)

# Save the Random Forest model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_best_model, file)
print("Random Forest model saved as 'random_forest_model.pkl'.")

# Evaluate Random Forest
rf_predictions = rf_best_model.predict(X_val)
print("Random Forest Classification Report:")
print(classification_report(y_val, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_val, rf_predictions))

# Plot ROC Curve for Random Forest
rf_probs = rf_best_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, rf_probs, pos_label="malicious")
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("random_forest_roc_curve.png")
print("Random Forest ROC Curve saved as 'random_forest_roc_curve.png'.")

# Support Vector Machine
# Data processing for SVM
# load dataset
raw_data01 = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)
raw_data02 = pd.read_csv("./dataset/ids_intrusion_15022018.csv", low_memory=False)
raw_data03 = pd.read_csv("./dataset/ids_intrusion_16022018.csv", low_memory=False)
raw_data04 = pd.read_csv("./dataset/ids_intrusion_21022018.csv", low_memory=False)
raw_data05 = pd.read_csv("./dataset/ids_intrusion_22022018.csv", low_memory=False)

# Data type fix
raw_data01 = fixDataType(raw_data01)
raw_data02 = fixDataType(raw_data02)
raw_data03 = fixDataType(raw_data03)
raw_data04 = fixDataType(raw_data04)
raw_data05 = fixDataType(raw_data05)

# Transform Target Label into Binary Format
raw_data01 = transformTargetLabel(raw_data01)
raw_data02 = transformTargetLabel(raw_data02)
raw_data03 = transformTargetLabel(raw_data03)
raw_data04 = transformTargetLabel(raw_data04)
raw_data05 = transformTargetLabel(raw_data05)

# data merger
network_data = pd.concat([raw_data01, raw_data02, raw_data03, raw_data04, raw_data05], axis=0)
network_data.reset_index(drop=True, inplace=True)

# drop all unneeded columns
network_data = dropColumns(network_data)
# network_data = dropColumns(raw_data01)

# drop all infinate and null data
network_data = dropInfinateNull(network_data)

# drop duplicates
network_data = network_data.drop_duplicates()

# Balance label data
network_data = balanceData(network_data)

# normalize dataset
# network_data = normalizeData(network_data)

print(network_data['Label'].value_counts())

# save processed dataset
network_data.to_csv('./dataset/cic-ids.csv', index=False)

# Training model with Support Vector Machine
dataset = pd.read_csv("./dataset/cic-ids.csv", low_memory=False)
# support vector machine training model
svc = supportvectorclassifier(dataset)
# Save the SVM model
with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svc, f)

# Create test dataset for SVM prediction
# load dataset
raw_data01 = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)
raw_data02 = pd.read_csv("./dataset/ids_intrusion_15022018.csv", low_memory=False)
raw_data03 = pd.read_csv("./dataset/ids_intrusion_16022018.csv", low_memory=False)
raw_data04 = pd.read_csv("./dataset/ids_intrusion_21022018.csv", low_memory=False)
raw_data05 = pd.read_csv("./dataset/ids_intrusion_22022018.csv", low_memory=False)

# Data type fix
raw_data01 = fixDataType(raw_data01)
raw_data02 = fixDataType(raw_data02)
raw_data03 = fixDataType(raw_data03)
raw_data04 = fixDataType(raw_data04)
raw_data05 = fixDataType(raw_data05)

# Transform Target Label into Binary Format
raw_data01 = transformTargetLabel(raw_data01)
raw_data02 = transformTargetLabel(raw_data02)
raw_data03 = transformTargetLabel(raw_data03)
raw_data04 = transformTargetLabel(raw_data04)
raw_data05 = transformTargetLabel(raw_data05)

# data merger
network_data = pd.concat([raw_data01, raw_data02, raw_data03, raw_data04, raw_data05], axis=0)
network_data.reset_index(drop=True, inplace=True)

# drop all unneeded columns
network_data = dropColumns(network_data)

# drop all infinate and null data
network_data = dropInfinateNull(network_data)

# save test dataset
network_data.to_csv('./dataset/cic-ids_test_bruteforce.csv', index=False)

# Evaluate the SVM
# load dataset
dataset = pd.read_csv("./dataset/cic-ids_test_bruteforce.csv", low_memory=False)
x=dataset.drop(["Label"], axis=1)
y=dataset["Label"]
# load model
with open('model/svm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
# predict
y_pred = loaded_model.predict(x)
# Generate the classification report and confusion matrix
report = classification_report(y, y_pred)
print("Classification Report:\n", report)
cm_display(y, y_pred)