# Import necessary libraries
import pickle
import pandas as pd
from sklearn.metrics import classification_report

from modules import cm_display

dataset = pd.read_csv("./dataset/cic-ids_test_bruteforce.csv", low_memory=False)
x=dataset.drop(["Label"], axis=1)
y=dataset["Label"]

# Load the model from the file
# with open('randomforest_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# with open('svm_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

with open('model/svm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict on the test set
y_pred = loaded_model.predict(x)

# Generate the classification report
report = classification_report(y, y_pred)
print("Classification Report:\n", report)

cm_display(y, y_pred)