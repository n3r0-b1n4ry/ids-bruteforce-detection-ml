import numpy as np
import pandas as pd
import os, re, time, math

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# import custom function or lib
from modules import fixDataType, transformTargetLabel, downsampling_benign, randomforest, supportvectorclassifier

# load dataset
raw_data = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)

# Data type fix
raw_data = fixDataType(raw_data)

# data properties
# print(raw_data.info())
print(raw_data['Label'].value_counts())

# replace infinity value as null value
processed_data = raw_data.replace(["Infinity", "infinity"], np.inf)
processed_data = processed_data.replace([np.inf, -np.inf], np.nan)

# drop all null values and timestamp column
processed_data.dropna(inplace=True)
processed_data.drop(columns="Timestamp", inplace=True)

# Transform Target Label into benign or malicious
processed_data = transformTargetLabel(processed_data)

# downsampling
sampling_data = downsampling_benign(processed_data)
print(sampling_data['Label'].value_counts())

# random forest
# rf = randomforest(sampling_data)

# support vector machine
svc = supportvectorclassifier(sampling_data)

