import numpy as np
import pandas as pd
import os, re, time, math

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# import custom function or lib
from modules import fixDataType, transformTargetLabel, balanceData, dropInfinateNull, dropColumns, normalizeData

# load dataset
raw_data01 = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)
# raw_data02 = pd.read_csv("./dataset/ids_intrusion_15022018.csv", low_memory=False)
# raw_data03 = pd.read_csv("./dataset/ids_intrusion_16022018.csv", low_memory=False)
# raw_data04 = pd.read_csv("./dataset/ids_intrusion_21022018.csv", low_memory=False)
# raw_data05 = pd.read_csv("./dataset/ids_intrusion_22022018.csv", low_memory=False)

# Data type fix
raw_data01 = fixDataType(raw_data01)
raw_data02 = fixDataType(raw_data02)
raw_data03 = fixDataType(raw_data03)
raw_data04 = fixDataType(raw_data04)
raw_data05 = fixDataType(raw_data05)

# data properties
# print(raw_data.info())
# print(raw_data['Label'].value_counts())

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



