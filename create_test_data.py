import numpy as np
import pandas as pd
import os, re, time, math

# import custom function or lib
from modules import fixDataType, transformTargetLabel, dropColumns, dropInfinateNull

# load dataset
raw_data01 = pd.read_csv("./dataset/ids_intrusion_14022018.csv", low_memory=False)
raw_data02 = pd.read_csv("./dataset/ids_intrusion_15022018.csv", low_memory=False)
raw_data03 = pd.read_csv("./dataset/ids_intrusion_16022018.csv", low_memory=False)
# raw_data04 = pd.read_csv("./dataset/ids_intrusion_21022018.csv", low_memory=False)
# raw_data05 = pd.read_csv("./dataset/ids_intrusion_22022018.csv", low_memory=False)

# Data type fix
raw_data01 = fixDataType(raw_data01)
raw_data02 = fixDataType(raw_data02)
raw_data03 = fixDataType(raw_data03)
raw_data04 = fixDataType(raw_data04)
# raw_data05 = fixDataType(raw_data05)

# Transform Target Label into Binary Format
raw_data01 = transformTargetLabel(raw_data01)
raw_data02 = transformTargetLabel(raw_data02)
raw_data03 = transformTargetLabel(raw_data03)
# raw_data04 = transformTargetLabel(raw_data04)
# raw_data05 = transformTargetLabel(raw_data05)

# data merger
network_data = pd.concat([raw_data01, raw_data02], axis=0)
# network_data = pd.concat([raw_data01, raw_data02, raw_data03, raw_data04, raw_data05], axis=0)
network_data.reset_index(drop=True, inplace=True)

# drop all unneeded columns
# network_data = dropColumns(raw_data01)
network_data = dropColumns(network_data)

# drop all infinate and null data
network_data = dropInfinateNull(network_data)

# save processed dataset
network_data.to_csv('./dataset/cic-ids_test_bruteforce.csv', index=False)



