from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# model classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def fixDataType(dataset):
    
    dataset = dataset[dataset['Dst Port'] != 'Dst Port']
    
    dataset['Dst Port'] = dataset['Dst Port'].astype(int)
    dataset['Protocol'] = dataset['Protocol'].astype(int)
    dataset['Flow Duration'] = dataset['Flow Duration'].astype(int)
    dataset['Tot Fwd Pkts'] = dataset['Tot Fwd Pkts'].astype(int)
    dataset['Tot Bwd Pkts'] = dataset['Tot Bwd Pkts'].astype(int)
    dataset['TotLen Fwd Pkts'] = dataset['TotLen Fwd Pkts'].astype(int)
    dataset['TotLen Bwd Pkts'] = dataset['TotLen Bwd Pkts'].astype(int)
    dataset['Fwd Pkt Len Max'] = dataset['Fwd Pkt Len Max'].astype(int)
    dataset['Fwd Pkt Len Min'] = dataset['Fwd Pkt Len Min'].astype(int)
    dataset['Fwd Pkt Len Mean'] = dataset['Fwd Pkt Len Mean'].astype(float)
    dataset['Fwd Pkt Len Std'] = dataset['Fwd Pkt Len Std'].astype(float)
    dataset['Bwd Pkt Len Max'] = dataset['Bwd Pkt Len Max'].astype(int)
    dataset['Bwd Pkt Len Min'] = dataset['Bwd Pkt Len Min'].astype(int)
    dataset['Bwd Pkt Len Mean'] = dataset['Bwd Pkt Len Mean'].astype(float)
    dataset['Bwd Pkt Len Std'] = dataset['Bwd Pkt Len Std'].astype(float)
    dataset['Flow Byts/s'] = dataset['Flow Byts/s'].astype(float)
    dataset['Flow Pkts/s'] = dataset['Flow Pkts/s'].astype(float)
    dataset['Flow IAT Mean'] = dataset['Flow IAT Mean'].astype(float)
    dataset['Flow IAT Std'] = dataset['Flow IAT Std'].astype(float)
    dataset['Flow IAT Max'] = dataset['Flow IAT Max'].astype(int)
    dataset['Flow IAT Min'] = dataset['Flow IAT Min'].astype(int)
    dataset['Fwd IAT Tot'] = dataset['Fwd IAT Tot'].astype(int)
    dataset['Fwd IAT Mean'] = dataset['Fwd IAT Mean'].astype(float)
    dataset['Fwd IAT Std'] = dataset['Fwd IAT Std'].astype(float)
    dataset['Fwd IAT Max'] = dataset['Fwd IAT Max'].astype(int)
    dataset['Fwd IAT Min'] = dataset['Fwd IAT Min'].astype(int)
    dataset['Bwd IAT Tot'] = dataset['Bwd IAT Tot'].astype(int)
    dataset['Bwd IAT Mean'] = dataset['Bwd IAT Mean'].astype(float)
    dataset['Bwd IAT Std'] = dataset['Bwd IAT Std'].astype(float)
    dataset['Bwd IAT Max'] = dataset['Bwd IAT Max'].astype(int)
    dataset['Bwd IAT Min'] = dataset['Bwd IAT Min'].astype(int)
    dataset['Fwd PSH Flags'] = dataset['Fwd PSH Flags'].astype(int)
    dataset['Bwd PSH Flags'] = dataset['Bwd PSH Flags'].astype(int)
    dataset['Fwd URG Flags'] = dataset['Fwd URG Flags'].astype(int)
    dataset['Bwd URG Flags'] = dataset['Bwd URG Flags'].astype(int)
    dataset['Fwd Header Len'] = dataset['Fwd Header Len'].astype(int)
    dataset['Bwd Header Len'] = dataset['Bwd Header Len'].astype(int)
    dataset['Fwd Pkts/s'] = dataset['Fwd Pkts/s'].astype(float)
    dataset['Bwd Pkts/s'] = dataset['Bwd Pkts/s'].astype(float)
    dataset['Pkt Len Min'] = dataset['Pkt Len Min'].astype(int)
    dataset['Pkt Len Max'] = dataset['Pkt Len Max'].astype(int)
    dataset['Pkt Len Mean'] = dataset['Pkt Len Mean'].astype(float)
    dataset['Pkt Len Std'] = dataset['Pkt Len Std'].astype(float)
    dataset['Pkt Len Var'] = dataset['Pkt Len Var'].astype(float)
    dataset['FIN Flag Cnt'] = dataset['FIN Flag Cnt'].astype(int)
    dataset['SYN Flag Cnt'] = dataset['SYN Flag Cnt'].astype(int)
    dataset['RST Flag Cnt'] = dataset['RST Flag Cnt'].astype(int)
    dataset['PSH Flag Cnt'] = dataset['PSH Flag Cnt'].astype(int)
    dataset['ACK Flag Cnt'] = dataset['ACK Flag Cnt'].astype(int)
    dataset['URG Flag Cnt'] = dataset['URG Flag Cnt'].astype(int)
    dataset['CWE Flag Count'] = dataset['CWE Flag Count'].astype(int)
    dataset['ECE Flag Cnt'] = dataset['ECE Flag Cnt'].astype(int)
    dataset['Down/Up Ratio'] = dataset['Down/Up Ratio'].astype(int)
    dataset['Pkt Size Avg'] = dataset['Pkt Size Avg'].astype(float)
    dataset['Fwd Seg Size Avg'] = dataset['Fwd Seg Size Avg'].astype(float)
    dataset['Bwd Seg Size Avg'] = dataset['Bwd Seg Size Avg'].astype(float)
    dataset['Fwd Byts/b Avg'] = dataset['Fwd Byts/b Avg'].astype(int)
    dataset['Fwd Pkts/b Avg'] = dataset['Fwd Pkts/b Avg'].astype(int)
    dataset['Fwd Blk Rate Avg'] = dataset['Fwd Blk Rate Avg'].astype(int)
    dataset['Bwd Byts/b Avg'] = dataset['Bwd Byts/b Avg'].astype(int)
    dataset['Bwd Pkts/b Avg'] = dataset['Bwd Pkts/b Avg'].astype(int)
    dataset['Bwd Blk Rate Avg'] = dataset['Bwd Blk Rate Avg'].astype(int)
    dataset['Subflow Fwd Pkts'] = dataset['Subflow Fwd Pkts'].astype(int)
    dataset['Subflow Fwd Byts'] = dataset['Subflow Fwd Byts'].astype(int)
    dataset['Subflow Bwd Pkts'] = dataset['Subflow Bwd Pkts'].astype(int)
    dataset['Subflow Bwd Byts'] = dataset['Subflow Bwd Byts'].astype(int)
    dataset['Init Fwd Win Byts'] = dataset['Init Fwd Win Byts'].astype(int)
    dataset['Init Bwd Win Byts'] = dataset['Init Bwd Win Byts'].astype(int)
    dataset['Fwd Act Data Pkts'] = dataset['Fwd Act Data Pkts'].astype(int)
    dataset['Fwd Seg Size Min'] = dataset['Fwd Seg Size Min'].astype(int)
    dataset['Active Mean'] = dataset['Active Mean'].astype(float)
    dataset['Active Std'] = dataset['Active Std'].astype(float)
    dataset['Active Max'] = dataset['Active Max'].astype(int)
    dataset['Active Min'] = dataset['Active Min'].astype(int)
    dataset['Idle Mean'] = dataset['Idle Mean'].astype(float)
    dataset['Idle Std'] = dataset['Idle Std'].astype(float)
    dataset['Idle Max'] = dataset['Idle Max'].astype(int)
    dataset['Idle Min'] = dataset['Idle Min'].astype(int)
    
    return dataset

def transformTargetLabel(dataset):
    dataset['Label'] = dataset['Label'].apply(lambda x: "Benign" if x == 'Benign' else "Malicious")
    return dataset

def downsampling_benign(dataset):
    # split data into features and target
    x=dataset.drop(["Label"], axis=1)
    y=dataset["Label"]

    # applying downsampling
    rus = RandomUnderSampler()
    x_balanced, y_balanced = rus.fit_resample(x, y)

    dataset = pd.concat([x_balanced, y_balanced], axis=1)

    return dataset

def randomforest(dataset):
    # split data
    x=dataset.drop(["Label"], axis=1)
    y=dataset["Label"]
    # K-fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # create a Random Forest model
    rf = RandomForestClassifier()

    # predict
    y_pred = cross_val_predict(rf, x, y, cv=kf)

    # # generate report
    # cm=confusion_matrix(y, y_pred)
    # cr=classification_report(y, y_pred)

    # print("Confusion Matrix:")
    # print(cm)
    # print("Performance Matrix:")
    # print(cr)

    return rf

def supportvectorclassifier(dataset):
    # split data
    x=dataset.drop(["Label"], axis=1)
    y=dataset["Label"]
    # K-fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # create a SVC model
    svc = SVC()

    # predict
    y_pred = cross_val_predict(svc, x, y, cv=kf)

    print ("Support Vector Classifier")
    # generate report
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("Performance Matrix:")
    print(classification_report(y, y_pred))

    return svc