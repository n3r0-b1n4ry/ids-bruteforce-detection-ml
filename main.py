from modules import randomforest, supportvectorclassifier
import pandas as pd
import pickle

dataset = pd.read_csv("./dataset/cic-ids.csv", low_memory=False)

# # random forest training model
# rf = randomforest(dataset)

# with open('randomforest_model.pkl', 'wb') as f:
#     pickle.dump(rf, f)

# support vector machine training model
svc = supportvectorclassifier(dataset)

with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svc, f)
