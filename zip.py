'''
Use Elastic Nets to filter some columns and then use Zero-inflated Model for Y 0-100
Ref from https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

from tqdm import tqdm
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Init_Settings
warnings.filterwarnings("ignore")
plotting = True

# Read data
data = pd.read_csv('data/data_0802.csv', encoding='utf-8')
data = data.drop(data.columns[0], axis=1)
X_columns = [c for c in data.columns if c != '已填寫問卷數量']

# Y Transformation & X Pre-process
Y_data = data['已填寫問卷數量'].tolist()#.map(lambda x: 0  if int(x) == 1 else 1)
Y_data = [int(y-1) for y in Y_data]
X_data = data[X_columns]
scaler = MinMaxScaler() # StandardScaler()
X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)

# Elastic Net model for feature selection
enet = ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=42)
enet.fit(X_data_scaled, Y_data)

# Get selected features based on non-zero coefficients --> X_enet, Y_data
selected_features = X_data_scaled.columns[enet.coef_ != 0].tolist()
X_enet = X_data_scaled[selected_features]




# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_enet, Y_data, test_size=0.2, random_state=42)

# Top performance
top_perf = pd.read_csv('data/seq_sel_top_10.csv', encoding='utf-8')
summary_list, feat_sel_list = [], []
for i, j in tqdm(top_perf.iterrows()):
    sel_feat = j[1][1:-1].split(', ')
    sel_feat = [int(i) for i in sel_feat]
    X_subset = X_train.iloc[:, sel_feat]

    # ZIP training, endog and exog with actual data and explanatory variables.
    zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_subset, exog_infl=X_subset, 
                                                    inflation='logit').fit(maxiter=1000) # Try to make it converge
    smy = zip_training_results.summary()
    
    # Evaluate by p-values
    p_value = zip_training_results.pvalues.tolist()
    exog_names = zip_training_results.model.exog_names
    
    feats = [e_name for e_name, pv in zip(exog_names, p_value) if pv < 0.05]
    feat_sel_list.append(feats)

print(feat_sel_list)
    
