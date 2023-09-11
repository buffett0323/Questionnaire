'''
Use Elastic Nets to filter some columns and then use Zero-inflated Model
Ref from https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import warnings

from tqdm import tqdm
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Init_Settings
warnings.filterwarnings("ignore")
plotting = True

# Read data
data = pd.read_csv('../Data/data_0802.csv', encoding='utf-8')
data = data.drop(data.columns[0], axis=1)
X_columns = [c for c in data.columns if c != '已填寫問卷數量']

# Y Transformation & X Pre-process
Y_data = data['已填寫問卷數量'].tolist()#.map(lambda x: 0  if int(x) == 1 else 1)
Y_data = [int(y-1) for y in Y_data]
X_data = data[X_columns]
scaler = MinMaxScaler() # StandardScaler()
X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


# Grid Search for Elastic Nets
param_grid = {
    'alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    'l1_ratio': [0.1, 0.5, 0.8, 0.9, 1]
} 

elastic_net = ElasticNet()
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=5)
grid_search.fit(X_data_scaled, Y_data)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Elastic Net model for feature selection
enet = ElasticNet(alpha=grid_search.best_params_['alpha'], l1_ratio=grid_search.best_params_['l1_ratio'], random_state=42)
enet.fit(X_data_scaled, Y_data)

# Get selected features based on non-zero coefficients
selected_features = X_data_scaled.columns[enet.coef_ != 0].tolist()


# Randomly select some features
sum_list, llf_list = [], []
for _ in tqdm(range(100)):
    l = len(selected_features)
    random_select = random.sample(selected_features, random.randint(l // 4, l // 2))
    X_enet = X_data_scaled[random_select]
    X_train, X_test, y_train, y_test = train_test_split(X_enet, Y_data, test_size=0.2, random_state=42)

    # ZIP training, endog and exog with actual data and explanatory variables.
    zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, exog_infl=X_train, 
                                                  inflation='logit').fit(maxiter=100) # Try to make it converge
    
    
    # print(f"Selected {len(random_select)} out of {len(selected_features)} features: {random_select}")
    sum_list.append(zip_training_results.summary())
    llf_list.append(zip_training_results.llf)


min_id = llf_list.index(max(llf_list))
print(sum_list[min_id])


