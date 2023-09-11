'''Get model and tune the parameters'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, roc_auc_score, roc_curve, accuracy_score, auc
from sklearn import metrics, model_selection, svm
from sklearn.naive_bayes import GaussianNB
import warnings

# Init_Settings
warnings.filterwarnings("ignore")
plotting = True

# Read data
data = pd.read_csv('../Data/data_0802.csv', encoding='utf-8')
data = data.drop(data.columns[0], axis=1)
X_columns = [c for c in data.columns if c != '已填寫問卷數量']

# Y Transformation & X Pre-process
Y_data = data['已填寫問卷數量'].map(lambda x: 0  if int(x) == 1 else 1)
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
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)


# Elastic Net model for feature selection
enet = ElasticNet(alpha=grid_search.best_params_['alpha'], l1_ratio=grid_search.best_params_['l1_ratio'], random_state=42)
enet.fit(X_data_scaled, Y_data)

# Get selected features based on non-zero coefficients
selected_features = X_data_scaled.columns[enet.coef_ != 0].tolist()
X_enet = X_data_scaled[selected_features]
# print(f"Selected {len(selected_features)} features: {selected_features}")


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_enet, Y_data, test_size=0.2, random_state=42)
y_train, y_test = y_train.tolist(), y_test.tolist()

# Random Forest Classification, LogisticRegression, SVM
model_name_list = ['Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Gaussian NB', 'XGB', 'Soft Voting', 'Stacking']
model_list = [RandomForestClassifier(n_estimators=100, random_state=42),  
              LogisticRegression(solver="liblinear", random_state=42), 
              svm.SVC(probability=True, random_state=42),
              GaussianNB()]

# XGB
model_xgb = xgb.XGBClassifier(random_state=42)
param_grid = {
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss'],
    'learning_rate': [0.5, 0.1, 0.01],
    'n_estimators': [20, 50, 100, 200],
    'max_depth': [3, 5, 6, 7],
}

# Perform grid search with cross-validation
grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search_xgb.best_params_
best_xgb_model = grid_search_xgb.best_estimator_
model_list.append(best_xgb_model)
# print(f"Best Hyperparameters of XGB Model: {best_params}")


# Ensemble voting with Soft voting
clf1 = LogisticRegression(solver="liblinear", random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = svm.SVC(probability=True, random_state=42)
eclf_soft = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')
model_list.append(eclf_soft)

# Stacking
stack_cf = StackingClassifier(estimators=[('rf', clf2), ('svm', clf3)], final_estimator=clf1, cv=5)
model_list.append(stack_cf)
