'''VIF or ElasticNet filtering data'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
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
data = pd.read_csv('data_0802.csv', encoding='utf-8')
data = data.drop(data.columns[0], axis=1)
X_columns = [c for c in data.columns if c != '已填寫問卷數量']

# Y Transformation & X Pre-process
Y_data = data['已填寫問卷數量'].map(lambda x: 0  if int(x) == 1 else 1)
X_data = data[X_columns]
scaler = MinMaxScaler() # StandardScaler()
X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


# Elastic Net model for feature selection
enet = ElasticNet(alpha=0.01, l1_ratio=0.1)
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
              GaussianNB(),
              xgb.XGBClassifier(random_state=42, eval_metric='logloss', learning_rate=0.1, 
                              max_depth=3, n_estimators=100, objective='binary:logistic')]


# Ensemble voting with Soft voting
clf1 = LogisticRegression(solver="liblinear", random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = svm.SVC(probability=True, random_state=42)
eclf_soft = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')
model_list.append(eclf_soft)

# Stacking
stack_cf = StackingClassifier(estimators=[('rf', clf2), ('svm', clf3)], final_estimator=clf1, cv=5)
model_list.append(stack_cf)
