'''
Version 0910 analysis with y: 0/1
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from tqdm import tqdm
from get_model import model_list, model_name_list
import warnings
import random
import csv

# Init_Settings
warnings.filterwarnings("ignore")
plotting = False
RANDOM_EPOCHS = 100

# Read data
data = pd.read_csv('../Data/data_0802.csv', encoding='utf-8')
data = data.drop(data.columns[0], axis=1)
X_columns = [c for c in data.columns if c != '已填寫問卷數量']

# Y Transformation & X Pre-process
Y_data = data['已填寫問卷數量'].map(lambda x: 0  if int(x) == 1 else 1)
X_data = data[X_columns]
scaler = MinMaxScaler() # StandardScaler()
X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


# Grid Search for Elastic Nets (Filtering columns using Lasso Elastic Nets)
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

# Get selected features based on non-zero coefficients --> X_enet, Y_data
selected_features = X_data_scaled.columns[enet.coef_ != 0].tolist()
X_enet = X_data_scaled[selected_features]
print(f"Selected {len(selected_features)} features") #: {selected_features}")


# Calculate the confusion matrix
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return y_scores, acc, cm


# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
    return scores



if pd.read_csv('../Data/res_0910.csv').shape[0] != (RANDOM_EPOCHS * len(model_list)):
    '''Evaluate the performance using criterion ROC_AUC and check whether it's consistent or model-dependent
    Store the performance into a csv file'''
    
    data_list = []
    for epoch in tqdm(range(RANDOM_EPOCHS)):
        rand_sel_feat = random.sample(selected_features, random.randint(X_enet.shape[1] // 4, X_enet.shape[1] // 2))
        X_sel = X_enet[rand_sel_feat]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, Y_data, test_size=0.2, random_state=42)
        y_train, y_test = y_train.tolist(), y_test.tolist()
        
        for mdn, md in zip(model_name_list, model_list):
            y_scores, acc, cm = train_model(md, X_train, X_test, y_train, y_test)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            data_list.append([str(mdn + '_' + epoch) , rand_sel_feat, acc, roc_auc])

    data_pd = pd.DataFrame(data_list, columns=['features', 'accuracy', 'roc_auc'])
    data_pd.to_csv('../Data/res_0910.csv')




# Already finish processing
res = pd.read_csv('../Data/res_0910.csv')
roc = res['roc_auc'].tolist()
roc_sort_id = sorted(range(len(roc)), key=lambda i: roc[i], reverse=True)

# Cross validation
for i in roc_sort_id[:10]:
    sel_feat = res.iloc[i, 1][1:-1]
    sel_columns = [st[1:-1] for st in sel_feat.split(', ')]
    print(sel_columns)
    X_selected = X_enet[sel_columns]
    
    res_list = []
    for md_name, model in zip(model_name_list, model_list):
        sc = evaluate_model(model, X_selected, Y_data)
        res_list.append(sc)
        print(f'{md_name}, {round(np.mean(sc), 4)}, {round(np.std(sc), 4)}')

    # Plot model performance for comparison
    if plotting:
        plt.boxplot(res_list, labels=model_name_list, showmeans=True)
        plt.xticks(rotation=45) 
        plt.show()