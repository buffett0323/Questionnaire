'''
To help answer our objective question: What kind of voters have the higher tendency to fill out the questionnaire
Methods tried: Random Forest, Lasso, Pearson, Clustering
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr, nbinom
warnings.filterwarnings("ignore")

# Multi one-hot-encoding
def mlb_process(data):
    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(data)
    return pd.DataFrame(encoded_data, columns=mlb.classes_)



# Method of Analysis
METHOD = 'nbinom'

# Pre-process
merge_data = pd.read_csv('../Data/merge_result.csv', encoding='utf-8')

# Select useful columns and Drop na values
x_column = ['生日', '年齡區間.x', '城市', '教育階段.x', '性別.x', '婚姻狀態', '政治傾向', '籍貫', '職業.x', 'class', 'X45', 'X72']
select_column = x_column + ['已填寫問卷數量']
sel_data = merge_data[select_column].dropna()

sel_data['生日'] = [2023 - int(y[:4]) for y in sel_data['生日']] # int(y.split('/')[0]) 
sel_data = sel_data[sel_data['生日'] <= 100] # Remove Outliers

sel_data['X45'] = [x.split('、') for x in sel_data['X45']]
sel_data['X72'] = [x.split('、') for x in sel_data['X72']]
party1 = mlb_process(sel_data['X45'])
party2 = mlb_process(sel_data['X72'])
new_pd = pd.concat([party1, party2], axis=1)

sel_data = sel_data.drop(columns=['X45', 'X72']).reset_index(drop=True)
data_pd = pd.concat([sel_data, new_pd], axis=1)

X = data_pd.drop('已填寫問卷數量', axis=1)
y = data_pd['已填寫問卷數量']




# Main
if METHOD == 'RF':
    # Number of top features to select
    K = 'all'  
    selector = SelectKBest(chi2, k=K)
    X_selected = selector.fit_transform(X, y)

    selected_feature_names = X.columns[selector.get_support()] # print(selected_feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Random Forest Regression
    model = RandomForestRegressor() # RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=selected_feature_names)

    # Plot results
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.xticks(rotation=45) 
    plt.show()



elif METHOD == 'Lasso':
    lasso_model = Lasso(alpha=0.1) 
    lasso_model.fit(X, y)
    lasso_coefficients = lasso_model.coef_
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.plot(lasso_coefficients)
    plt.title("Lasso Coefficient")
    plt.xticks(range(len(lasso_coefficients)), X.columns, rotation=45)
    for i in range(len(lasso_coefficients)):
        plt.text(i, lasso_coefficients[i], f'({round(lasso_coefficients[i], 3)})', ha='center', va='bottom')
    plt.show()
 
 

elif METHOD == 'Pearson':
    Pearson_pd = pd.DataFrame()
    for i in range(X.shape[1]):
        corr_coefficient, p_value = pearsonr(X.iloc[:,i], y)
        Pearson_pd[X.columns[i]] = (corr_coefficient, p_value)

    new_row_names = {0: 'Pearson_Coef', 1: 'P_Value'}
    Pearson_pd.rename(index=new_row_names, inplace=True)
    Pearson_pd = Pearson_pd.T
    Pearson_pd.to_csv('../Data/Pearson.csv')
    
    

# Not useful
elif METHOD == 'Cluster':
    kmeans = KMeans(n_clusters=3)
    data = pd.concat([X, y], axis=1)
    kmeans.fit(data)
    print(kmeans.labels_)
    
