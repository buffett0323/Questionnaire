'''
Using multiprocessing
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, roc_auc_score, roc_curve, accuracy_score, auc
from sklearn import metrics, model_selection, svm
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from get_model import model_list, model_name_list
import multiprocessing
import warnings
import random
import csv


def worker(rank, selected_features, X_enet, Y_data):
    
    # Do some work with the item (each iteration of the loop)
    rand_sel_feat = random.sample(selected_features, random.randint(2, 10))
    X_sel = X_enet[rand_sel_feat]
    
    '''
    eval_acc = np.mean([evaluate_model(md, X_sel, Y_data) for md in model_list])
    return rand_sel_feat, eval_acc

    '''
    X_train, X_test, y_train, y_test = train_test_split(X_sel, Y_data, test_size=0.2, random_state=42)
    y_train, y_test = y_train.tolist(), y_test.tolist()
            
    acc_list = [] 
    
    for md in model_list:
        acc = train_model(md, X_train, X_test, y_train, y_test)
        # fpr, tpr, _ = roc_curve(y_test, y_scores)
        # roc_auc = auc(fpr, tpr)
        acc_list.append(acc)
        # ra_list.append(roc_auc)
        
    print(f'Finish epoch: {rank}')
    return [rand_sel_feat, np.mean(acc_list)]
    # return [rand_sel_feat, sum(acc_list)/len(acc_list), sum(ra_list)/len(ra_list)]
    


# Calculate the confusion matrix
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # y_scores = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # return y_scores, acc, cm
    return acc
    

# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Perform 5-fold cross-validation
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores




if __name__ == "__main__":
    
    # Init_Settings
    warnings.filterwarnings("ignore")
    num_cores = multiprocessing.cpu_count()
    plotting = True
    RANDOM_EPOCHS = 1000

    # Read data
    data = pd.read_csv('data/data_0802.csv', encoding='utf-8')
    data = data.drop(data.columns[0], axis=1)
    X_columns = [c for c in data.columns if c != '已填寫問卷數量']

    # Y Transformation & X Pre-process
    Y_data = data['已填寫問卷數量'].map(lambda x: 0  if int(x) == 1 else 1)
    X_data = data[X_columns]
    scaler = MinMaxScaler() # StandardScaler()
    X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


    # Elastic Net model for feature selection
    enet = ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=42)
    enet.fit(X_data_scaled, Y_data)

    # Get selected features based on non-zero coefficients --> X_enet, Y_data
    selected_features = X_data_scaled.columns[enet.coef_ != 0].tolist()
    X_enet = X_data_scaled[selected_features]
    print(f"Selected {len(selected_features)} features") #: {selected_features}")



    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use the map function to apply the perform_calculation function to each iteration in parallel
        results = pool.starmap(worker, [(i, selected_features, X_enet, Y_data) for i in range(RANDOM_EPOCHS)])
 
    
    print("Start storing files!")
    data_pd = pd.DataFrame(results, columns=['features', 'performance'])
    data_pd.to_csv('data/mp_res_0910.csv')
    print("Successfully stored!")

        
    



