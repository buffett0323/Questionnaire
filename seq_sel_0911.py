'''
Version 0911 analysis with y: 0/1
Approach testing: 
Step 1. Randomly search 3-10 features from the dataset, and get top 5 performance
Step 2. Sequential selection
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
import multiprocessing



# Sequential selection
def seq_sel(sel_feats, X_enet, Y_data):
    
    best_accuracy = 0
    X_train, X_test, y_train, y_test = train_test_split(X_enet, Y_data, test_size=0.2, random_state=42)
    y_train, y_test = y_train.tolist(), y_test.tolist()

    # Perform forward feature selection
    while len(sel_feats) < X_train.shape[1]:
        print(f'Epoch record: {len(sel_feats)}')
        
        best_feature = None
        for feature_idx in tqdm(range(X_train.shape[1])):
            if feature_idx not in sel_feats:
                
                # Try adding the feature to the selected set
                trial_features = sel_feats + [feature_idx]
                X_subset = X_train.iloc[:, trial_features]
                
                # Evaluation
                eval_acc = np.mean([evaluate_model(md, X_subset, y_train) for md in model_list])

                if eval_acc > best_accuracy:
                    best_accuracy = eval_acc
                    best_feature = feature_idx
        
        '''Break if adding any feature can't improve the model performance'''
        if best_feature is not None:
            sel_feats.append(best_feature)
            print(f"Selected feature: {best_feature}, Best accuracy: {best_accuracy:.2f}")
        else: break

    # print("Final selected features:", selected_features)
    return [sel_feats, best_accuracy]


def worker(feature_idx, sel_feats, X_train, y_train):

    # Try adding the feature to the selected set
    trial_features = sel_feats + [feature_idx]
    X_subset = X_train.iloc[:, trial_features]
    
    # Evaluation
    acc = np.mean([evaluate_model(md, X_subset, y_train) for md in model_list]) # evaluation accuracy
    print(f'Worker {feature_idx}')
    
    return [acc, feature_idx] 



# Evaluate a given model using 5-fold cross-validation
def evaluate_model(model, X, y):
    return cross_val_score(model, X, y, scoring='accuracy', cv=5)



if __name__ == "__main__":
        
    # Init_Settings
    warnings.filterwarnings("ignore")
    num_cores = multiprocessing.cpu_count()
    plotting = True
    TOP_PERFORMANCE = 10

    # Get data
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

    # Read data and select top 5 performance
    res = pd.read_csv('data/mp_res_0910.csv', encoding='utf-8')
    perf = res['performance'].tolist()
    perf_sort_id = sorted(range(len(perf)), key=lambda i: perf[i], reverse=True)

    # Top performances
    all_feats, all_best_acc = [], []
    sel_columns_id_list = []
    for order, i in enumerate(perf_sort_id[:TOP_PERFORMANCE]):
        sel_feat = res.iloc[i, 1][1:-1]
        sel_columns = [st[1:-1] for st in sel_feat.split(', ')]
        sel_columns_id = [sel_columns.index(id) for id in sel_columns]
        sel_columns_id_list.append(sel_columns_id)


    
    X_train, X_test, y_train, y_test = train_test_split(X_enet, Y_data, test_size=0.2, random_state=42)
    y_train, y_test = y_train.tolist(), y_test.tolist()

    
    total_acc, total_sel_feats, total = [], [], []
    
    # Start searching
    for sel_feats in sel_columns_id_list:
        best_accuracy = 0
        
        # Perform forward feature selection
        while len(sel_feats) < X_train.shape[1]:
            print(f'Epoch record: {len(sel_feats)}')
            
            best_feature = None
            
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.starmap(worker, [(feature_idx, sel_feats, X_train, y_train) for feature_idx in range(X_train.shape[1]) if feature_idx not in sel_feats])
            
            acc, feat_idx = [r[0] for r in results], [r[1] for r in results]
            
            if max(acc) > best_accuracy:
                best_accuracy = max(acc)
                best_feature = feat_idx[acc.index(best_accuracy)]
        
            
            '''Break if adding any feature can't improve the model performance'''
            if best_feature is not None:
                sel_feats.append(best_feature)
                print(f"Selected feature: {best_feature}, Best accuracy: {best_accuracy:.2f}")
            else: break
            
        # End of while
        
        
        total_acc.append(best_accuracy)
        total_sel_feats.append(sel_feats)
        total.append([sel_feats, best_accuracy])


    print(total_acc)
    print(total_sel_feats)

    print("Start storing files!")
    data_pd = pd.DataFrame(total, columns=['features', 'performance'])
    data_pd.to_csv(f'seq_sel_top_{TOP_PERFORMANCE}.csv')
    print("Successfully stored!")

        
    


