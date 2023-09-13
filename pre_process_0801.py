'''
To help answer our objective question: What kind of voters have the higher tendency to fill out the questionnaire
Methods tried: Random Forest, Lasso, Pearson, Clustering
Take raw data (merge data result) to do pre-process
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr, nbinom
warnings.filterwarnings("ignore")


# Pre-process
merge_data = pd.read_csv('../Data/merge_result.csv', encoding='utf-8')

# Select useful columns and Drop na values
x_column = ['年齡區間.x', '城市', '教育階段.x', '性別.x', '婚姻狀態', '政治傾向', '籍貫', '已填寫問卷數量', '職業.x', 'X28.1', 'X30', 'X32', 
            'X35', 'X36', 'X48', 'X53', 'X54', 'X4', 'X5', 'X9', 'X12', 'X14', 'X56', 'X58', 'X64', 'X65', 'X66', 'X70', '主要使用聯絡方式'] 
ohe_column = ['城市', '政治傾向', '籍貫', '職業.x', 'X28.1', 'X30', 'X12', 'X64', 'X65', 'X66', '主要使用聯絡方式'] 
orig_col = [x for x in x_column if x not in ohe_column]


sel_data = merge_data[x_column].dropna()
sel_data = sel_data.reset_index(drop=True)
print(f'Original shape after dropping na value bfft: {sel_data.shape}')





''' Other fine-tuning '''
# Sex
sel_data['性別.x'] = [int(x) - 1 for x in sel_data['性別.x']]

# Merry
merry = []
for x in sel_data['婚姻狀態']:
    if int(x) >= 2 and int(x) <= 6:
        merry.append(0)
    else:
        merry.append(int(x))

sel_data['婚姻狀態'] = merry
sel_data = sel_data[sel_data['婚姻狀態'] != 7]

# Native place
for index in [8, 99]: # [1, 6, 99]:
    sel_data = sel_data[sel_data['籍貫'] != index]

# Job
sel_data = sel_data[sel_data['職業.x'] != 99]

# Other Xs
x4, x5 = [], []
for x in sel_data['X4']:
    if int(x) >= 4:
        x4.append(4)
    else:
        x4.append(int(x))

for x in sel_data['X5']:
    if int(x) >= 4:
        x5.append(4)
    else:
        x5.append(int(x))
        
        
sel_data['X4'] = x4
sel_data['X5'] = x5
sel_data = sel_data[sel_data['X9'] != 3]
sel_data['X9'] = [int(x) - 1 for x in sel_data['X9']]
sel_data = sel_data[sel_data['X12'] != 6]
sel_data['X56'] = [int(x) - 1 for x in sel_data['X56']]
sel_data['X58'] = [int(x) - 1 for x in sel_data['X58']]


for index in [1, 6, 99]:
    sel_data = sel_data[sel_data['X64'] != index]
    sel_data = sel_data[sel_data['X65'] != index]
    sel_data = sel_data[sel_data['X66'] != index]

'''sel_data = sel_data[sel_data['主要使用聯絡方式'] != 1]'''

# Reset the index
sel_data = sel_data.reset_index(drop=True)
print(f'Original shape after dropping na value and pre-process: {sel_data.shape}')


# Data without the need to pre-process
orig_pd = sel_data[orig_col].reset_index(drop=True)

# One Hot Encoding
ohe = OneHotEncoder()
ohe_data = ohe.fit_transform(sel_data[ohe_column])
ohe_data = ohe_data.toarray()

# Ohe column names
ohe_cat = [] 
for id, cat in enumerate(ohe.categories_):
    for num in cat:
        tmp_name = ohe_column[id] + '_' + str(num)
        ohe_cat.append(tmp_name)
        
ohe_pd = pd.DataFrame(ohe_data, columns=ohe_cat)




# Mix the data
mix_pd = pd.concat([orig_pd, ohe_pd], axis=1).reset_index(drop=True)
mix_pd.to_csv('data_0802.csv')

