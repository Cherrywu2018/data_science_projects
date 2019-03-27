#!/usr/bin/env python
# coding: utf-8

# ## goal: build a model that predicts whether a user has a high probability of using the site to perform some illegal activity or not.

# In[1]:


# ignore warnings
import  warnings
warnings.simplefilter('ignore')

# maintain a list in sorted order without having to sort the list after each insertion
import bisect
# import necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, classification_report
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
import xgboost as xgb


# ## Import data and join tables

# In[2]:


# import data
data = pd.read_csv('/Users/cherry/Desktop/Machine learning/Fraud/Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
data.head()


# In[3]:


# import another dataset: mapping each numeric ip address to its country
ad2c = pd.read_csv('/Users/cherry/Desktop/Machine learning/Fraud/IpAddress_to_Country.csv')
ad2c.head()


# In[5]:


# add country as a feature in fraud data, determine country based on IP address
countries = []
for i in range(len(data)):
    ip_address = data.loc[i, 'ip_address']
    tmp = ad2c[(ad2c['lower_bound_ip_address'] <= ip_address) &
                          (ad2c['upper_bound_ip_address'] >= ip_address)]
    if len(tmp) == 1:
        countries.append(tmp['country'].values[0])
    else:
        countries.append('NA')
        
data['country'] = countries

data.head()


# ## Feature engineering

# In[6]:


# feature engineering - possible frauds
# 1. small time difference between signup time and purchase time
time_diff = data['purchase_time'] - data['signup_time']
time_diff = time_diff.apply(lambda x: x.seconds)
data['time_diff'] = time_diff


# In[7]:


# 2. different user ids for the same device 
device_num = data[['user_id', 'device_id']].groupby('device_id').count().reset_index()
device_num = device_num.rename(columns={'user_id': 'device_num'})
data = data.merge(device_num, how='left', on='device_id')


# In[8]:


# 3. different user ids from the same IP address
ip_num = data[['user_id', 'ip_address']].groupby('ip_address').count().reset_index()
ip_num = ip_num.rename(columns={'user_id': 'ip_num'})
data = data.merge(ip_num, how='left', on='ip_address')


# In[9]:


# 4. signup/ purchase day of a week and week of a year
# Signup day and week
data['signup_day'] = data['signup_time'].apply(lambda x: x.dayofweek)
data['signup_week'] = data['signup_time'].apply(lambda x: x.week)

# Purchase day and week
data['purchase_day'] = data['purchase_time'].apply(lambda x: x.dayofweek)
data['purchase_week'] = data['purchase_time'].apply(lambda x: x.week)


# In[10]:


data.head()


# In[26]:


# 5. transactions happened in countries with less users
country_count = data[['user_id','country']].groupby(['country']).count().reset_index()
data = data.merge(country_count,how = 'left',on = 'country')


# In[27]:


data.head()


# In[29]:


data.rename(columns={'user_id_y': 'country_count'}, inplace=True)


# In[30]:


data.head()


# In[31]:


# delete unnecessary features such as user id and device_id and keep other features

features = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source', 
           'browser', 'sex', 'age', 'time_diff', 'device_num', 'ip_num', 'class','country_count']
data = data[features]
data.head()


# In[33]:


# one-hot code
data['is_male'] = (data.sex == 'M').astype(int)
del data['sex']


# In[38]:


data = pd.get_dummies(data,columns=['source','browser'])

del data['source']
del data['browser']
data.head()


# ## Train the model

# In[44]:


data.head()


# In[46]:


seed = 999
X = data.loc[:,data.columns != 'class']
y = data['class']

# split into training dataset and test dataset
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=seed)
train_matrix = xgb.DMatrix(Xtrain,ytrain)
test_matrix = xgb.DMatrix(Xtest)


# In[47]:


# find best number of trees
params = {}
params['silent'] = 1
params['objective'] = 'binary:logistic'  # output probabilities
params['eval_metric'] = 'auc'
params["num_rounds"] = 300
params["early_stopping_rounds"] = 30
# params['min_child_weight'] = 2
params['max_depth'] = 6
params['eta'] = 0.1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8

cv_results = xgb.cv(params,train_matrix,
                    num_boost_round = params["num_rounds"],
                    nfold = params.get('nfold',5),
                    metrics = params['eval_metric'],
                    early_stopping_rounds = params["early_stopping_rounds"],
                    verbose_eval = True,
                    seed = seed)
cv_results


# In[48]:


n_best_trees = cv_results.shape[0]
n_best_trees


# ## plot ROC and choose threshold

# In[49]:


# Training and test sets are biased, we cannot plot ROC on them
# split the training dataset into training set and validation set
# retrain on training set and plot ROC on validation set and choose a proper cutoff value

def plot_validation_roc():
 
    Xtrain_only,Xvalid,ytrain_only,yvalid = train_test_split(Xtrain,ytrain,test_size=0.3,random_state=seed)
    onlytrain_matrix = xgb.DMatrix(Xtrain_only,ytrain_only)
    valid_matrix = xgb.DMatrix(Xvalid,yvalid)

    temp_gbt = xgb.train(params, onlytrain_matrix, n_best_trees,[(onlytrain_matrix,'train_only'),(valid_matrix,'validate')])
    yvalid_proba_pred = temp_gbt.predict(valid_matrix,ntree_limit=n_best_trees)

    fpr,tpr,thresholds = roc_curve(yvalid,yvalid_proba_pred)
    return pd.DataFrame({'FPR':fpr,'TPR':tpr,'Threshold':thresholds})

roc = plot_validation_roc()


# In[50]:


plt.figure(figsize=(10,5))
plt.plot(roc.FPR,roc.TPR,marker='h')
plt.xlabel("FPR")
plt.ylabel("TPR")


# In[ ]:


# Trade-off
# when FP is more important, we should minimize FPR, increase the threshold. (TPR is also decreased.)
# when FN is more important, we should min FNR, decrease the threshold. (FPR is also increased)
# in this case, even if we find some fraudulent activities that are actually not, there are next steps, so min FN.
# choose a smaller threshold


# In[ ]:


# how to use the model:
# If predicted fraud probability < X, no fraudulent activities. (majority)
# If X <= predicted fraud probability < Z, the user in the risk of frauds;
# solution: use additional verification step, including verifying phone number via a code sent by SMS or email.
# If predicted fraud probability >= Z, there is a high risk of a fraud.
# solution: warn the user; freeze the account; check the activity manually.

