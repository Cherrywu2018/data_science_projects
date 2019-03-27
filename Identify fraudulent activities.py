#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# goal: build a model that predicts whether a user has a high probability of using the site to 
# perform some illegal activity or not. 


# In[1]:


# ignore warnings
import  warnings
warnings.simplefilter('ignore')

# import necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report

# use h2o data frame: for large dataset, H2O is really fast. 
import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[3]:


# import data
data = pd.read_csv('/Users/cherry/Desktop/Machine learning/Fraud/Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
data.head()


# In[5]:


# import another dataset: mapping each numeric ip address to its country
ad2c = pd.read_csv('/Users/cherry/Desktop/Machine learning/Fraud/IpAddress_to_Country.csv')
ad2c.head()


# In[6]:


# add country as a feature in fraud data
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


# In[8]:


# feature engineering - possible frauds
# 1. small time difference between signup time and purchase time
time_diff = data['purchase_time'] - data['signup_time']
time_diff = time_diff.apply(lambda x: x.seconds)
data['time_diff'] = time_diff


# In[9]:


# 2. different user ids for the same device 
device_num = data[['user_id', 'device_id']].groupby('device_id').count().reset_index()
device_num = device_num.rename(columns={'user_id': 'device_num'})
data = data.merge(device_num, how='left', on='device_id')


# In[10]:


# 3. different user ids from the same IP address
ip_num = data[['user_id', 'ip_address']].groupby('ip_address').count().reset_index()
ip_num = ip_num.rename(columns={'user_id': 'ip_num'})
data = data.merge(ip_num, how='left', on='ip_address')


# In[11]:


# 4. signup/ purchase day of a week and week of a year
# Signup day and week
data['signup_day'] = data['signup_time'].apply(lambda x: x.dayofweek)
data['signup_week'] = data['signup_time'].apply(lambda x: x.week)

# Purchase day and week
data['purchase_day'] = data['purchase_time'].apply(lambda x: x.dayofweek)
data['purchase_week'] = data['purchase_time'].apply(lambda x: x.week)


# In[12]:


# have a look at our new data after features engineering
data.head()


# In[13]:


# delete unnecessary features such as user id and device_id and keep other features
features = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source', 
           'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num', 'class']
data = data[features]
data.head()


# In[16]:


# build the model
# Initialize H2O cluster
h2o.init()
h2o.remove_all()


# In[17]:


# convert to h2o frame
h2o_df = H2OFrame(data)

# convert features to categories
for f in ['signup_day', 'purchase_day', 'source', 'browser', 'sex', 'country', 'class']:
    h2o_df[f] = h2o_df[f].asfactor()

h2o_df.summary()


# In[18]:


# Split training and test sets (70/30)
# binary feature - class: use stratified split method to aviod imbalance
data_split = h2o_df['class'].stratified_split(test_frac=0.3, seed=42)

train = h2o_df[data_split == 'train']
test = h2o_df[data_split == 'test']

# Define features and target
feature = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 
           'source', 'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num']
target = 'class'

# Build random forest model 
model = H2ORandomForestEstimator(balance_classes=True, ntrees=100, mtries=-1, stopping_rounds=5, 
                                 stopping_metric='auc', score_each_iteration=True, seed=42)
model.train(x=feature, y=target, training_frame=train, validation_frame=test)


# In[19]:


model.score_history()


# In[22]:


# Plot feature importance
# varimp: Calculation Of Variable Importance For Regression And Classification Models, for objects produced by train
importance = model.varimp(use_pandas=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='scaled_importance', y='variable', data = importance)
plt.show()


# In[32]:


# Make predictions 
train_data = train.as_data_frame()['class'].values
test_data = test.as_data_frame()['class'].values
train_pred = model.predict(train).as_data_frame()['p1'].values
test_pred = model.predict(test).as_data_frame()['p1'].values

print(test_data)
print(test_pred > 0.5)


# In[35]:


# get model scores
train_fpr, train_tpr, _ = roc_curve(train_data, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_data, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)

print(test_fpr)
print(test_tpr)

# Classification report
print(classification_report(y_true=test_data, y_pred=(test_pred > 0.5).astype(int)))
# test_pred > 0.5: threshold is 0.5


# In[36]:


# add (0,0) to plot AUC
train_fpr = np.insert(train_fpr, 0, 0)
train_tpr = np.insert(train_tpr, 0, 0)
test_fpr = np.insert(test_fpr, 0, 0)
test_tpr = np.insert(test_tpr, 0, 0)

print(test_fpr)
print(test_tpr)


# In[38]:


# plot AUC
# when there are imbalanced classes, it's more useful to report AUC for a precision-recall curve.
# AUC meatures accuracy. 1 represents a perfect test; 0.5 represents a worthless test (change curve)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc))
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc))
ax.plot(train_tpr, train_tpr, 'k--', label='Chance Curve')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.grid(True)
ax.legend(fontsize=12)
plt.show()


# In[ ]:


# how different assumptions about the cost of false positives vs true positives would impact the model.
# 1. if we want to minimize FPR: TPR = ~0.5 and FPR = zero 
# 2. if we want to maximize TPR, we will have to decrease the cut-off. 
# lower the threshold, classify more events as “1”: true positive goes up, false positive will also go up.


# In[39]:


# Shutdown h2o instance
h2o.cluster().shutdown()


# In[ ]:


# explanation of the model:
# different users in the same device are more likely to be classified as at risk; 
# small time difference between signup time and purshase time
# puchase weeks aer in the first five weeks of a year
# users in certain counties.


# In[ ]:


# how to use the moel:
# If predicted fraud probability < X, no fraudulent activities. (majority)
# If X <= predicted fraud probability < Z, the user in the risk of frauds;
# solution: use additional verification step, including verifing phone number via a code sent by SMS or email.
# If predicted fraud probability >= Z, there is a high risk of a fraud.
# solution: warn the user; freeze the account; check the activity manually.


# In[ ]:




