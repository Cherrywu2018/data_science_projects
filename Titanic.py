#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('/users/cherry/Desktop/Titanic/train.csv')
test_data = pd.read_csv('/users/cherry/Desktop/Titanic/test.csv')


print(train_data.shape)
train_data.info()


# In[2]:


# explore data - sex
train_data.groupby(['Sex','Survived'])['Survived'].count()

import matplotlib.pyplot as plt
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# In[3]:


# explore data - pclass 
train_data.groupby(['Pclass','Survived'])['Survived'].count()

import matplotlib.pyplot as plt
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()


# In[4]:


# explore data - cabin 
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'na'
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()

# create feature for the alphabetical part of t he cabin number
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x:x[0])

# convert the distinct cabin letters with incremental integer values
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()


# In[5]:


# explore data - age
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(train_data['Age'].min(), train_data['Age'].max()))
facet.add_legend()


# In[6]:


# explore data - name
train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

# name length
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)


# In[7]:


# explore data - SibSp
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(132)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(133)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')

plt.show()


# In[8]:


# explore data - parch
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')

plt.show()


# In[9]:


fig,ax=plt.subplots(1,3,figsize=(15,5))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar(ax=ax[2])
ax[2].set_title('Family and Survived')


# In[10]:


# explore data - fare
fare_notsurvived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_notsurvived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_notsurvived.std(), fare_survived.std()])
average_fare.plot.bar()


# In[11]:


# explore data - embarked
sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
plt.title('Embarked and Survived rate')
plt.show()


# In[12]:


# feature engineering
# combine data
test_data['Survived'] = 0
train_test = train_data.append(test_data)

# embarked
train_test['Embarked'].fillna(train_test['Embarked'].mode().iloc[0], inplace=True)
emb_dummies_df = pd.get_dummies(train_test['Embarked'], prefix=train_test[['Embarked']].columns[0])
train_test = pd.concat([train_test, emb_dummies_df], axis=1)


train_test.head()


# In[13]:


# sex
train_test = pd.get_dummies(train_test,columns=["Sex"])

# title
import re
train_test['Title'] = train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
train_test = pd.get_dummies(train_test,columns=['Title'])

# name length
train_test['Name_length'] = train_test['Name'].apply(len)


train_test.head()


# In[15]:


# fare
train_test['Fare'] = train_test[['Fare']].fillna(train_test.groupby('Pclass').transform(np.mean))

train_test.info()


# In[16]:


# pclass
train_test = pd.get_dummies(train_test,columns=['Pclass'])

train_test.head()


# In[18]:


# ticket
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]

train_test.info()


# In[20]:


train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)


# In[21]:


train_test = pd.get_dummies(train_test,columns=['Ticket_Letter'],drop_first=True)
train_test.info()


# In[24]:


train_test['Cabin'] = train_test['Cabin'].apply(lambda x:1 if pd.notnull(x) else 0)

train_test.head()


# In[23]:


train_test.info()


# In[25]:


train_test.drop("Embarked",inplace = True,axis=1)
train_test.drop("CabinLetter",inplace = True,axis=1)
train_test.drop("Has_Cabin",inplace = True,axis=1)
train_test.drop("Name",inplace = True,axis=1)


# In[26]:


train_test.info()


# In[27]:


train_test['Family_Size'] = train_test['Parch'] + train_test['SibSp'] + 1
train_test.info()


# In[34]:


train_test.info()


# In[35]:


train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['age_nan'])

missing_age = train_test.drop(['Survived'],axis=1)
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)


# In[36]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(missing_age_X_train,missing_age_X_test)
missing_age_X_train = ss.transform(missing_age_X_train)
missing_age_X_test = ss.transform(missing_age_X_test)


# In[37]:


from sklearn import linear_model
lin = linear_model.BayesianRidge()
lin.fit(missing_age_X_train,missing_age_Y_train)
train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)


# In[38]:


train_test.info()


# In[40]:


train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)


# In[41]:


from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
ss2.fit(train_data_X)
train_data_X_sd = ss2.transform(train_data_X)
test_data_X_sd = ss2.transform(test_data_X)


# In[46]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=2,max_depth=6,oob_score=True)

rf.fit(train_data_X,train_data_Y)

print(rf.predict(test_data_X))


test_data["Survived"] = rf.predict(test_data_X)
RF = test_data[['PassengerId','Survived']].set_index('PassengerId')
RF.to_csv('RF1.csv')


# In[ ]:




