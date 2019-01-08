#!/usr/bin/env python
# coding: utf-8

# In[20]:


history


# In[ ]:


# 1. Python Packages


# In[21]:


import numpy as np
from scipy import sparse

# np.eye(n): n x n digonal matrix
matrix = np.eye(6) 

sparse_matrix = sparse.csr_matrix(matrix)

print(matrix)
print(sparse_matrix)


# In[22]:


# pandas package
import pandas
data = {"Name":["Amy","Bruin","Candy","David"],"City":["Orlando","Miami","Chicago","LA"],
        "Age":["12","15","28","20"],"Height":["151","165","180","172"]}
data_frame = pandas.DataFrame(data)
display(data_frame)

display(data_frame[data_frame.City != "Chicago"])


# In[23]:


# matplotlib package

# activate matplotlib
import matplotlib.pyplot as plt
x = np.linspace(-20,20,10)
y = 2*x
plt.plot(x,y,marker = "o")
plt.show()


# In[ ]:


# 2. KNN classification & regression


# In[24]:


# KNN for classification
# training set
from sklearn.datasets import make_blobs
# import KNN
from sklearn.neighbors import KNeighborsClassifier
# import visualization tools
import matplotlib.pyplot as plt
# import datasets-spliting tools
from sklearn.model_selection import train_test_split

# create new datasets(number of classification = 2)
data = make_blobs(n_samples = 200, centers = 2,random_state = 8)
X,y = data

plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.spring, edgecolor = "k")
plt.show()


# In[37]:


# testing set
import numpy as np
clf = KNeighborsClassifier()
clf.fit(X,y)

# x,y axis
x_min,x_max = X[:,0].min() - 1, X[:,0].max()+1
y_min,y_max = X[:,1].min() - 1, X[:,1].max()+1

# np.meshgrid: draw axis; np.arange: (start, stop, step)
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
# ravel() converts multi-demansional arrays into one-demansion array, column.
# np.c_[] adds y column to the x column. We get [x1,y1];[x2,y2]
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])

# Z must be the same shape as xx
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
plt.scatter(6.75,4.82,marker = '*',c = 'blue',s=200)
plt.show()

print(clf.predict([[6.75,4.82]]))


# In[38]:


data2 = make_blobs(n_samples = 500, centers = 5,random_state = 8)
X2,y2 = data2
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap = plt.cm.spring, edgecolor = "k")
plt.show()


# In[40]:


clf = KNeighborsClassifier()
clf.fit(X2,y2)
x_min,x_max = X2[:,0].min() - 1, X2[:,0].max()+1
y_min,y_max = X2[:,1].min() - 1, X2[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Pastel1)
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap=plt.cm.spring,edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
plt.show()

# model accuracy
print(format(clf.score(X2,y2)))


# In[42]:


# KNN for regression (KNN: K-Nearst Neighbors)
# new dataset
from sklearn.datasets import make_regression
X,y = make_regression(n_features = 1, n_informative = 1, noise = 50, random_state = 8)
plt.scatter(X,y,c = 'orange',edgecolor = 'k')
plt.show()


# In[44]:


# use KNN to build model
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor()
reg.fit(X,y)
z = np.linspace(-3,3,200).reshape(-1,1)
plt.scatter(X,y,c='orange',edgecolor='k')
plt.plot(z,reg.predict(z),c='k',linewidth=3)
plt.title('KNN Regressor')
plt.show()

print(reg.score(X,y))


# In[46]:


# adjust parameter to improve accuracy
from sklearn.neighbors import KNeighborsRegressor
reg2 = KNeighborsRegressor(n_neighbors=2)
reg2.fit(X,y)
plt.scatter(X,y,c='orange',edgecolor='k')
plt.plot(z,reg2.predict(z),c='k',linewidth=3)
plt.title('KNN Regressor: n_neighbors = 2')
plt.show()

print(reg2.score(X,y))


# In[51]:


# An example with KNN classification
# observe data
from sklearn.datasets import load_wine
wine_dataset = load_wine()
print(wine_dataset.keys())
print(wine_dataset['data'].shape)
# 178 observations and 13 features


# In[54]:


# split data (75% for training, 25% for testing)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(wine_dataset['data'],wine_dataset['target'],random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# build model with KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
print(knn)

# socre with testing set
print(knn.score(X_test,y_test))


# In[55]:


# import new data and predict with model
import numpy as np
X_new = np.array([[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]])
prediction = knn.predict(X_new)
print(wine_dataset['target_names'][prediction])


# In[ ]:


# 3. linear model


# In[73]:


# Intro
from sklearn.linear_model import LinearRegression
X = [[1],[4]]
y = [3,5]
lr = LinearRegression().fit(X,y)
z = np.linspace(0,5,20)
# s parameter adjust the size of dots
plt.scatter(X,y,s=80)
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature 
# or array.reshape(1, -1) if it contains a single sample.
plt.plot(z,lr.predict(z.reshape(-1,1)),c = 'k')
plt.title('Straight Line')
plt.show()
print('y={:.3f}'.format(lr.coef_[0]),'x','+ {:.3f}'.format(lr.intercept_))


# In[79]:


from sklearn.datasets import make_regression
X,y = make_regression(n_samples=50,n_features=1,n_informative=1,noise=50,random_state=1)
reg = LinearRegression()
reg.fit(X,y)
z = np.linspace(-3,3,200).reshape(-1,1)
# c parameter is color
plt.scatter(X,y,c = 'b',s=60)
plt.plot(z,reg.predict(z),c = 'k')
plt.title('Linear Regression')
plt.show()

print('{:.2f}'.format(reg.coef_[0]))
print('{:.2f}'.format(reg.intercept_))


# In[83]:


# Linear Regression - OLS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X,y = make_regression(n_samples=100,n_features=2,n_informative=2,random_state=38)
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=8)
lr = LinearRegression().fit(X_train,y_train)

print(format(lr.coef_[:]))
print(format(lr.intercept_))

# socre
print('{:.2f}'.format(lr.score(X_train,y_train)))
print('{:.2f}'.format(lr.score(X_test,y_test)))
# score is high because when making the dataset, we didn't add noise


# In[84]:


# Linear Regression - OLS  with an example
from sklearn.datasets import load_diabetes
X,y = load_diabetes().data,load_diabetes().target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8)
lr = LinearRegression().fit(X_train,y_train)

# socre
print('{:.2f}'.format(lr.score(X_train,y_train)))
print('{:.2f}'.format(lr.score(X_test,y_test)))


# In[97]:


# Ridge Regression - make the model more complex, avoid overfitting
# L2 Regularization - keep all features, but lower the coef of each feature by changing the alpha parameter 
#   to avoid overfitting
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)

print('{:.2f}'.format(ridge.score(X_train,y_train)))
print('{:.2f}'.format(ridge.score(X_test,y_test)))


# adjust alpha parameter: higher alpha lowers the coef of features(near to 0), lower the performance of the model in
# the training set, but is better for generalization
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print('{:.2f}'.format(ridge10.score(X_train,y_train)))
print('{:.2f}'.format(ridge10.score(X_test,y_test)))
# we can higher the aplha to solve the overfitting problem

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print('{:.2f}'.format(ridge01.score(X_train,y_train)))
print('{:.2f}'.format(ridge01.score(X_test,y_test)))

plt.plot(ridge.coef_,'s',label = 'Ridge alpha =1')
plt.plot(ridge10.coef_,'^',label = 'Ridge alpha =10')
plt.plot(ridge01.coef_,"v",label = 'Ridge alpha =0.1')
plt.plot(lr.coef_,'o',label = 'linear regression')
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.legend()




# In[104]:


from sklearn.model_selection import learning_curve, KFold
# define a function to draw learning curve
def plot_learning_curve(est,X,y):
    training_set_size,train_scores,test_scores = learning_curve(est,X,y,train_sizes=np.linspace(.1,1,20),
                                                                cv=KFold(20,shuffle=True),random_state=1)
    estimator_name = est.__class__.__name__
    line=plt.plot(training_set_size,train_scores.mean(axis=1),'--',label="training "+ estimator_name)
    plt.plot(training_set_size,test_scores.mean(axis=1),'-',label="test "+estimator_name,c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.ylim(0,1.1)
    
plot_learning_curve(Ridge(alpha=1),X,y)
plot_learning_curve(LinearRegression(),X,y)
plt.legend(loc=(0,1.05),ncol=2,fontsize=11)


# In[114]:


# Lasso Regression - make the model more complex, avoid overfitting
# L1 Regularization - lower the coef of each feature by changing the alpha parameter, coefs are close to 0, some are 0
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train,y_train)
print('{:.2f}'.format(lasso.score(X_train,y_train)))
print('{:.2f}'.format(lasso.score(X_test,y_test)))
print(format(np.sum(lasso.coef_ != 0)))

# scores are very low. underfitting problem
# use lower alpha to increase the coef, alpha must use with max_iter

lasso01 = Lasso(alpha = 0.1,max_iter = 100000).fit(X_train,y_train)
print('{:.2f}'.format(lasso01.score(X_train,y_train)))
print('{:.2f}'.format(lasso01.score(X_test,y_test)))
print(format(np.sum(lasso01.coef_ != 0)))


# In[118]:


# comparison of Ridge regression and Lasso regression
plt.plot(lasso.coef_,'s',label = 'Lasso alpha =1')
plt.plot(lasso01.coef_,"^",label = 'Lasso alpha =0.1')
plt.plot(ridge01.coef_,'o',label = 'Ridge alpha = 0.1')
plt.legend(ncol=2,loc=(0,1.05))
plt.ylim(-1000,800)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")

# Ridge regression is the first choice. Howvever, for datasets with too many features 
#    and only some of them are important, we should choose the Lasso Regression. 


# In[ ]:


# conslution for linear model:
# 1. linear regression (overfitting problems) ridge regression(L2, keep all features) lasso(L1, keep some features)
# 2. other linear models: logistic regression, linear SVM
# 3. regularization parameter: for ridge and lasso, the parameter is alhpa; for logistic and SVM, the parameter is C
# 4. Training a linear model is very fast, especially for large datasets


# In[151]:


# 4. Naive Bayes - supervised learning     features are independent
import numpy as np
X = np.array([[0,1,0,1],[1,1,1,0],[0,1,1,0],[0,0,0,1],[0,1,1,0],[0,1,0,1],[1,0,0,1]])
y = np.array([0,1,1,0,1,0,0])

# for each feature, calculate the number of y == 0 and y == 1
counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis = 0)
print(format(counts))

# explain: when y == 0, there are 1 for feature 1, 2 for feature 2, 0 for feature 3 and 4 for feature 4
#          when y == 1, there are 1 for feature 1, 3 for feature 2, 3 for feature 3 and 0 for feature 4
# Bayes will predict with the results above.


# In[152]:


# Naive Bayes - Bernoulli Naive Bayes (for data that follow bernoulli distribution, which is 0-1 distribution)
# import bernoulli bayes
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X,y)
# the data we are going to predict
Next_Day = [[0,0,1,0]]
pre = clf.predict(Next_Day)
if pre == [1]:
    print("It's going to rain")
else:
    print("Sunny day. No worries!")
    
# prediction accuracy
print(clf.predict_proba(Next_Day))


# In[178]:


# how bernoulli naive bayes works
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X,y = make_blobs(n_samples = 500, centers = 5,random_state = 8)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 8)
nb = BernoulliNB()
nb.fit(X_train,y_train)
print('{:.3f}'.format(nb.score(X_test,y_test)))

import matplotlib.pyplot as plt
# define the max and min of x-axis and y-axis
x_min, x_max = X[:,0].min()-0.5,X[:,0].max()+0.5
y_min, y_max = X[:,1].min()-0.5,X[:,1].max()+0.5
# paint different classifications as different color
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z = nb.predict(np.c_[(xx.ravel(),yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.spring)
# scatter the training set and the testing set
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolor='k')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.cool,marker = '*',edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: BernoulliNB')
plt.show()


# In[179]:


# Naive Bayes - Gaussian Naive Bayes (for data that follow gaussian distribution, which is normal distribution)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print('{:.3f}'.format(gnb.score(X_test,y_test)))
  
z = gnb.predict(np.c_[(xx.ravel(),yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.spring)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolor='k')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.cool,marker = '*',edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: GaussianNB')
plt.show()


# In[180]:


# Naive Bayes - Multinomial Naive Bayes (for data that follow multinomial distribution)
from sklearn.naive_bayes import MultinomialNB
# data preprocessing tools: MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# make all Xs non-negative
# MinMaxScaler: make all X 0-1
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_scaled,y_train)
mnb.score(X_test_scaled,y_test)


z = mnb.predict(np.c_[(xx.ravel(),yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.spring)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolor='k')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.cool,marker = '*',edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: MultinomialNB')
plt.show()


# In[181]:


# An example with Naive Bayes - first step: have a look at the data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['target_names'])
print(cancer['feature_names'])

# data, target(calssification values),target_names(classification names),DESCR(data description),feature_names


# In[182]:


# An example with Naive Bayes - second step: build model(Gaussian NB)
# split data into training and testing set
X,y = cancer.data,cancer.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 38)
print(X_train.shape)
print(X_test.shape)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
print('{:.3f}'.format(gnb.score(X_train,y_train)))
print('{:.3f}'.format(gnb.score(X_test,y_test)))
  


# In[183]:


# An example with Naive Bayes - third step: make prediction
# classification with prediction
print(format(gnb.predict(X[[312]])))
# actual classification
print(y[312])


# In[184]:


# An example with Naive Bayes - learning curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,
                        train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    plt.grid()
    plt.plot(train_sizes,train_scores_mean,'o-',color = "r", label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color = "g", label="Cross-validation score")
    plt.legend(loc="lower right")
    return plt

title = "Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=100,test_size = 0.2,random_state=0)
estimator=GaussianNB()
plot_learning_curve(estimator,title,X,y,ylim=(0.9,1.01),cv=cv,n_jobs=4)
plt.show()


# In[197]:


# decision tree classification
# first step: prepare data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.model_selection import train_test_split
wine = datasets.load_wine()
# select only first two features
X = wine.data[:,:2]
y = wine.target
X_train,X_test,y_train,y_test = train_test_split(X,y)

# second step: build model
# max_depth: the depth of decision tree (questions asked)
clf = tree.DecisionTreeClassifier(max_depth=1)
clf.fit(X_train,y_train)
# get parameters of the model

# third step: plot the classifier
# define the color of classification and scatters
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: (max_depth = 1)')
plt.show()


# In[198]:


clf2 = tree.DecisionTreeClassifier(max_depth = 3)
clf2.fit(X_train,y_train)
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z = clf2.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: (max_depth = 3)')
plt.show()


# In[199]:


clf3 = tree.DecisionTreeClassifier(max_depth = 5)
clf3.fit(X_train,y_train)
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z = clf3.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: (max_depth = 5)')
plt.show()

# overfitting problem


# In[202]:


# Random Forests - classification & regression

# build random forest
from sklearn.ensemble import RandomForestClassifier
wine = datasets.load_wine()
X = wine.data[:,:2]
y = wine.target
X_train, X_test,y_train,y_test = train_test_split(X,y)
# decide how many trees in the random forest
forest = RandomForestClassifier(n_estimators = 8, random_state = 3)
forest.fit(X_train,y_train)


# In[203]:


# parameters in Random Forests
# 1. bootstrap: True means sampling with replacement. Becasuse of the replacement of sample, some data may be lost.
# 2. max_features: max features that the model selects to fit the data. default is the maximum of feature numbers.
# 3. n_estimators: number of trees in forest
# 4. n_jobs: same-time processing. set it same to the CPU cores. n_jobs = -1 (maximum)

cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z = forest.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)

plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Classifier: RandomForest')
plt.show()

# random forest: don't request data pre-precessing
# not good for superhigh dementional data; not good for sparse data (linear model is better)


# In[213]:


# An example of Decision Tree
import pandas as pd
data = pd.read_csv('/Users/cherry/Desktop/adult.csv',header = None, index_col = False,names = ['age','company type','weight','education',
                                                                         'edu_times','marriage','job','family','race',
                                                                        'gender','earnings','loss','worktime',
                                                                         'nationality','income'])
data_lite = data[['age','company type','education','gender','worktime','job','income']]
display(data_lite.head())

# use get_dummies to convert text data to number data
data_dummies = pd.get_dummies(data_lite)
print(list(data_lite.columns))
print(list(data_dummies.columns))

data_dummies.head()


# In[232]:


# define feature values
features = data_dummies.loc[:,'age':'job_ Transport-moving']
X = features.values

y = data_dummies['income_ >50K'].values

X_train, X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
go_dating_tree = tree.DecisionTreeClassifier(max_depth = 5)
go_dating_tree.fit(X_train,y_train)
print('{:.2f}'.format(go_dating_tree.score(X_test,y_test)))


Mr_Z = [[37,40,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
dating_dec = go_dating_tree.predict(Mr_Z)
if dating_dec == 1:
    print("Go")
else:
    print("No")


# In[257]:


# Support Vector Machine(SVM) - nonlinear problems
# covert/project data to high-dimension through kernel trick
# two methods: polynomial kernel and Radial basis function kernel (RBF, Gaussian kernel)
# support vectors: data that is on the classification boundary


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples = 50, centers = 2, random_state = 6)
# c is Penalty parameter C of the error term.

# make kernel linear
clf = svm.SVC(kernel = 'linear',C=1000)
clf.fit(X,y)

# s is the size of points
plt.scatter(X[:,0],X[:,1],c = y, s = 30, cmap = plt.cm.Paired)

# gca stands for 'get current axis'.
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
# Return coordinate matrices from coordinate vectors.
YY,XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# paint the boundary
ax.contour(XX,YY,Z,colors = 'k',levels = [-1,0,1],alpha = 0.5,linestyles=['--','-','--'])
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s = 100, linewidth = 1, facecolors = 'none')
plt.show()

# '-' is the classifier; points on the '--' are support vectors
# Maximum Margin Separating Hyperplane: the total distance between the '-' and all support vectors are maximum


# In[260]:


# make kernel RBF
clf_rbf = svm.SVC(kernel = 'rbf', C = 1000)
clf_rbf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c = y, s = 30, cmap = plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
YY,XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf_rbf.decision_function(xy).reshape(XX.shape)
ax.contour(XX,YY,Z,colors = 'k',levels = [-1,0,1],alpha = 0.5,linestyles=['--','-','--'])
ax.scatter(clf_rbf.support_vectors_[:,0],clf_rbf.support_vectors_[:,1],s = 100, linewidth = 1, facecolors = 'none')
plt.show()


# In[265]:


# compare different SVM kernels

from sklearn.datasets import load_wine
def make_meshgrid(x,y,h=.02):
    x_min, x_max = x.min()-1,x.max()+1
    y_min, y_max = y.min()-1,y.max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx,yy
def plot_contours(ax,clf,xx,yy,**params):
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx,yy,Z,**params)
    return out

wine = load_wine()
X = wine.data[:,:2]
y = wine.target

C = 1.0 # para for the SVM regularization
models = (svm.SVC(kernel='linear',C=C),svm.LinearSVC(C=C),svm.SVC(kernel = 'rbf',gamma=0.7,C=C),
          svm.SVC(kernel = 'poly',degree=3,C=C))
models = (clf.fit(X,y) for clf in models)

titles = ('SVC with linear kernel','LinearSVC','SVC with RBF kernel','SVC with polynomial (degree3) kernel')

fig,sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
X0,X1 = X[:,0],X[:,1]
xx,yy = make_meshgrid(X0,X1)

for clf,title,ax in zip(models,titles,sub.flatten()):
    plot_contours(ax,clf,xx,yy,cmap = plt.cm.plasma,alpha = 0.8)
    ax.scatter(X0,X1,c=y,cmap=plt.cm.plasma,s=20,edgecolors = 'k')
    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    
plt.show()


# In[268]:


# RBF SVM parameter : gamma
C = 1.0  # regularization parameter
models = (svm.SVC(kernel = 'rbf',gamma=0.1,C=C),svm.SVC(kernel = 'rbf',gamma=1,C=C),
          svm.SVC(kernel = 'rbf',gamma=10,C=C))
models = (clf.fit(X,y) for clf in models)

titles = ('gamma = 0.1','gamma = 1','gamma = 10')

fig,sub = plt.subplots(1,3,figsize = (10,3))

X0,X1 = X[:,0],X[:,1]
xx,yy = make_meshgrid(X0,X1)

for clf, title,ax in zip(models,titles, sub.flatten()):
    plot_contours(ax,clf,xx,yy,cmap=plt.cm.plasma,alpha = 0.8)
    ax.scatter(X0,X1,c=y, cmap=plt.cm.plasma,s=20,edgecolors='k')
    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    
plt.show()
    
# smaller gamma: underfitting problem; larger gamma: overfitting problem
# smaller C: underfitting problem(simple model); larger C: overfitting problem(complicated model)

# SVM is good for datasets in all types. For super larger datasets, it costs more time.
# SVM is good for low-dimensional data and high-dimensional data.
# SVM requires data pre-processing and parameter adjustments.
# three important parameters for SVM: kernel, gamma/degree, C.


# In[269]:


# an example with SVM 
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())


# In[270]:


from sklearn.model_selection import train_test_split
X,y = boston.data,boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 8)
print(X_train.shape)
print(y_train.shape)


# In[273]:


from sklearn.svm import SVR
for kernel in ['linear','rbf']:
    svr = SVR(kernel=kernel)
    svr.fit(X_train,y_train)
    print(kernel,'training:{:.3f}'.format(svr.score(X_train,y_train)))
    print(kernel,'test:{:.3f}'.format(svr.score(X_test,y_test)))


# In[274]:


# low score may result from the large range of data
plt.plot(X.min(axis=0),'v',label = 'min')
plt.plot(X.max(axis=0),'^',label = 'max')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('features')
plt.ylabel('feature magnitude')
plt.show()


# In[275]:


# data pre-processing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.plot(X_train_scaled.min(axis=0),'v',label = 'train set min')
plt.plot(X_train_scaled.max(axis=0),'^',label = 'train set max')
plt.plot(X_test_scaled.min(axis=0),'v',label = 'test set min')
plt.plot(X_test_scaled.max(axis=0),'^',label = 'test data max')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('scaled features')
plt.ylabel('scaled feature magnitude')
plt.show()


# In[277]:


for kernel in ['linear','rbf']:
    svr = SVR(kernel=kernel)
    svr.fit(X_train_scaled,y_train)
    print('new',kernel,'training:{:.3f}'.format(svr.score(X_train_scaled,y_train)))
    print('new',kernel,'test:{:.3f}'.format(svr.score(X_test_scaled,y_test)))
    
# adjust parameters
svr = SVR(C=100,gamma = 0.1)
svr.fit(X_train_scaled,y_train)
print('after',kernel,'training:{:.3f}'.format(svr.score(X_train_scaled,y_train)))
print('after',kernel,'test:{:.3f}'.format(svr.score(X_test_scaled,y_test)))


# In[280]:


# neural networks - Multilayer Perception (MLP)
# After creating hidden layers, we should do retifying nonlinearity (relu), or tangens hyperbolicus(tanh)

import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-5,5,200)

# plot retifying nonlinearity
plt.plot(line,np.tanh(line),label = 'tanh')
plt.plot(line,np.maximum(line,0),label = 'relu')

plt.legend(loc = 'best')
plt.xlabel('x')
plt.ylabel('relu(x) and tanh(x)')
plt.show()

# relu : replace all values less than 0 with 0
# tanh : covert values to between -1 and 1
# why we do this: simplify features for better learning on complicated nonlinear datasets


# In[292]:


# for small dataset, usually 10 nodes. for large datasets, we can set more nodes or set more hidden layers

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X = wine.data[:,:2]
y = wine.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
mlp = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [1000,])
mlp.fit(X_train,y_train)

# activation: 'identity' f(x) = x; 'logistic' F(x) = 1/[1+exp(-x)],from 0-1 ; relu,default ; tanh
# alpha: L2 regularization; default: 0.0001
# hidden_layer_sizes: default [100,] - 100 nodes, 1 layer; [10,10] - 2 layers, 10 nodes/layer
# The solver for weight optimization:
#  ‘lbfgs’ is an optimizer in the family of quasi-Newton methods; ‘sgd’ refers to stochastic gradient descent.
#  ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
#   The default solver ‘adam’ works well on large datasets. For small datasets,‘lbfgs’ is faster and perform better.


# In[293]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
x_min, x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z = mlp.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap = cmap_light)

plt.scatter(X[:,0],X[:,1], c=y, edgecolor = 'k',s = 60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier: solver=lbfgs")
plt.show()


# In[286]:


# 10 nodes
mlp_20 = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [10,])
mlp_20.fit(X_train,y_train)

Z1 = mlp_20.predict(np.c_[xx.ravel(),yy.ravel()])
Z1 = Z1.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z1,cmap = cmap_light)

plt.scatter(X[:,0],X[:,1], c=y, edgecolor = 'k',s = 60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier: nodes=10")
plt.show()

# more nodes make boundries more smooth


# In[294]:


# 2 layer
mlp_2L = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [10,10,10])
mlp_2L.fit(X_train,y_train)

Z2 = mlp_2L.predict(np.c_[xx.ravel(),yy.ravel()])
Z2 = Z2.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z2,cmap = cmap_light)

plt.scatter(X[:,0],X[:,1], c=y, edgecolor = 'k',s = 60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier: 2layers")
plt.show()

# more layers make boundaries more smooth


# In[295]:


# tanh
mlp_tanh = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [10,10,10],activation = 'tanh')
mlp_tanh.fit(X_train,y_train)

Z3 = mlp_tanh.predict(np.c_[xx.ravel(),yy.ravel()])
Z3 = Z3.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z3,cmap = cmap_light)

plt.scatter(X[:,0],X[:,1], c=y, edgecolor = 'k',s = 60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier: 2layers with tanh")
plt.show()

# tanh makes boundries more smooth


# In[296]:


# alpha
mlp_alpha = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [10,10,10],activation = 'tanh',alpha = 1)
mlp_alpha.fit(X_train,y_train)

Z4 = mlp_alpha.predict(np.c_[xx.ravel(),yy.ravel()])
Z4 = Z4.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z4,cmap = cmap_light)

plt.scatter(X[:,0],X[:,1], c=y, edgecolor = 'k',s = 60)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("MLPClassifier: alpha=1")
plt.show()

# larger alpha makes model more simple (less smooth)


# In[ ]:


# an exmaple with neural networks (MLP)
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist
print(format(mnist.data.shape[0],mnist.data.shape[1]))

# data preprocessing
X = mnist.data/255
y = mnist.target
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=5000,test_size=1000,random_state=62)

# build model
mlp_hw = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[100,100],activation='relu',alpha = 1e-5,random_state =62)
mlp_hw.fit(X_train,y_train)
print(format(mlp_hw.score(X_test,y_test)))

# test
# image-processing tool: PIL
from PIL import Image
image = Image.open('4.png').convert('F')
image = image.resize((28,28))
arr=[]
for i in range(28):
    for j in range(28):
        pixel = 1.0 - float(image.getpixel((j,i)))/255
        arr.append(pixel)
arr1 = np.array(arr).reshape(1,-1)
print(format(mlp_hw.predict(arr1)[0]))


# In[ ]:


# MLP in sk-learn is only good for small-size data
# other library: keras,tensorflow,theano
# MM is good for single-feature data; for more features, random forset and GD decision trees work better

