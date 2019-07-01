#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:18:25 2018

@author: ericzheng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")
#Part1: EDA######################################################################
#1.0 read data and drop Date Column
df_original = pd.read_csv('MLF_GP2_EconCycle.csv')
df_econ = df_original.drop('Date', 1)

#1.1 print summary statistics of data frame
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 100)
print('Summary Statistics:' + '\n')
summary = df_econ.describe()
print(summary)
print('\n')

#1.1.2 print head and tail of original data frame
pd.set_option('display.width', 120)
print('Head:' + '\n')
print(df_econ.head())
print('\n')
print('Tail:' + '\n')
print(df_econ.tail())
print('\n')

#1.1.3 scatterplot of original data frame
sns.pairplot(df_econ, size = 1.5)
plt.tight_layout()
plt.title('scatterplot of original data frame')
plt.show()

#1.1.4 heatmap of original data frame
corr=df_econ.corr()
fig, ax = plt.subplots(figsize=(12,12))
hm = sns.heatmap(corr,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2g',
                 annot_kws={'size': 10},
                 ax = ax,
                 yticklabels=df_econ.columns,
                 xticklabels=df_econ.columns)

plt.tight_layout()
plt.title('heatmap of original data frame')
plt.show()

#Part2: PREPROCESSING############################################################
#2.0 insert columns
#2 yr
result12=(df_econ['CP1M']/df_econ.iloc[:,1])
df_econ.insert(df_econ.shape[1]-4,'CP1M_T2Y', result12)
result32=(df_econ['CP3M']/df_econ.iloc[:,1])
df_econ.insert(df_econ.shape[1]-4,'CP3M_T2Y', result32)
result62=(df_econ['CP6M']/df_econ.iloc[:,1])
df_econ.insert(df_econ.shape[1]-4,'CP6M_T2Y', result62)

#3 yr
result13=(df_econ['CP1M']/df_econ.iloc[:,2])
df_econ.insert(df_econ.shape[1]-4,'CP1M_T3Y', result13)
result33=(df_econ['CP3M']/df_econ.iloc[:,2])
df_econ.insert(df_econ.shape[1]-4,'CP3M_T3Y', result33)
result63=(df_econ['CP6M']/df_econ.iloc[:,2])
df_econ.insert(df_econ.shape[1]-4,'CP6M_T3Y', result63)

#5 yr
result15=(df_econ['CP1M']/df_econ.iloc[:,3])
df_econ.insert(df_econ.shape[1]-4,'CP1M_T5Y', result15)
result35=(df_econ['CP3M']/df_econ.iloc[:,3])
df_econ.insert(df_econ.shape[1]-4,'CP3M_T5Y', result35)
result65=(df_econ['CP6M']/df_econ.iloc[:,3])
df_econ.insert(df_econ.shape[1]-4,'CP6M_T5Y', result65)

#7 yr
result17=(df_econ['CP1M']/df_econ.iloc[:,4])
df_econ.insert(df_econ.shape[1]-4,'CP1M_T7Y', result17)
result37=(df_econ['CP3M']/df_econ.iloc[:,4])
df_econ.insert(df_econ.shape[1]-4,'CP3M_T7Y', result37)
result67=(df_econ['CP6M']/df_econ.iloc[:,4])
df_econ.insert(df_econ.shape[1]-4,'CP6M_T7Y', result67)

#10 yr
result110=(df_econ['CP1M']/df_econ.iloc[:,5])
df_econ.insert(df_econ.shape[1]-4,'CP1M_T10Y', result110)
result310=(df_econ['CP3M']/df_econ.iloc[:,5])
df_econ.insert(df_econ.shape[1]-4,'CP3M_T10Y', result310)
result610=(df_econ['CP6M']/df_econ.iloc[:,5])
df_econ.insert(df_econ.shape[1]-4,'CP6M_T10Y', result610)




df_econ.insert(df_econ.shape[1]-4,'pct_T1Y',df_econ.iloc[:,0].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_T2Y',df_econ.iloc[:,1].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_T3Y',df_econ.iloc[:,2].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_T5Y',df_econ.iloc[:,3].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_T7Y',df_econ.iloc[:,4].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_T10Y',df_econ.iloc[:,5].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_CP1M',df_econ.iloc[:,6].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_CP3M',df_econ.iloc[:,7].pct_change(periods=1))
df_econ.insert(df_econ.shape[1]-4,'pct_CP6M',df_econ.iloc[:,8].pct_change(periods=1))

df_econ=df_econ.iloc[2:,:]


#1.3 scatterplot of modified data frame
sns.pairplot(df_econ, size = 1.5)
plt.tight_layout()
plt.title('scatterplot of modified data frame')
plt.show()

#1.4 heatmap of modified data frame
corr=df_econ.corr()
fig, ax = plt.subplots(figsize=(12,12))
hm = sns.heatmap(corr,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2g',
                 annot_kws={'size': 10},
                 ax = ax,
                 yticklabels=df_econ.columns,
                 xticklabels=df_econ.columns)

plt.tight_layout()
plt.title('heatmap of modified data frame')
plt.show()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 100)
print('Summary Statistics:' + '\n')
summary = df_econ.describe()
print(summary)
print('\n')

#2.1 select columns

X=df_econ.iloc[:,:-4]  
y1=df_econ.iloc[:,-3]
y2=df_econ.iloc[:,-2]
y3=df_econ.iloc[:,-1]

y=y1

#2.2 Test-Train Split
X_train, X_test, y_train, y_test = train_test_split (X, y, 
                                                     test_size = 0.2, 
                                                     random_state=42)

print('Summary Statistics:' + '\n')
summary = df_econ.describe()
print(summary)
print('\n')

#2.3 standardize
# standardize all features in X
sc = StandardScaler()
X_std = pd.DataFrame(sc.fit_transform(X))
X_std.columns = X.columns
X_train_std = pd.DataFrame(sc.fit_transform(X_train))
X_train_std.columns = X.columns
X_test_std = pd.DataFrame(sc.transform(X_test))
X_test_std.columns = X.columns
#boxplot for X_std
X_std_boxplot = X_std.boxplot(figsize=[35,20])
plt.show()

#2.4 Variance ratio plot
cov_mat = np.cov(X_std.T) 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals) 
var_exp = [(i / tot) for i in 
           sorted(eigen_vals, reverse=True)] 
cum_var_exp = np.cumsum(var_exp) 
plt.bar(range(1,37),var_exp, alpha=0.5, align='center',label='individual explained variance') 
plt.step(range(1,37),cum_var_exp, where='mid',label='cumulative explained variance') 
plt.ylabel('Explained variance ratio') 
plt.xlabel('Principal component index') 
plt.legend(loc='best') 
plt.show()

#2.5 PCA
#pca = PCA(n_components=2)
#X_train_pca = pca.fit_transform(X_train_std)
#X_test_pca = pca.transform(X_test_std)


#Part3: Model fitting and evaluation######################################
#3.1 Linear Regression
print('\n' * 3)
print("---------1. LinearRegression----------")


lr = LinearRegression()
lr.fit(X_train_std, y_train)
y_train_pred = lr.predict(X_train_std)
y_test_pred=lr.predict(X_test_std)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(LinearRegression)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.02, 0.051])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()
print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))




#3.2Ridge
print('\n' * 3)
print("-----------------2. Ridge---------------------")

c=[0.000001,0.0005,0.001,0.018,0.01,0.02,0.018,0.046, 0.047, 0.048,0.049, 0.06,0.0718,0.072]
params = {'alpha':c}
ridge = Ridge()
searcher = GridSearchCV(ridge, params,cv=10)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print('for Ridge Regression: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV R^2: ", searcher.best_score_)

param_ridge=searcher.best_params_
best_ridge=searcher.best_estimator_

ridge = Ridge(alpha=0.0718)
ridge.fit(X_train_std, y_train)
y_train_pred = ridge.predict(X_train_std)
y_test_pred = ridge.predict(X_test_std)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Ridge)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.04])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


#3.3 Lasso regression model
print('\n' * 3)
print('--------------------3. Lasso---------------------')
params = {'alpha':c}
lasso = Lasso()
searcher = GridSearchCV(lasso, params,cv=10)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print('for Lasso Regression: ')
print("Best CV params: ", searcher.best_params_)
print("Best CV R^2: ", searcher.best_score_)

lasso = Lasso(alpha=0.000001)
lasso.fit(X_train_std, y_train)
y_train_pred = lasso.predict(X_train_std)
y_test_pred = lasso.predict(X_test_std)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Lasso)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.04])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


##3.4 ElasticNet
print('\n' * 3)
print('---------4. ElasticNet model--------------')
elanet = ElasticNet()
param_range =[0.0001,0.001,0.01,0.02,0.046,0.05]
param_grid = [{'alpha':param_range,'l1_ratio':param_range}]
GS = GridSearchCV(estimator=elanet,
                      param_grid=param_grid,
                      cv=10,
                      n_jobs=-1) 
GS.fit(X_train_std,y_train)
best_elanet=GS.best_estimator_
print('for ElasticNet: ')
print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)
print('mean_cross_val_score: ', np.mean(cross_val_score(best_elanet,X_train_std,y_train,cv=10,scoring="r2")))


best_elanet.fit(X_train_std, y_train)
y_train_pred = best_elanet.predict(X_train_std)
y_pred = best_elanet.predict(X_test_std)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(ElasticNet)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-2.0, xmax=5.0, color='black', lw=2)
plt.xlim([-0.02, 0.05])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_elanet.score(X_train_std, y_train),
        best_elanet.score(X_test_std, y_test)))


#4 Ensemble******
#4.1 Decision Tree Regression
print()
print('Decision Tree Regression')

# =============================================================================
# param_range1 = [1,2,5,10]
# param_range2 = [0.01, 0.1, 0.5]
# param_range3 = [0.01, 0.05, 0.1, 0.5]
# =============================================================================
param_grid = [{"max_depth":[ 4, 5, 6],
               "min_samples_split":[0.00001, 0.0001, 0.001], 
               "min_samples_leaf":[0.02, 0.03, 0.04, 0.05, 0.06], 
               "max_features":["auto","sqrt","log2"],
               "random_state":[42]}]
DT = DecisionTreeRegressor()
GS = GridSearchCV(estimator=DT, param_grid = param_grid,cv=10)
GS.fit(X_train_std,y_train)
best_DT=GS.best_estimator_

print('Decision Tree Regression: ')
print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)

print('mean cross_val_score: ', np.mean(cross_val_score(best_DT, X_train_std, y_train, cv=10, scoring="r2")))

best_DT.fit(X_train_std, y_train)

y_train_pred = best_DT.predict(X_train_std)
y_pred=best_DT.predict(X_test_std)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_pred,  y_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Decision Tree)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.046])
plt.tight_layout()
plt.show()
print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_DT.score(X_train_std, y_train),
        best_DT.score(X_test_std, y_test)))

# =============================================================================
# #importance plot
# importances = pd.Series(data=best_DT.feature_importances_,
#                         index= X_train_std.columns)
# sorted_index = np.argsort(importances)[::-1]
# 
# # Sort importances
# importances_sorted = importances.sort_values()
# 
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, 
#                             X.columns[sorted_index[f]], 
#                             importances[sorted_index[f]]))
# # Draw a horizontal barplot of importances_sorted
# importances_sorted.plot(kind='barh', color='lightgreen',figsize=(10,10))
# plt.title('Features Importances')
# plt.show()
# 
# =============================================================================

#4.2 Bagging
param_grid = [{'base_estimator':[best_DT],'n_estimators':[50, 100, 500],'bootstrap':[True],'max_samples':[0.9, 0.95, 0.99, 1],'max_features':[0.7, 0.8, 0.9, 1],'random_state':[42]}]
bagging=BaggingRegressor()

GS = GridSearchCV(estimator=bagging, param_grid = param_grid, cv=10)
GS.fit(X_train_std,y_train)
best_bagging=GS.best_estimator_

print('Bagging: ')
print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)

print('mean cross_val_score: ', np.mean(cross_val_score(best_bagging, X_train_std, y_train, cv=10, scoring="r2")))

best_bagging.fit(X_train_std, y_train)

y_train_pred = best_bagging.predict(X_train_std)
y_pred=best_bagging.predict(X_test_std)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Bagging)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.046])
plt.tight_layout()
plt.show()
print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_bagging.score(X_train_std, y_train),
        best_bagging.score(X_test_std, y_test)))

#4.3 Random forest
print()
print("Random Forest")

param_grid = [{'n_estimators':[50,100, 300],'max_features':[0.8, 0.9, 1]}]
forest=RandomForestRegressor()

GS = GridSearchCV(estimator=forest, param_grid = param_grid, cv=10)
GS.fit(X_train_std,y_train)
best_forest=GS.best_estimator_

print('Random Forrest Regression: ')
print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)

print('mean cross_val_score: ', np.mean(cross_val_score(best_forest, X_train_std, y_train, cv=10, scoring="r2")))

best_forest.fit(X_train_std, y_train)

y_train_pred = best_forest.predict(X_train_std)
y_pred=best_forest.predict(X_test_std)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Random Forrest)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.046])
plt.tight_layout()
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_forest.score(X_train_std, y_train),
        best_forest.score(X_test_std, y_test)))


#importance plot
importances = pd.Series(data=best_forest.feature_importances_,
                        index= X_train_std.columns)
sorted_index = np.argsort(importances)[::-1]

# Sort importances
importances_sorted = importances.sort_values()

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            X.columns[sorted_index[f]], 
                            importances[sorted_index[f]]))
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen',figsize=(10,10))
plt.title('Features Importances')
plt.show()

#4.4 Ada Boost
print()
print("Ada Boost")
param_grid = [{'base_estimator':[best_DT],'n_estimators':[ 69, 70, 75, 80 ],
               'learning_rate':[0.3, 0.31, 0.35, 0.4],
       'loss':['linear','square','exponential']}]

Ada=AdaBoostRegressor()

GS = GridSearchCV(estimator=Ada, param_grid = param_grid, cv=10)
GS.fit(X_train_std,y_train)
best_Ada=GS.best_estimator_

print('AdaBoost Regression: ')
print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)

print('mean cross_val_score: ', np.mean(cross_val_score(best_Ada, X_train_std, y_train, cv=10, scoring="r2")))

best_Ada.fit(X_train_std, y_train)

y_train_pred = best_Ada.predict(X_train_std)
y_pred=best_Ada.predict(X_test_std)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(AdaBoost)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.046])
plt.tight_layout()
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_Ada.score(X_train_std, y_train),
        best_Ada.score(X_test_std, y_test)))


#4.5 Gradiant Boost
print()
print("Gradiant Boost")

param_grid = [{'n_estimators':[50, 51, 52],
               'learning_rate':[ 0.10, 0.11, 0.12],
       'loss':['ls', 'lad', 'huber', 'quantile'], 'alpha':[ 0.1, 0.2, 0.3] }]

GB=GradientBoostingRegressor()

GS = GridSearchCV(estimator=GB, param_grid=param_grid,cv=10)
GS.fit(X_train_std,y_train)
best_GB=GS.best_estimator_

print("Best CV params: ", GS.best_params_)
print("Best CV R^2: ", GS.best_score_)

print('mean cross_val_score: ', np.mean(cross_val_score(best_GB, X_train_std, y_train, cv=10, scoring="r2")))

best_GB.fit(X_train_std, y_train)

y_train_pred = best_GB.predict(X_train_std)
y_pred=best_GB.predict(X_test_std)


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals(Gradiant Boost)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.015, 0.046])
plt.tight_layout()
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RMSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred)**(1/2),
        mean_squared_error(y_test, y_test_pred)**(1/2)))
print('R^2 train: %.10f, test: %.10f' % (
        best_GB.score(X_train_std, y_train),
        best_GB.score(X_test_std, y_test)))



<<<<<<< HEAD
=======
#1.0 read data and drop Date Column
df_original = pd.read_csv('MLF_GP2_EconCycle.csv')
df_econ = df_original.drop('Date', 1)
>>>>>>> 56d83e650d7ed0d9577d4248b9da5f70450c4c00

#1.1 print summary statistics of data frame
pd.set_option('display.width', 100)
print('Summary Statistics:' + '\n')
summary = df.describe()
print(summary)

#1.2 scatterplot
sns.pairplot(df_econ, size = 1.5)
plt.tight_layout()
plt.show()

#1.3 heatmap
corr=df_econ.corr()
fig, ax = plt.subplots(figsize=(12,12))
hm = sns.heatmap(corr,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2g',
                 annot_kws={'size': 10},
                 ax = ax,
                 yticklabels=df_econ.columns,
                 xticklabels=df_econ.columns)


plt.tight_layout()
plt.show()


