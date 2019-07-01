import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
import pylab
import sys
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
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

#1.2 print head and tail of data frame
pd.set_option('display.width', 120)
print('Head:' + '\n')
print(df_econ.head())
print('\n')
print('Tail:' + '\n')
print(df_econ.tail())
print('\n')
'''
#1.3 scatterplot
sns.pairplot(df_econ, size = 1.5)
plt.tight_layout()
plt.show()

#1.4 heatmap
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
plt.show()'''

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
X_std = sc.fit_transform(X)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


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
print('\n' * 3)
print("----------------1. LinearRegression-------------------")


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
plt.ylabel('Residuals(Ridge)')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-0.00250, 0.015])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()
print('MSE train: %.8f, test: %.8f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.8f, test: %.8f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))




#3.1Ridge
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
plt.xlim([-0.00250, 0.015])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()
print('MSE train: %.8f, test: %.8f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.8f, test: %.8f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


#[Lasso regression model] 
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
plt.hlines(y=0, xmin=0.0010, xmax=0.01400, color='black', lw=2)
plt.xlim([0, 0.014])
plt.tight_layout()
# plt.savefig('images/10_09.png', dpi=300)
plt.show()

print('MSE train: %.10f, test: %.10f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.10f, test: %.10f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



#3.3 ElasticNet
print('\n' * 3)
print('--------------------4. Elastic Net---------------------')
elanet = ElasticNet()
param_range =[0.0001,0.001,0.01,0.02,0.046, 0.047, 0.048,0.049, 0.05,0.051, 0.052]
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
print('mean_cross_val_score: ', np.mean(cross_val_score(best_elanet,X_train,y_train,cv=10,scoring="r2")))
best_elanet.fit(X_train_std, y_train)
y_train_pred = best_elanet.predict(X_train)
y_pred = best_elanet.predict(X_test)

print("R^2 for training sample: {}".format(best_elanet.score(X_train_std, y_train)))
rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
print("Root Mean Squared Error training sample: {}".format(rmse))
print("R^2 for testing sample: {}".format(best_elanet.score(X_test_std, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error testing sample: {}".format(rmse))