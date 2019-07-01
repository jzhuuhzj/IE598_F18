from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score


#From Iris dataset get iris data, split it with 90% training and 10% test
iris = datasets.load_iris()
X = iris.data
y = iris.target

#random test
in_sample_scores = []
out_sample_scores = []

dt = DecisionTreeClassifier()

print("Random state     In-sample score     Out-of-sample score")
for k in range(1,11):  
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=k)
    dt.fit(X_train, y_train)
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test) 
    in_sample_accuracy = accuracy_score(y_train, y_pred_train)
    out_sample_accuracy = accuracy_score(y_test, y_pred_test) 
    in_sample_scores.append(in_sample_accuracy)
    out_sample_scores.append(out_sample_accuracy)
    
    print(k, end=' ')
    print('                   ', end=' ')
    print(in_sample_scores[k-1], end=' ')
    print('                   ', end=' ')
    print(out_sample_scores[k-1])
    
    
in_sample_mean = np.mean(in_sample_scores)
in_sample_std = np.std(in_sample_scores)
out_sample_mean = np.mean(out_sample_scores)
out_sample_std = np.std(out_sample_scores)
print('In-sample mean: ', in_sample_mean)
print('In-sample standard deviation: ', in_sample_std)
print('Out-of-sample mean: ', out_sample_mean)
print('Out-of-sample standard deviation: ', out_sample_std)


#cross validation
in_sample_scores = []
out_sample_scores = []

dt = DecisionTreeClassifier()


    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=1)
dt.fit(X_train, y_train)
    
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
    
cv_scores = cross_val_score(dt, X_train, y_train, cv=10)
    
print('\n')

print('CV accuracy scores:')
print(cv_scores)
print('Mean of CV scores: %s' % np.mean(cv_scores))
print('Standard deviation of CV scores: %s' % np.std(cv_scores))

y_pred = dt.predict(X_test)
print('Out-of-sample accuracy:', accuracy_score(y_test, y_pred))
    
    
print("My name is Jingxia Zhu")
print("My NetID is: jingxia6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

