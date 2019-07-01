from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/'
                      'wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.1,
                                                   random_state=0,
                                                   stratify=y)


out_sample_scores = []

n_list = [1, 2, 5, 10, 15, 25, 50, 100, 250, 500, 1000]

print('N_estimators     In-sample accuracy     Out-of-sample accuracy')
for n in n_list:
    print(n, end='')
    
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=n,
                                    random_state=1,
                                    n_jobs=-1)
    forest.fit(X_train, y_train)  
    cv_scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('               ', end='')
   
    print("%0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2), end='')
    
    y_test_pred = forest.predict(X_test)
    out_sample_score = accuracy_score(y_test, y_test_pred) 
    out_sample_scores.append(out_sample_score)
    print('            ', end='')
    print(out_sample_score)
    
print('\n')
    

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=15,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


print("My name is Jingxia Zhu")
print("My NetID is: jingxia6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")










