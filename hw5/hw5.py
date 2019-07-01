import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC




df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print('Wine data excerpt:\n\n:', df_wine.head())

##########EDA##############################################
#scatterplot
sns.set(font_scale=1)
sns.pairplot(df_wine, size=2.5)
plt.tight_layout()
plt.savefig('scatter.png', dpi=300)
plt.show()

#heatmap
cm = np.corrcoef(df_wine.values.T)
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(18,18)) 
hm = sns.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
           fmt='0.09f',
            annot_kws={'size': 10},
           yticklabels=df_wine.columns,
            xticklabels=df_wine.columns, ax=ax)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
plt.show()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#standardize
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


##########logistic regression##############################
lr = LogisticRegression(C=1.0, random_state=42)
lr.fit(X_train_std, y_train)
lr_train_pred = lr.predict(X_train_std)
lr_test_pred = lr.predict(X_test_std)

lr_train_score = accuracy_score(y_train, lr_train_pred)
lr_test_score = accuracy_score(y_test, lr_test_pred)
print('Logistic regression train/test accuracies %.3f/%.3f'
      % (lr_train_score, lr_test_score))


##########SVM##############################################
svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
svm.fit(X_train_std, y_train)
svm_train_pred = svm.predict(X_train_std)
svm_test_pred = svm.predict(X_test_std)

svm_train_score = accuracy_score(y_train, svm_train_pred)
svm_test_score = accuracy_score(y_test, svm_test_pred)
print('SVM train/test accuracies %.3f/%.3f'
      % (svm_train_score, svm_test_score))


##########PCA&LR###########################################
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
pca_lr_train_pred = lr.predict(X_train_pca)
pca_lr_test_pred = lr.predict(X_test_pca)

pca_lr_train_score = accuracy_score(y_train, pca_lr_train_pred)
pca_lr_test_score = accuracy_score(y_test, pca_lr_test_pred)
print('PCA_LR train/test accuracies %.3f/%.3f'
      % (pca_lr_train_score, pca_lr_test_score))


##########PCA&SVM##########################################
svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
svm.fit(X_train_pca, y_train)
pca_svm_train_pred = lr.predict(X_train_pca)
pca_svm_test_pred = lr.predict(X_test_pca)
pca_svm_train_score = accuracy_score(y_train, pca_svm_train_pred)
pca_svm_test_score = accuracy_score(y_test, pca_svm_test_pred)
print('PCA_SVM train/test accuracies %.3f/%.3f'
      % (pca_svm_train_score, pca_svm_test_score))


##########LDA&LR###########################################
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
lda_lr_train_pred = lr.predict(X_train_lda)
lda_lr_test_pred = lr.predict(X_test_lda)
lda_lr_train_score = accuracy_score(y_train, lda_lr_train_pred)
lda_lr_test_score = accuracy_score(y_test, lda_lr_test_pred)
print('LDA_LR train/test accuracies %.3f/%.3f'
      % (lda_lr_train_score, lda_lr_test_score))


##########LDA&SVM##########################################
svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
svm.fit(X_train_lda, y_train)
lda_svm_train_pred = svm.predict(X_train_lda)
lda_svm_test_pred = svm.predict(X_test_lda)
lda_svm_train_score = accuracy_score(y_train, lda_svm_train_pred)
lda_svm_test_score = accuracy_score(y_test, lda_svm_test_pred)
print('LDA_SVM train/test accuracies %.3f/%.3f'
      % (lda_svm_train_score, lda_svm_test_score))


##########kPCA#############################################
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)

svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
svm.fit(X_train_kpca, y_train)

kpca_train_pred = svm.predict(X_train_kpca)
kpca_test_pred = svm.predict(X_test_kpca)
kpca_train_score = accuracy_score(y_train, kpca_train_pred)
kpca_test_score = accuracy_score(y_test, kpca_test_pred)
print('KPCA train/test accuracies %f/%f'
      % (kpca_train_score, kpca_test_score))

gammas = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
kpca_lr_train_scores = []
kpca_lr_test_scores = []
kpca_svm_train_scores = []
kpca_svm_test_scores = []

for i in range(0,16):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gammas[i])
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    kpca_lr_train_pred = svm.predict(X_train_kpca)
    kpca_lr_test_pred = svm.predict(X_test_kpca) 
    lr = LogisticRegression()
    lr.fit(X_train_kpca, y_train)

    print("Gamma: ", gammas[i])
    kpca_lr_train_scores.append(accuracy_score(y_train, kpca_lr_train_pred))
    kpca_lr_test_scores.append(accuracy_score(y_test, kpca_lr_test_pred))
    print('KPCA_LR train/test accuracies %f/%f'
          % (kpca_lr_train_scores[i], kpca_lr_test_scores[i]))
    
    svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
    svm.fit(X_train_kpca, y_train)
    kpca_svm_train_pred = svm.predict(X_train_kpca)
    kpca_svm_test_pred = svm.predict(X_test_kpca)
    
    kpca_svm_train_scores.append(accuracy_score(y_train, kpca_svm_train_pred))
    kpca_svm_test_scores.append(accuracy_score(y_test, kpca_svm_test_pred))
    print('KPCA_SVM train/test accuracies %f/%f'
          % (kpca_svm_train_scores[i], kpca_svm_test_scores[i]))

plt.plot(gammas, kpca_lr_train_scores, label='KPCA_LR training')
plt.plot(gammas, kpca_lr_test_scores, label='KPCA_LR test')
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.legend()
plt.title("KPCA_LR accuracy scores")
plt.show()

plt.plot(gammas, kpca_svm_train_scores, label='KPCA_SVM training')
plt.plot(gammas, kpca_svm_test_scores, label='KPCA_SVM test')
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.legend()
plt.title("KPCA_SVM accuracy scores")
plt.show()
    
print("My name is Jingxia Zhu")
print("My NetID is: jingxia6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
