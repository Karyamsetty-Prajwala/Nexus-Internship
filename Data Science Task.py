import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import plot_tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

url ="https://doc-0s-04-prod-03-apps-viewer.googleusercontent.com/viewer2/prod-03/pdf/nf2pjcfgv88j1nbsn4uq4ov7sst6juqd/m6fiqeti7l2pvirp7v3pqbsi24d72dij/1704905325000/3/100845707426691717614/APznzaZUocnCKQUmjhQZLTjxMRQa3gjlR0xFqhWlMh19RISGX7nnTzR5e1oUu1dQB1BYZMLHo4ehWzP14lNsP8wTTJqAyu6YuNAzwKj5mdoTlEUwn9QPbLosTuwnuI161HMDL7Y3qGn8qoA1LzDMnzVmCAcaYtKilRxurlvtbWs4GNpN0D4AQfuTieAjV88cbkby2SVK7FJxE9j48ymuA5t2QpX3vT5yXHilt8G-NmFH7wuCBUAvZ1Aq67e3pylsNR3xE2LPYnf2T6GumKwE2RVgm12kwQPaue45rVU7ZIwOPunH2AtAxnqOpIAZ8qQ0fadHGfsrKFkGe0hMnl8YWwLRpKTxRCMzbAXqxO_j2y8fzT-O4MQoDUnHj9NsWWYqyQLrQnM2PWD9lBvfNUzrc4T4hT1OQ_8-dg==?authuser=0&nonce=0hl8llkv11dee&user=100845707426691717614&hash=atd42jmoegusnlms4ofgf7ekhonc9i4v"
df = pd.read_csv(url, header=None, names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)', 'Species'])
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T
ols = df.columns[0:-1]
for j in cols:
    sns.boxplot(a=df[j])
    plt.show()
s1 = df['sepal width (cm)'].quantile(0.25)
s3 = df['sepal width (cm)'].quantile(0.75)
iqr = s3 - s1
df = df[(df['sepal width (cm)'] >= s1-1.5*iqr) & (df['sepal width (cm)'] <= s3+1.5*iqr)]
df.shape
sns.boxplot(a=df['sepal width (cm)'])
plt.show()
p = df.drop("Species",axis=1)
q = df["Species"]
p_train,p_test,q_train,q_test=train_test_split(p,q,test_size=0.3, random_state= 1)
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1 )
dt.fit(p, q)
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1 )
dt.fit(p, q)
dt = DecisionTreeClassifier(random_state=1)
dt.fit(p_train, q_train)

q_pred_train = dt.predict(p_train)
q_pred = dt.predict(p_test)
q_prob = dt.predict_proba(p_test)
print('Accuracy of Decision Tree-Train: ', accuracy_score(q_pred_train, q_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(q_pred, q_test))
print(classification_report(q_test,q_pred))
dt = DecisionTreeClassifier(random_state=1)

params = {'max_depth' : [2,3,4,5],
        'min_samples_split': [2,3,4,5],
        'min_samples_leaf': [1,2,3,4,5]}

gsearch = GridSearchCV(dt, param_grid=params, cv=3)

gsearch.fit(p,q)

gsearch.best_params_
