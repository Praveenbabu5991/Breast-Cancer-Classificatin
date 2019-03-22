# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:49:36 2018

@author: Bolt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#loading data
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer.keys()

#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])


print(cancer['data'])
print(cancer['DESCR'])
print(cancer['feature_names'])
cancer.data.shape

'''
np.c_[np.array([1,2,3]), np.array([4,5,6])]
array([[1, 4],
       [2, 5],
       [3, 6]])
np.append=Append values to the end of an array
'''
#data
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))

#visualizing the data
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])
sns.countplot(df_cancer['target'])
sns.scatterplot(x='mean area',y='mean smoothness',hue='taregt',data=df_cancer)

plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)

#model training
x=df_cancer.drop(['target'],axis=1)
y=df_cancer['target']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svm_model=SVC()
svm_model.fit(x_train,y_train)

#evaluating the model
y_predict=svm_model.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)

#improving the model =part1
#normalization
'''x'=x-xmin/xmax-xmin
'''
min_train=x_train.min()
range_train=(x_train-min_train).max()
x_train_scaled=(x_train-min_train)/range_train
sns.scatterplot(x=x_train['mean area'],y=y_train['mean smoothness'],hue=y_train)
sns.scatterplot(x=x_train_scaled['mean area'],y=y_train['mean smoothness'],hue=y_train)



min_test=x_test.min()
range_test=(x_test-min_test).max()
x_test_scaled=(x_test-min_test)/range_test

svm_model.fit(x_train_scaled,y_train)
y_predict=svm_model.predict(x_test_scaled)

cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_predict))

#improving the model part2
param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf'] }

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(x_train_scaled,y_train)

grid.best_params_
grid_predictions=grid.predict(x_test_scaled)
cm=confusion_matrix(y_test,grid_predictions)
sns.heatmap(cm,annot=True)




