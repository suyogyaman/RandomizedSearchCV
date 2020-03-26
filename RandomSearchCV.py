# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:45:14 2020

@author: suyog
"""

#Importing Libraries
import pandas as pd

#Importing Dataset and feature selection
data = pd.read_csv('Purchase.csv')
X = data.iloc[:,2:4]
y = data.iloc[:,4]

#Splitting the dataset into test set and train set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Applying Standarization using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#Fitting Random Forest Classification to Traning set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#Prediction for test dataset
y_pred = classifier.predict(X_test)

#Metrics Evaluation through confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

#Lets explore RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

est = RandomForestClassifier(n_jobs=-1)

rf_p_dist = {'max_depth': [3,5,10,None],
             'n_estimators': [10,100,200,300,400,500],
             'max_features' : randint(1,3),
             'criterion' : ['gini','entropy'],
             'bootstrap' : [True,False],
             'min_samples_leaf' : randint(1,4)
            }

def hypertuning_rscv(est,p_distr,nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est,param_distributions=p_distr,
                                   n_jobs=-1,n_iter=nbr_iter,cv=9)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params,ht_score

rf_parameters,rf_ht_score = hypertuning_rscv(est,rf_p_dist,40,X,y)

#Fitting as per the rf_parameters
classifier = RandomForestClassifier(
        n_jobs=-1,
        bootstrap=True,
        criterion='entropy',
        max_depth= 3,
        max_features= 2,
        min_samples_leaf= 3,
        n_estimators= 100)

#Metrics Evaluation through confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

#Cross Validation good for selecting models
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(classifier,X,y,cv=10,scoring='accuracy').mean()


















