# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:54:51 2019

@author: tauseef.ur.rahman
"""

import pandas as pd
from sklearn import tree
import sklearn
import pydot
import io

#print(sklearn.__version__)

#creation of data frames from csv
titanic_train = pd.read_csv("C:/Users/tauseef.ur.rahman/Desktop/Python-Docs/Titanic/train.csv")
#print(titanic_train.info())

features = ['Pclass']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
classifer = tree.DecisionTreeClassifier() #object created for  DecisionTreeClassifier

#learn the pattern automatically
classifer.fit(X_train, y_train)
#print(classifer.tree_)
#get the logic or model learned by Algorithm
#issue: not readable
#https://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/

dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/tauseef.ur.rahman/Desktop/Python-Docs/Titanic/tree.pdf")

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())
X_test = titanic_test[features]
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)