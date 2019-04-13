# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 08:44:23 2019

@author: tauseef.ur.rahman
"""
import pandas as pd
titanic_train = pd.read_csv("C:/Users/tauseef.ur.rahman/Desktop/Python-Docs/Titanic/train.csv")
titanic_test = pd.read_csv("C:/Users/tauseef.ur.rahman/Desktop/Python-Docs/Titanic/test.csv")
print(titanic_test.shape)
print(titanic_train.shape)
print(titanic_test.info())
print(titanic_train.info())
#which class is majority
print(titanic_train.groupby('Survived').size())
titanic_test['Survived']=0
print(titanic_test.head(10))
titanic_test.loc[(titanic_test['Sex']=='female') & (titanic_test['Embarked']=='Q'),['Sex','Pclass','Embarked']] 
titanic_test.loc[((titanic_test['Sex'] == 'female') & (titanic_test['Embarked'] =='Q')), ['Sex','Pclass']] 
titanic_test.loc[((titanic_test.Sex == 'male') & (titanic_test.Pclass==1)),'Survived']=1
#titanic_test.loc[titanic_test.Sex == 'female',['Sex']=1, # how to set male as 0
#titanic_test.to_csv("C:/Users/tauseef.ur.rahman/Desktop/Python-Docs/Titanic/submittion.csv",
#                    columns=['PassengerId','Survived','Sex'],index = False)
import seaborn as sns
