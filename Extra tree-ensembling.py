import pandas as pd
from sklearn import ensemble,tree,neighbors
from sklearn import preprocessing,model_selection
from sklearn_pandas import CategoricalImputer

titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.info())

#Continous Imputer
cont_impute_feature = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[cont_impute_feature])
titanic_train[cont_impute_feature] = cont_imputer.transform(titanic_train[cont_impute_feature])

#Categorical Imputer
Cat_imputer = CategoricalImputer()
Cat_imputer.fit(titanic_train['Embarked'])
titanic_train['Embarked']=Cat_imputer.transform(titanic_train['Embarked'])

#label Encoding
le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
titanic_train['Embarked'] = le_embarked.transform(titanic_train['Embarked'])

le_Sex = preprocessing.LabelEncoder()
le_Sex.fit(titanic_train['Sex'])
titanic_train['Sex'] = le_Sex.transform(titanic_train['Sex'])

le_Pclass = preprocessing.LabelEncoder()
le_Pclass.fit(titanic_train['Pclass'])
titanic_train['Pclass'] = le_Pclass.transform(titanic_train['Pclass'])


features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
X_train = titanic_train[features]
Y_train = titanic_train['Survived']

et_estimator = ensemble.ExtraTreesClassifier()
et_grid = {'n_estimators':[10,50,100,200],'max_depth':[3,4,5,6,7],'max_features':[2,3,4],'bootstrap':[True,False]}
et_grid_estimator = model_selection.GridSearchCV(et_estimator,et_grid,cv=10,return_train_score=True)
et_grid_estimator.fit(X_train,Y_train)

print(et_grid_estimator.best_score_)
print(et_grid_estimator.best_params_)

final_estimator = et_grid_estimator.best_estimator_
final_estimator.score(X_train,Y_train)

#read test csv
titanic_test = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
print(titanic_test.info())
titanic_test[cont_impute_feature] = cont_imputer.transform(titanic_test[cont_impute_feature])
titanic_test['Pclass'] = le_Pclass.transform(titanic_test['Pclass'])
titanic_test['Sex'] = le_Sex.transform(titanic_test['Sex'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
X_test = titanic_test[features]
titanic_test['Survived']=final_estimator.predict(X_test)
titanic_test.to_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\submission.csv', columns=["PassengerId", "Survived"], index=False)
