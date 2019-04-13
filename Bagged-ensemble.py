import pandas as pd
from sklearn import neighbors,ensemble,tree
from sklearn import preprocessing,model_selection
from sklearn_pandas import CategoricalImputer

#creating dataframe
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.info())

#preprocessing stage
#impute mising values for continous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

#label encoding
le_emabarked = preprocessing.LabelEncoder()
le_emabarked.fit(titanic_train['Embarked'])
titanic_train['Embarked'] = le_emabarked.transform(titanic_train['Embarked'])

le_Sex = preprocessing.LabelEncoder()
le_Sex.fit(titanic_train['Sex'])
titanic_train['Sex'] = le_Sex.transform(titanic_train['Sex'])

le_Pclass = preprocessing.LabelEncoder()
le_Pclass.fit(titanic_train['Pclass'])
titanic_train['Pclass'] = le_Pclass.transform(titanic_train['Pclass'])

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
Y_train = titanic_train['Survived']

##bagged ensemble with decision tree
dt_estimator = tree.DecisionTreeClassifier()
bag_estimator = ensemble.BaggingClassifier(base_estimator=dt_estimator)
bag_grid = {'n_estimators':[10,50,100,200,250,300],'base_estimator__max_depth':[3,4,5,6,7,10,11]}
bag_grid_estimator = model_selection.GridSearchCV(bag_estimator,bag_grid,cv=10,return_train_score=True)
bag_grid_estimator.fit(X_train,Y_train)
print(bag_grid_estimator.best_score_)

print(bag_grid_estimator.best_params_)
final_estimator = bag_grid_estimator.best_estimator_
final_estimator.score(X_train, Y_train)    
print(final_estimator.estimators_)

##bagged ensemble with knn
knn_estimator = neighbors.KNeighborsClassifier()
bag_estimator = ensemble.BaggingClassifier(base_estimator=knn_estimator)
bag_grid = {'n_estimators':[10, 50, 100, 200], 'base_estimator__n_neighbors':[3,5,10,20] }
bag_grid_estimator = model_selection.GridSearchCV(bag_estimator, bag_grid, cv=10, return_train_score=True)
bag_grid_estimator.fit(X_train, Y_train)
print(bag_grid_estimator.best_score_)

print(bag_grid_estimator.best_params_)
final_estimator = bag_grid_estimator.best_estimator_
final_estimator.score(X_train, Y_train)    
print(final_estimator.estimators_)