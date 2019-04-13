import pandas as pd
from sklearn import neighbors
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

categorical_feature = ['Pclass', 'Sex', 'Embarked']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[categorical_feature])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[categorical_feature]).toarray()
tmp1 = pd.DataFrame(tmp1)
continuous_features = ['Fare', 'Age', 'SibSp', 'Parch']
tmp2 = titanic_train[continuous_features]
tmp = pd.concat([tmp1, tmp2], axis=1)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(tmp)
y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
Knn_grid ={'n_neighbors':[5,7,8,10,20,25,30],'weights':['uniform','distance']}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator,Knn_grid,cv=10,return_train_score='True')
knn_grid_estimator.fit(X_train,y_train)
print(knn_grid_estimator.best_estimator_)
print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.best_params_)
results = knn_grid_estimator.cv_results_
final_estimator = knn_grid_estimator.best_estimator_
print(final_estimator.score(X_train,y_train))

#read test data
titanic_test=pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_emabarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_Sex.transform(titanic_test['Sex'])
titanic_test['Pclass'] = le_Pclass.transform(titanic_test['Pclass'])
tmp1 = ohe.transform(titanic_test[categorical_feature]).toarray()
tmp1 = pd.DataFrame(tmp1)
tmp2 = titanic_test[continuous_features]
tmp = pd.concat([tmp1, tmp2], axis=1)
X_test = scaler.fit_transform(tmp)
titanic_test['Survived']=final_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\submission.csv", columns=["PassengerId", "Survived"], index=False)
