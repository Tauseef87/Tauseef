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

sex_emabarked = preprocessing.LabelEncoder()
sex_emabarked.fit(titanic_train['Sex'])
titanic_train['Sex'] = sex_emabarked.transform(titanic_train['Sex'])

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
Y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
knn_estimator.fit(X_train,Y_train)

scores = model_selection.cross_validate(knn_estimator,X_train,Y_train,cv=10)
test_scores = scores.get("test_score")
print(test_scores.mean())

train_scores = scores.get("train_score")
print(train_scores.mean())

#read test data
titanic_test=pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_emabarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = sex_emabarked.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived']=knn_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\submission.csv", columns=["PassengerId", "Survived"], index=False)
