import pandas as pd
from sklearn import tree
import pydot
import io
from sklearn import preprocessing,model_selection
from sklearn_pandas import CategoricalImputer

#creation of data frames from csv
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.info())

#Preprocessing Stage
#impute missing values for continous data
imputable_cont_feature = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_feature])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_feature]=cont_imputer.transform(titanic_train[imputable_cont_feature])

#impute missing values for categorical data
#imputable_cat_feature = ['Embarked']
#print(type(imputable_cat_feature))  Why we can't take list in fit of cat_imputer
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

#cat_imputer.fit(titanic_train['Cabin']) Getting ValueError: No value is repeated more than once in the column even Cabin contains repated valuse

#label encoding
le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
print(le_embarked.classes_)
titanic_train['Embarked']=le_embarked.transform(titanic_train['Embarked'])

le_Sex = preprocessing.LabelEncoder()
le_Sex.fit(titanic_train['Sex'])
print(le_Sex.classes_)
titanic_train['Sex']=le_Sex.transform(titanic_train['Sex'])

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
Y_train = titanic_train['Survived']

#create an instance of decision tree classifier type
dt_estimator = tree.DecisionTreeClassifier()

#grid_Search
dt_grid = {'criterion':["gini", "entropy"],'max_depth':[3,4,5,6,7],'min_samples_split':[2,10,20,30]}
dt_grid_estimator=model_selection.GridSearchCV(dt_estimator,dt_grid,cv=10)
dt_grid_estimator.fit(X_train,Y_train)

#random search
dt_grid_estimator = model_selection.RandomizedSearchCV(dt_estimator, dt_grid, cv=10, n_iter=20)
dt_grid_estimator.fit(X_train, Y_train)

#Question which is best either GridSearch or RandomSearch
#Question how to check train error and validation errro
#Question - what is mean test score and mean train score and how it is useful means what we can drive from this

#print result
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_estimator_)
final_estimator = dt_grid_estimator.best_estimator_
results = dt_grid_estimator.cv_results_
print(results.get("mean_test_score")) #Question - what is mean test score and mean train score and how it is useful means what we can drive from this
print(results.get("mean_train_score"))
print(results.get("split0_train_score"))
print(results.get("params"))

#read test data

titanic_test = pd.read_csv("C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_feature] = cont_imputer.transform(titanic_test[imputable_cont_feature])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_Sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = final_estimator.predict(X_test)
