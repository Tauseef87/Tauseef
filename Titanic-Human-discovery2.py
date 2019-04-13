import pandas as pd
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.shape)
print(titanic_train.info())
#discover patter : majority class
titanic_train.groupby(['Survived','Sex']).size()
titanic_test = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
titanic_test['Survived']=0
titanic_test.loc[titanic_test.Sex=='female','Survived']=1
titanic_test.loc[(((titanic_test.Sex =='male') & (titanic_test.Pclass==1)),'Survived')]=1
titanic_test.to_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\submission.csv',columns=['Survived','PassengerId','Sex','Pclass'],index=False)
