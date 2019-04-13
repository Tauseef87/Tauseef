import pandas as pd
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.shape)
print(titanic_train.info())
#discover patter which class is majority
titanic_train.groupby('Survived').size()
titanic_test = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
print(titanic_test.shape)
titanic_test['Survived']=0
print(titanic_train.info())
titanic_test.to_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\submission.csv',columns=['Survived','PassengerId'],index=False)
