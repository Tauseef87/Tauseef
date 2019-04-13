import pandas as pd
import seaborn as sns
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')

#visual discovery of pattern

#univariate plot
#categorical columns: count/bar plot
#x: categories of feature, y: frequency
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Sex',data=titanic_train)

#continuous columns: histogram/density plot/box-whisker plot
#x: bins of continuous data, y: frequency
sns.distplot(titanic_train['Fare'],kde=False)
#kde = Whether to plot a gaussian kernel density estimate,
#KDE plots encode the density of observations on one axis with height along the other axis

#A bin in a histogram is the block that you use to combine values before getting the frequency. 

#For instance, if you  were creating a histogram of age, the bins might be 0-5, 6-10, 11-15, and so on up to 85 and up (or whatever).
sns.distplot(titanic_train['Fare'], bins=10, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)

#density plot to understand continuous feature
#it doesnt require bins argument
#x: fare y:density
sns.distplot(titanic_train['Fare'], hist=True)
#box-whisker plot to understand continuous feature
sns.boxplot(x='Fare',data=titanic_train)

##Bi-variate plots
#category vs category: factor plot
sns.factorplot(x='Sex',hue='Survived',data=titanic_train,kind='count',size=6)
#kind : {point, bar, count, box, violin, strip} The kind of plot to draw.
#size : scalar, optional Height (in inches) of each facet

#continuous vs categorical: facet grid
sns.FacetGrid(titanic_train,hue='Survived',size=8).map(sns.kdeplot,"Age").add_legend()

#continuous vs continuous: scatter plot
sns.jointplot(x="Fare",y="Age",data=titanic_train)

##multi-variate plots
#3-categorical features
g = sns.FacetGrid(titanic_train,row="Pclass",col="Sex",hue="Survived").map(sns.kdeplot,"Fare")


