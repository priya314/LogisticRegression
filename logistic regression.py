import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import math
#data=pd.read_csv(r'/Users/priyasridharan/Documents/7 layers.csv')
titanic_data=pd.read_csv(r'/Users/priyasridharan/Downloads/titanic-extended/full.csv')
titanic_data.head(12)


print("no of passengers "+str(len(titanic_data.index)))
sns.countplot(x="Survived",data=titanic_data)
sns.countplot(x="Survived",hue="Sex",data=titanic_data)
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist()
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)
titanic_data.isnull()
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(),yticklabels=False)
sns.boxplot(x="Pclass",y="Age",data=titanic_data)
titanic_data.head(5)
titanic_data.drop("Cabin",axis=1,inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)
titanic_data.drop("Age",axis=1,inplace=True)
titanic_data.drop("Lifeboat",axis=1,inplace=True)
titanic_data.drop("Body",axis=1,inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)
titanic_data.head(5)
sns.heatmap(titanic_data.isnull(),yticklabels=False)
titanic_data.isnull().sum()
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)
embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head(5)
Pcl=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
Pcl.head(5)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data.head(5)
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic_data.head()
titanic_data.drop(['Name_wiki','Hometown','Boarded','Destination'],axis=1,inplace=True)
titanic_data.drop((['Pclass']),axis=1,inplace=True)
titanic_data.head()
X=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)