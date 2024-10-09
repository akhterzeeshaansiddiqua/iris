import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
iris = pd.read_csv('Iris.csv')
iris.head()

iris= iris.rename(columns={'SepalLengthCm':'Sepal_Length',
                           'SepalWidthCm':'Sepal_Width',
                           'PetalLengthCm':'Petal_Length',
                           'PetalWidthCm':'Petal_Width'})
iris.head()
# checking null values
iris.isnull().sum()
Id              0
Sepal_Length    0
Sepal_Width     0
Petal_Length    0
Petal_Width     0
Species         0
dtype: int64
# checking if the data is biased or not
iris ['Species'].value_counts()
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
# checking statistical features
iris.describe()
sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter,"Petal_Length","Sepal_Width").add_legend()
sns.pairplot(iris[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']], hue="Species",diag_kind='kde')
# Separate features and target
data=iris.values
# slicing the matrices
X=data[:,0:4]
Y=data[:,5]
print(X.shape)
print(X)
print(Y.shape)
print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)
print(X_train.shape)
print(X_train)
print(y_test.shape)
print(y_test)
print(X_test.shape)
print(X_test)
print(y_train.shape)
print(y_train)
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(X_train,y_train)
prediction1 = model_svc.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1))
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)
prediction3= model_svc.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction3))
# converting categorical variables into numbers
flower_mapping = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
iris['Species']=iris['Species'].map(flower_mapping)
iris.head()
X=iris [['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']].values
y= iris[['Species']].values
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,y)
# accuracy
model.score(X,y)
expected = y
predicted = model.predict(X)
predicted
# summarize the fit of the model
from sklearn import metrics
print(metrics.classification_report(expected, predicted))
# confusion metrics
print(metrics.confusion_matrix(expected, predicted))
# New data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
# Predicting the sizes of the iris flowers
predicted_sizes = model.predict(X_new)
# Output the predicted sizes
print(predicted_sizes)
