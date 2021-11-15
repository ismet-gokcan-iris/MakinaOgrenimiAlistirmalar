# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:52:48 2020

@author: ismet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("veriler.csv")
print(veriler)


x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)



from sklearn.preprocessing import PolynomialFeatures

poly_Reg=PolynomialFeatures(degree=4)
x_poly=poly_Reg.fit_transform(X)
print(x_poly)


lr2=LinearRegression()
lr2.fit(x_poly,y)

#görselleştirme
plt.scatter(X,Y,color="red")
plt.plot(X,lr2.predict(poly_Reg.fit_transform(X)),color="blue")
plt.show()




from sklearn.linear_model import LogisticRegression

logr=LogisticRegression()
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix

cfm=confusion_matrix(y_test,y_pred)
print(cfm)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)

y_predd=knn.predict(X_test)

cfm2=confusion_matrix(y_test,y_predd)
print(cfm2)


