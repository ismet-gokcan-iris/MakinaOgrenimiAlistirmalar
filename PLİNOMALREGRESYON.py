# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 03:28:13 2020

@author: İSMETGÖKCANİRİS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


veriler=pd.read_csv("maaslar.csv")
print(veriler)

#dataframedilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy array dçnüşümü
X=x.values
Y=y.values



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)



#lineerregresyon
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X,Y)
#görselleştirme
plt.scatter(X,Y,color="red")
plt.plot(x,lr.predict(X), color="blue")
plt.show()
#polinomalregresyon

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
#tahmin
print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

print(lr2.predict(poly_Reg.fit_transform([[6.6]])))
print(lr2.predict(poly_Reg.fit_transform([[11]])))



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


model = Sequential()
model.add(Dense(5, activation='relu', input_dim=2))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))




y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

print(veriler.shape)
veriler.describe()

target_column = ['maas'] 

predictors = list(set(list(veriler.columns))-set(target_column))
veriler[predictors] = veriler[predictors]/veriler[predictors].max()
veriler.describe()

a= veriler[predictors].values
b = veriler[target_column].values

a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

A_train=sc.fit_transform(a_train)
A_test=sc.transform(a_test)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(A_train,a_train, epochs=20)
