# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:51:11 2020

@author: ismet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("satislar.csv")

print(veriler)

#veri ön işleme

aylar = veriler[["Aylar"]]

print(aylar)

satislar=veriler[["Satislar"]]

print(satislar)


satislar2=veriler.iloc[:,0:1].values

print(satislar2)

from sklearn.model_selection import train_test_split

#verilerin eğitim ve test için bölünmesi
x_train , x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)


#verilerin ölçeklenmesş

from sklearn.preprocessing import StandardScaler


sc=StandardScaler()


X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)

#lineer model inşası

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)


plt.title("aylara göre satis")
plt.xlabel("aylar")
plt.ylabel("satislar")

from sklearn.linear_model import LogisticRegression

logr= LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)