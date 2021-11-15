# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:13:19 2020

@author: ismet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")
print(veriler)

#dataframedilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy array dçnüşümü
X=x.values
Y=y.values

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

from sklearn.preprocessing import StandardScaler

scl=StandardScaler()

x_olcekli=scl.fit_transform(X)

sc2=StandardScaler()

y_olcekli=np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

