# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:17:19 2020

@author: ismet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler=pd.read_csv("maaslar_yeni.csv")
print(veriler)

x=veriler.iloc[:,2:3]
y=veriler.iloc[:,5:]
X=x.values
Y=y.values


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X,Y)

print("lr ols")
model=sm.OLS(lr.predict(X),X)
print(model.fit().summary())

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
print(x_poly)
lr2=LinearRegression()
lr2.fit(x_poly,y)
print("poly ols ")
model2=sm.OLS(lr2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

from sklearn.preprocessing import StandardScaler

scl=StandardScaler()

x_olcekli=scl.fit_transform(X)

sc2=StandardScaler()

y_olcekli=np.ravel(sc2.fit_transform(Y.reshape(-1,1)))





from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("svr ols")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)


model4=sm.OLS(dtr.predict(X),X)
print(model4.fit().summary())


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(X,Y.ravel())
print("rfr ols")
model5=sm.OLS(rfr.predict(X),X)
print(model5.fit().summary())