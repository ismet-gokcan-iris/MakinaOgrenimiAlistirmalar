# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 01:38:52 2020

@author: ismet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing



veriler=pd.read_csv("maaslar.csv")
print(veriler)

#dataframedilimleme(slice)
x=veriler.iloc[:,2:3]
y=veriler.iloc[:,2: ]


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

print("svr")
from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


print("decision")
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))




import seaborn as sns
corr =x.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);








