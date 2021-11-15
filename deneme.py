# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:30:49 2020

@author: ismet
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:43:51 2020

@author: ismet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

Veriler=pd.read_csv("veriler.csv")
print(Veriler)




yas=Veriler.iloc[:,1:4].values
print(yas)

ulke=Veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing




le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(Veriler.iloc[:,0])
print(ulke)


ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


c=Veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing




le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(Veriler.iloc[:,-1])
print(c)


ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)


sonuc=pd.DataFrame(data=ulke, index=range(22), columns = ["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=yas, index=range(22), columns = ["boy","kilo","yas"])
print(sonuc2)

cinsiyet=Veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=c[:,:1], index =range(22), columns=["cinsiyet"])
print(sonuc3)


s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()


X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


boy=s2.iloc[:,3:4].values
print(boy)

sol=s2.iloc[:,:3]
sağ=s2.iloc[:,4:]

veri=pd.concat([sol,sağ],axis=1)
x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)


r2=LinearRegression()
r2.fit(x_train,y_train)

y_pred=r2.predict(x_test)

import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)
 

X_1=veri.iloc[:,[0,1,2,3,4,5]].values
X_1=np.array(X_1,dtype=float)
model=sm.OLS(boy,X_1).fit()
print(model.summary())