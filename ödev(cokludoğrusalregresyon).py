# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 00:20:08 2020

@author:İSMETGÖKCANİRİS
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler=pd.read_csv("odev_tenis0.csv")
print(veriler)


outlk=veriler.iloc[:,0:1].values
print(outlk)

from sklearn import preprocessing


le=preprocessing.LabelEncoder()
outlk[:,0]=le.fit_transform(veriler.iloc[:,0])
print(outlk)

ohe=preprocessing.OneHotEncoder()
outlk=ohe.fit_transform(outlk).toarray()
print(outlk)

play=veriler.iloc[:,-2:].values
print(play)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
play[:,-2]=le.fit_transform(veriler.iloc[:,-2])
print(play)

ohe=preprocessing.OneHotEncoder()
play=ohe.fit_transform(play).toarray()
print(play)





sonuc=pd.DataFrame(data=outlk, index=range(14), columns=["sunny","overcast","rainy"])
print(sonuc)

temp=veriler.iloc[:,1:3].values
print(temp)

sonuc2=pd.DataFrame(data=temp,index=range(14),columns=["temperature","humidity"])
print(sonuc2)

sonuc3=pd.DataFrame(data=play, index=range(14), columns=["no","yes","false","true"])
print(sonuc3)


alms=sonuc3.iloc[:,0:1].values
print(alms)

sonuc4=pd.DataFrame(data=alms,index=range(14),columns=["no"])
print(sonuc4)


s=pd.concat([sonuc,sonuc2], axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)
s3=pd.concat([s,sonuc4],axis=1)
print(s3)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc4,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler


sc=StandardScaler()


X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)




humidity=s3.iloc[:,3:4].values
print(humidity)

sol=s2.iloc[:,:3]
sağ=s2.iloc[:,4:]

veri=pd.concat([sol,sağ],axis=1)
x_train,x_test,y_train,y_test = train_test_split(veri,humidity,test_size=0.33,random_state=0)


r2=LinearRegression()
r2.fit(x_train,y_train)

y_pred=r2.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=veri, axis=1)

X_1=veri.iloc[:,[0,1,2,3,4,5]].values
X_1=np.array(X_1,dtype=float)
model=sm.OLS(humidity,X_1).fit()
print(model.summary())

X_1=veri.iloc[:,[1,2,3,4,5]].values
X_1=np.array(X_1,dtype=float)
model=sm.OLS(humidity,X_1).fit()
print(model.summary())

X_1=veri.iloc[:,[1,3,4,5]].values
X_1=np.array(X_1,dtype=float)
model=sm.OLS(humidity,X_1).fit()
print(model.summary())

