# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:18:30 2020

@author: ismet
"""

import numpy as np
import pandas as pd


veri=pd.read_csv("veriler.csv")



from sklearn.impute import SimpleImputer


imputer =SimpleImputer(missing_values=np.nan,strategy="mean")


Yas=veri.iloc[0:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)



ülke=veri.iloc[:,0:1].values

print(ülke)

from sklearn import preprocessing

le= preprocessing.LabelEncoder()


ülke[:,0]=le.fit_transform(veri.iloc[:,0])


print(ülke)

ohe=preprocessing.OneHotEncoder()

ülke=ohe.fit_transform(ülke).toarray()

print(ülke)

sonuç=pd.DataFrame(data=ülke,index =range(22),columns=['fr','tr','us'])

print(sonuç)

sonuç2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuç2)

cinsiyet=veri.iloc[:,-1].values

print(cinsiyet)

sonuç3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuç3)

s=pd.concat([sonuç,sonuç2],axis=1)

print(s)

s2=pd.concat([s,sonuç3],axis=1)

print(s2)


from sklearn.model_selection import train_test_split


x_train, x_test , y_train, y_test = train_test_split(s,sonuç3,test_size=0.33,random_state=0)