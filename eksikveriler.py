# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:17:24 2020

@author: ismet
"""
import numpy as np
import pandas as pd

veri = pd.read_csv("eksikveriler.csv")

print(veri)

boy=veri[['boy']]


print(boy)


boykilo = veri[['boy','kilo']]

print(boykilo)

x=10
class insan:
    boy = 180
    
    def koşmak(b):
        return b+10
    
ali=insan()

print(ali.boy)

#eksikveriler

from sklearn.impute import SimpleImputer

imputer =SimpleImputer(missing_values=np.nan,strategy="mean")

yas=veri.iloc[:,1:4].values
print(yas)

imputer=imputer.fit(yas[:,1:4])

yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

        