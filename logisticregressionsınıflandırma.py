# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:02:31 2020

@author: ismetgökcaniris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers



layer=layers.Dense(32, input_shape=(50, ))



veriler=pd.read_csv("veriler.csv")
print(veriler)

x=veriler.iloc[:,1:4].values #bağımsızdeğişken
y=veriler.iloc[:,4:].values #bağımlıdeğişken


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train= sc.fit_transform(x_train)
X_test=sc.transform(x_test)




from sklearn.linear_model import LogisticRegression

logr= LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix

cfm=confusion_matrix(y_test,y_pred)
print(cfm)

import seaborn as sns
corr =veriler.corr()
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

from keras import models

model=models.Sequential()

model.add(layers.Dense(32, input_shape=(50, )))
