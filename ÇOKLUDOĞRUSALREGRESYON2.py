# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:43:51 2020

@author: ismet
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from IPython.display import clear_output


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






from sklearn.linear_model import LogisticRegression

logr=LogisticRegression()
logr.fit(X_train,y_train)

y_pred2=logr.predict(X_test)
print(y_pred2)
print(y_test)




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

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())



X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())




corr = Veriler.corr()
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



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)

y_predd=knn.predict(X_test)


from sklearn.metrics import confusion_matrix

cfm=confusion_matrix(y_test,y_predd)
print(cfm)


model1 = Sequential()

model1.add(Dense(6, activation='relu', input_shape=(6,)))

model1.add(Dense(6, activation='relu'))

model1.add(Dense(1, activation='sigmoid'))



model1.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
                   
model1.fit(X_train,y_train ,epochs=6,validation_data=(x_test, y_test),
          callbacks=[plot_losses] ,batch_size=40, verbose=1)


y_pred10=model1.predict_classes(X_train)
score22 = model1.evaluate(X_train, y_train,verbose=1)

print(score22)









