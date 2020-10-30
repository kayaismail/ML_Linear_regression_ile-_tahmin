#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
Yas = veriler.iloc[:,1:4].values
print(Yas)

#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
#ulke sutununu label enkod
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

#ulke sutununu one hot enkod
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
#cinsiyet sutununun lbl encd
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)#ulke en son one hot encod halde 

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


#dataframe birlestirme islemi9
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)# lineer reg. ile boy kilo yas ulke  verilerinden cinsiyet tahmini

######################################################################################################

boy = s2.iloc[:,3 :4].values  #sadece boy sutununu aldik
#print(boy)
sol = s2.iloc[:,:3] #bastaki 3 sutunu aldik
#print(sol) #icergi gor
sag = s2.iloc[:,4:] #sondaki 3 stunu aldik 

veri = pd.concat([sol,sag],axis=1) #s2 yi parcalamistik veri de boyu haric tutarak birlestirdik
print(veri)
#x bagimsiz degisken(girdi) y bagimli degisken(cikti)
 #test train olarak bolduk
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train) #lineer reg ile x~y kullanarak ogren 

y_pred = r2.predict(x_test) #ogrendigin ile x_test e gore boyu tahmin et 

###################################################################################################
#backward elemination
import statsmodels.api as sm
# 22 satir 1 sutundan olusan matris olustur ve ekle
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
print(X_l)

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
print(X_l)



X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
print(X_l)
#backward elemination uyguladigimi veriler uzerinde dogrusal tahmin
x_train, x_test,y_train,y_test = train_test_split(X_l,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train) #lineer reg ile x~y kullanarak ogren 

y_pred = r2.predict(x_test) # x_test e gore boyu tahmin et 










