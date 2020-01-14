# Volkan Erek-2149896
# Süleyman Çelebi-2148815
# Mehmet İkiz-2258978
# Adem Acar-2148484
# Emre Can Özkaya-2149292

import pandas as pd
import numpy as np
import os
from sklearn import linear_model as lm
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from numpy import cov
from statistics import stdev

lr = LinearRegression()

datas = pd.read_excel("Data.xlsx")


x_train = datas.iloc[0:48,33:]
y_train = datas.iloc[0:48,32:33]
x_test = datas.iloc[48:,33:]
y_actual = datas.iloc[48:,32:33]

lr.fit(x_train,y_train)
# X_train and y_train include data up to 2013, and these data were used as training.x_test data includes our variables such as gold,dollar,export.
y_pred = lr.predict(x_test)

r2 = r2_score(y_actual,y_pred)

r_ols = sm.OLS(endog=y_train,exog=x_train)
r = r_ols.fit()

print(r.summary())
def cluster():
    x = datas.iloc[:,2:32]
    
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
        
        
    plt.plot(range(1, 11), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    kor = KMeans(n_clusters = 3,init ="k-means++",random_state = 1)
    
    kor.fit(x)
    c_x = kor.cluster_centers_[:, 0]
    c_y = kor.cluster_centers_[:, 1]
    show(c_x,c_y)

def show(X,Y):
    x = datas.iloc[1:,2:32]
    y = datas.iloc[1:,2:32]

    plt.scatter(X,Y,color="r")
    plt.scatter(x,y)
    plt.show()


cluster()

print(y_pred)
print(y_actual)
print(x_test)
