# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:57:28 2019

@author: Duong Hung
"""

import numpy as np
import pandas as pd
import time as tm
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score

def proj_sca(a,b): #scalar projection of a on b
    magn_b=np.sum(b**2)
    #print(magn_b)
    a_b=np.sum(a*b)
    #print(a_b)
    proj_a=(a_b/magn_b)
    return proj_a

def knn(a,data,k):
    noc=len(data)
    nof=len(a)
    dist=np.empty(noc)
    for j in np.arange(nof):
        dist[j]=sum((a[j]-data[j,:])**2)
        dist[j]=dist[j]**0.5
    print(dist)
    sort_ind=np.argsort(dist)
    return sort_ind[:k]
"""=================================================================="""
start_time=tm.time()
#read file
url=('COURIER.csv','CALIBRI.csv','TIMES.csv')
# creat an empty list of data frame
df={}
nof=len(url) # number of data frame
#nof=1
for i in np.arange(nof):
  df[i] = pd.read_csv(url[i])
#print(df[1].axes[1])
#print(df[1].axes[1])

#Remove unwanted columns
discard_columns=('fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w')
for j in np.arange(nof):
  for i in np.arange(len(discard_columns)):
    df[j]=df[j].drop(discard_columns[i],1)
#df1.axes[1] contains names of all column
cl={}   
for j in np.arange(nof):
  length_data=len(df[j])
  cl[j]=df[j][df[j].strength==0.4]
  cl[j]=cl[j][cl[j].italic==0]
  print('Size of CL',j+1,'is',len(cl[j]))  
#print(cl)     
# Combine 3 CL into a full data set DATA
data=pd.concat([cl[0],cl[1],cl[2]])
data_df=data # Data frame of DATA
data=np.array(data.loc[:,'r0c0':'r19c19'].values) # Array of DATA, no CL index

#PART 0
#Compute the means m1 = mean(X1) ....mean(X400) = m400 and the standard deviations
#s1 = std(X1) ....s400 = std(X400)
nofe=data.shape[1] #Number of features
mean_f=np.empty(nofe) # array contains means of all features
std_f=np.empty(nofe) # array contains standard deviation of all features
for i in np.arange(nofe):
    mean_f[i]=np.mean(data[:,i])
    std_f[i]=np.std(data[:,i])

#Standardize the features matrix DATA
sdata = (data - mean_f)/std_f
sdata_df=pd.DataFrame(sdata,data_df.font)
#PART 1
#1.1) Compute the correlation matrix COR of the 400 random variables Y1,..., Y400
cor_mtx=np.corrcoef(data.T)
#1.2) For the matrix COR , compute its 400 eigenvalues λ1 > λ2 > ... > λ400 > 0 , and its 400 eigenvectors v1, v2, ..., v400
eig_val, eig_vec = LA.eig(cor_mtx)
#1.3) Plot the decreasing curve λj versus j for j=1 , 2, ..., 400
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(400),eig_val)
ax.set(title='Curve lamdaj versus j', ylabel='lamda', xlabel='j')
plt.show()

R=np.empty(400)
for i in np.arange(400):
  R[i]=sum(eig_val[:i])/400

for i in np.arange(400):
    if R[i]>0.90:
        min_r=i
        break
print(min_r)    

noc=sdata.shape[0] #Number of cases
scor1=np.empty(noc)
scor2=np.empty(noc)
scor3=np.empty(noc)
for i in np.arange(noc):
  scor1[i]=proj_sca(sdata[i,:],eig_vec[:,0])
  scor2[i]=proj_sca(sdata[i,:],eig_vec[:,1])
  scor3[i]=proj_sca(sdata[i,:],eig_vec[:,2])  
l_cl1=len(cl[0])
l_cl2=len(cl[0])+len(cl[1])
plt.scatter(scor1[:l_cl1],scor2[:l_cl1],color='blue',s=0.1) 
plt.scatter(scor1[l_cl1:l_cl2],scor2[l_cl1:l_cl2],color='black',s=0.1)
plt.scatter(scor1[l_cl2:],scor2[l_cl2:],color='red',s=0.1) 
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scor1[:l_cl1],scor2[:l_cl1],scor3[:l_cl1],color='blue',s=0.1) 
ax.scatter(scor1[l_cl1:l_cl2],scor2[l_cl1:l_cl2],scor3[l_cl1:l_cl2],color='black',s=0.1)
ax.scatter(scor1[l_cl2:],scor2[l_cl2:],scor3[l_cl2:],color='red',s=0.1) 
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scor1[:l_cl1],scor2[:l_cl1],scor3[:l_cl1],color='blue',s=0.1) 
ax.scatter(scor1[l_cl1:l_cl2],scor2[l_cl1:l_cl2],scor3[l_cl1:l_cl2],color='black',s=0.1)
#ax.scatter(scor1,scor2,scor3)
plt.show()    

"""k=15
classifier = KNeighborsClassifier(n_neighbors=k)
score=np.empty(5)
for i in np.arange(5):
  tes_siz=0.2+0.05*i
  sdata_train, sdata_test,font_train,font_test= train_test_split(sdata,data_df.font, test_size=tes_siz)
  classifier.fit(sdata_train, font_train)
  font_pred = classifier.predict(sdata_test)
  print(confusion_matrix(font_test, font_pred))
  print(accuracy_score(font_test, font_pred))
plt.plot(np.arange(5)*0.05+0.2,score)
"""
k_vals=[5, 10 , 15, 20, 30, 40, 50 ,100,200]
k_vals=[2, 3 , 4, 5, 6, 7, 8 ,9,10]

per=np.empty(len(k_vals))
for i in np.arange(len(k_vals)):
  classifier = KNeighborsClassifier(n_neighbors=k_vals[i])
  sdata_train, sdata_test,font_train,font_test= train_test_split(sdata,data_df.font, test_size=0.2)
  classifier.fit(sdata_train, font_train)
  font_pred = classifier.predict(sdata_test)
  #print(confusion_matrix(font_test, font_pred))
  per[i]=metrics.accuracy_score(font_test, font_pred)
#print(perf)  
plt.scatter(k_vals,per)
plt.plot(k_vals,per)
plt.show

"""

end_time=tm.time()
print(end_time-start_time)      