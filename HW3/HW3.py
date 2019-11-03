# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:14:51 2019

@author: Duong Hung
"""
# Import libraries
import time as tm
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('nbagg')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

#Import data
url=('https://raw.githubusercontent.com/duonghung86/hello-world/master/COURIER.csv',
     'https://raw.githubusercontent.com/duonghung86/hello-world/master/CALIBRI.csv',
    'https://raw.githubusercontent.com/duonghung86/hello-world/master/TIMES.csv')
# creat an empty list of data frame
df=pd.DataFrame() # Data frame contain all data
nof=len(url) # number of input files
font_name=['COURIER','CALIBRI','TIMES']
for fn in url:
  dat = pd.read_csv(fn)
  df=[df,dat]
  df= pd.concat(df)

#Discard columns
dis_col=('fontVariant','m_label','orientation','m_top','m_left','originalH',
         'originalW','h','w')
for dc in dis_col:
    df=df.drop(dc,1)
print('Display names of Data frame after discarding \n',df.columns)

#Filter the data
df=df[df.strength==0.4]
df=df[df.italic==0]

CL1o=df[df.font==font_name[0]]
CL2o=df[df.font==font_name[1]]
CL3o=df[df.font==font_name[2]]

data=np.array(df.loc[:,'r0c0':'r19c19'].values)

nofe=data.shape[1] #Number of features
mean_f=np.mean(data,axis=0) # array contains means of all features
std_f=np.std(data,axis=0)   # array contains standard deviation of all features

sdata = (data - mean_f)/std_f   
cor_mtx=np.corrcoef(data.T)
#EIGEN VECTOR
eig_val, eig_vec = LA.eig(cor_mtx)
R=np.cumsum(eig_val)/400

rtest=R-0.35
rtest=rtest[rtest>0]
a=np.where(R==(rtest[0]+0.35))
a=a[0][0]
a+=1

rtest=R-0.6
rtest=rtest[rtest>0]
b=np.where(R==(rtest[0]+0.6))
b=b[0][0]
b+=1

sdatf=pd.DataFrame(sdata)
sdatf['font']=np.array(df.font)
CL1=sdatf[sdatf.font==font_name[0]]
CL2=sdatf[sdatf.font==font_name[1]]
CL3=sdatf[sdatf.font==font_name[2]]
cl1_train, cl1_test = train_test_split(CL1, test_size=0.2, random_state=42)
cl2_train, cl2_test = train_test_split(CL2, test_size=0.2, random_state=42)
cl3_train, cl3_test = train_test_split(CL3, test_size=0.2, random_state=42)
train=[cl1_train,cl2_train,cl3_train]
train=pd.concat(train)
f_train=train.font
train=pd.DataFrame(train)
train=np.array(train.iloc[:,:400].values)

test=[cl1_test,cl2_test,cl3_test]
test=pd.concat(test)
f_test=test.font
test=np.array(test.iloc[:,:400].values)

vm_a=eig_vec[:,:a]
A=np.matmul(train,vm_a)
km_A = KMeans(n_clusters=3, random_state=0).fit(A)
print(km_A.inertia_)
print(km_A.cluster_centers_)
km_A.labels_