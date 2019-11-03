
####################### CODE PART ####################################

#### PRELIMINARY ####
rm(list=ls())
fon_nam = c('COURIER','CALIBRI','TIMES')
ful_dat=list()
# Import data 
for (fn in fon_nam){
  fdat=data.frame(read.csv(paste(fn,'.csv',sep = "")))
  ful_dat=rbind(ful_dat,fdat)
  #str(ful_dat)
}
#fdat1=data.frame(read.csv(paste(fon_nam[1],'.csv',sep = "")))
#fdat2=data.frame(read.csv(paste(fon_nam[2],'.csv',sep = "")))
#fdat3=data.frame(read.csv(paste(fon_nam[3],'.csv',sep = "")))
#ful_dat2=rbind(fdat1,fdat2,fdat3) #Merge vertically
#rm(fdat1,fdat2,fdat3)
#Discard 9 columns
drops = c('fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w')
ful_dat = ful_dat[,!(names(ful_dat) %in% drops)]

#Define DATA and CLi
oDATA = ful_dat[ which(ful_dat$strength==0.4 & ful_dat$italic ==0), ]
CL1 =oDATA[ which(oDATA$font==fon_nam[1]),]
CL2 =oDATA[ which(oDATA$font==fon_nam[2]),]
CL3 =oDATA[ which(oDATA$font==fon_nam[3]),]
nDATA=oDATA[c(-3:-1)]
#Standardlize the data set

sDATA=oDATA
sDATA[4:403]=scale(sDATA[4:403])
nDATA=data.matrix(sDATA[c(-3:-1)])

#Correlation matrix
corr=cor(nDATA)
ei=eigen(corr)
lamb=ei$values
Rj=cumsum(lamb)/400

#### QUESTION 1 ####

#Find a
rtest=Rj-0.35
rtest=Rj[ which(rtest>0)]
a=which(Rj %in% rtest[1])

#Find b
rtest=Rj-0.6
rtest=Rj[ which(rtest>0)]
b=which(Rj %in% rtest[1])

#Split the data set into Train set and Test set rationally
#Data set will be splitted into 2 set : TRAIN and TEST
TRAIN=NULL 
TEST=NULL
for (i in 1:3){
  CL=sDATA[ which(sDATA$font==fon_nam[i]),]
  samp=sample.int(n=nrow(CL), size = floor(.8*nrow(CL)), replace = F)
  TRAIN=rbind(TRAIN,CL[samp,])
  TEST=rbind(TEST,CL[-samp,])
}
TRAIN= TRAIN[sample(nrow(TRAIN)),]
TEST= TEST[sample(nrow(TEST)),]

#Verify that the sizes m1 m2 m3 of classes CL1 , CL2, CL3 verify mj/NTST ~ nj /N for j=1,2,3
NTST=nrow(TEST)
N=nrow(sDATA)
for (i in 1:3){
  CL=TEST[ which(TEST$font==fon_nam[i]),]
  m=nrow(CL)
  n=nrow(sDATA[ which(sDATA$font==fon_nam[i]),])
  print(paste('For j =',i,', mj/NTST=',round(m/NTST,4),'vs nj/N ',round(n/N,4)))
}

#### QUESTION 2 ####
# Find Ai
vm=ei$vectors[,1:5]
# Split Ai into train and test set
A.train=data.matrix(TRAIN[c(-3:-1)])%*%vm
A.test=data.matrix(TEST[c(-3:-1)])%*%vm

# Apply kNN on Ai
library(class) #load the library that contains knn function
cl.train=TRAIN$font # This will be the classes of the train set
cl.test=TEST$font
# Run knn function
sta_tim=Sys.time()
Ai.knn=knn(A.train,A.test,cl=cl.train,k=5)
Ai.cfm = table(Ai.knn,cl.test) #create confusion matrix
end_tim=Sys.time()
print('The confusion matrix of data set Ai is')
print(Ai.cfm)
# Check the accuracy
accuracy = function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
print(paste('The percentage of successful classifications on Ai is ',round(accuracy(Ai.cfm),2),"%", sep=""))
print(paste('The computing time is',round(end_tim-sta_tim,4),'second'))
# Compare with the accuracy of the sDATA
sD.train=data.matrix(TRAIN[c(-3:-1)])
sD.test=data.matrix(TEST[c(-3:-1)])
# Run knn function
sta_tim=Sys.time()
sDATA.knn=knn(sD.train,sD.test,cl=cl.train,k=5)
sDATA.cfm = table(sDATA.knn,cl.test) #create confusion matrix
end_tim=Sys.time()
print('The confusion matrix of data set sDATA is')
print(sDATA.cfm)
# Check the accuracy
print(paste('The percentage of successful classifications on sDATA is ',round(accuracy(sDATA.cfm),2),"%", sep=""))
print(paste('The computing time is',round(end_tim-sta_tim,4),'second'))

#### QUESTION 3 ####

# Find Gi
g.vm=ei$vectors[,a+1:b]
# Split Gi into train and test set
G.train=data.matrix(TRAIN[c(-3:-1)])%*%g.vm
G.test=data.matrix(TEST[c(-3:-1)])%*%g.vm

# Apply kNN on Gi
library(class) #load the library that contains knn function
cl.train=TRAIN$font # This will be the classes of the train set
cl.test=TEST$font
# Run knn function
sta_tim=Sys.time()
Gi.knn=knn(G.train,G.test,cl=cl.train,k=5)
Gi.cfm = table(Gi.knn,cl.test) #create confusion matrix
end_tim=Sys.time()
print('The confusion matrix of data set Gi is')
print(Gi.cfm)
# Check the accuracy
print(paste('The percentage of successful classifications on Gi is ',round(accuracy(Gi.cfm),2),"%", sep=""))
print(paste('The computing time is',round(end_tim-sta_tim,4),'second'))

#### QUESTION 4 ####
A.cluster=NULL
cost=Inf
set.seed(10)
for (i in 1:10){
  clus = kmeans(A.train,3,nstart = i)
  print(clus$tot.withinss)
  if (cost>clus$tot.withinss){
    cost=clus$tot.withinss
    A.cluster=clus
  }
}
print(A.cluster$tot.withinss)

#### QUESTION 5 ####

#identify H1,H2,H3

pre_dat=TRAIN
pre_dat$font=A.cluster$cluster
#h1.tr=A.train
H1=pre_dat[ which(pre_dat$font==1),]
H2=pre_dat[ which(pre_dat$font==2),]
H3=pre_dat[ which(pre_dat$font==3),]
H1=data.matrix(H1[c(-3:-1)])
H2=data.matrix(H2[c(-3:-1)])
H3=data.matrix(H3[c(-3:-1)])

#compute Cost(CL1,CL2,CL3)
cen.h1=colMeans(h1cl)
co.h1cl=sum((h1cl-cen.h1)^2)
print(co.h1)
#a function calc_SS that returns the within sum-of-squares for a (numeric) data.frame
calc_SS = function(df) {sum(as.matrix(dist(df)^2)) / (nrow(df))}

#wit_ss=calc_SS(CL1[c(-3:-1)])


#Verify the calc_SS function
wss1=calc_SS(H1)
print(wss1)
#wss2=calc_SS(H2)
#wss3=calc_SS(H3)

# Compute Cost(cl1,cl2,cl3)
#c.cl=NULL
#c.cl[1]=calc_SS(CL1[c(-3:-1)])
#c.cl[2]=calc_SS(CL2[c(-3:-1)])
#c.cl[3]=calc_SS(CL3[c(-3:-1)])
#cost.cl=sum(c.cl)
#Compute P and Q




