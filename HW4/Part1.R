###### Import libraries ###
library(dplyr)
library(caret)
library(e1071)

###### Question 1 - Step1 ###
#Set the coefficients for the polynomial
set.seed(57)
Aij=runif(16,-2,2)
Aij=matrix(Aij,4,4)
Bi=matrix(runif(4,-2,2))
#Bi=t(Bi)
C=runif(1,-2,2)
print('Aij 4x4 matrix:')
print(Aij,digits=3)
print('Bi 4x1 matrix:')
print(Bi,digits=3)
print(paste('C = ',format(C,digits = 4)))


#Function to calculate the polynominal
polyx <- function(x) {
  max=matrix(x,1,4)
  pol=max%*%Aij%*%t(max)+x%*%Bi+C/20
  pol=pol[1,1]
  return(pol)
}
#test the polynomial
set.seed(56)
x0=runif(4,-2,2)
print('Test the polynomial function')
print('If x = ') 
print(x0,digits = 3)
print(paste(',then Pol(x)=', format(polyx(x0),digits = 4)))

#generate 40000 random numbers
N=10000
Dataset=matrix(runif(4*N,-2,2),N,4)
#Compute Yn
Yn=NULL
for (i in 1:N){
  Yn[i]=sign(polyx(Dataset[i,]))
}
Dataset=data.frame(Dataset)
Dataset=cbind(Dataset,Yn)

#Classify and reduce size
CL1=Dataset[Yn==1,]
CL0=Dataset[Yn==-1,]
CL1=CL1[sample(nrow(CL1), 2500), ]
CL0=CL0[sample(nrow(CL0), 2500), ]

#Standardize the data
Dataset=bind_rows(CL1,CL0)
Dataset.dat=Dataset[,1:4]
Dataset.dat=scale(Dataset.dat)
Dataset=cbind(Dataset.dat,Dataset[,5])
rm(Dataset.dat) #remove unuse variable
colnames(Dataset)[colnames(Dataset)==""] <- "Yn" #rename the true class column
Dataset=data.frame(Dataset)

#Split into train and test set 80/20
intrain=createDataPartition(Dataset$Yn,p=0.8,list = FALSE,times = 1)
TRAIN=Dataset[intrain,]
TRAIN=TRAIN[sample(nrow(TRAIN)),]
TEST=Dataset[-intrain,]
TEST=TEST[sample(nrow(TEST)),]



################ A function to calculate the error of estimation
std_p = function(acc,n) {sqrt(acc*(1-acc)/n)*100}


############## A function to generate a nice confusion matrix and table
generate_cfm=function(predi,truec,cap){
  cfm=table(truec, predi)
  #colnames(cfm)=c('Pred_CL-1','Pred_CL1')
  #rownames(cfm)=c('True_CL-1','True_CL1')
  print(cfm,caption = cap)
  nocc=rowSums(cfm)
  pocp=cfm/nocc
  print(paste('Confustion matrix for the set ',cap))
  print(pocp,digits = 3)
  acc=(cfm[1,1]+cfm[2,2])/length(truec)
  print(paste('Global accuracy =',round(acc,3)))
  std=std_p(acc,length(truec))
  std_1=std_p(pocp[1,1],nocc[1])
  std_2=std_p(pocp[2,2],nocc[2])
  ac=c(acc,pocp[1,1],pocp[2,2])
  st=c(std,std_1,std_2)
  pred_table=data.frame(ac,st)
  rownames(pred_table) = c('Global accuracy','CL-1','CL1')
  colnames(pred_table) = c('Accuracy','Standard deviation')
  print(paste('Prediction table for the set ',cap))
  print(pred_table,digits = 3 )
}

############## Question 2 ###################
N=5000
x=TRAIN[,1:4]
y=TRAIN$Yn

#Alternative way to use svm without specify the type is to conver Y to factor
#model =  svm(Yn~., cost=5, kernel='linear',type='C-classification',scale=FALSE,data=TRAIN)

dat = data.frame(x=x, y=as.factor(y)) # must convert to factor to avoid the svm regression
model =  svm(y~., cost=5, kernel='linear',scale=FALSE,data=dat)
print(summary(model))
S=sum(model$nSV)
s=S/N
print(paste('Number S of support vectors is ',S))
print(paste('And the ratio s = ',s))
PredTRAIN =predict(model,TRAIN[,1:4])
PredTEST =predict(model,TEST[,1:4])
# Check accuracy:
generate_cfm(PredTRAIN, TRAIN$Yn,'TRAIN')
generate_cfm(PredTEST, TEST$Yn,'TEST')

set.seed(1)
tune.out = tune(svm ,y~., data=dat , kernel ="linear",scale=FALSE,
                ranges =list (cost=c(0.001 , 0.01 , 0.1, 1 ,5 ,10 ,100) ))
print(summary(tune.out))