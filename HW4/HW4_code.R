###### Import libraries ###
library(dplyr)
library(caret)
library(e1071)
library(xlsx)
library(reshape)
library(gplots)
# Question 1 - Step1 ###
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

# Question 1 - Step 2 ####

# generate 40000 random numbers
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
DS=Dataset[,1:4] # preserve an original data set
Dataset.dat=Dataset[,1:4]
Dataset.dat=scale(Dataset.dat)
Dataset=cbind(Dataset.dat,Dataset[,5])
rm(Dataset.dat) #remove unuse variable
colnames(Dataset)[colnames(Dataset)==""] <- "Yn" #rename the true class column
Dataset=data.frame(Dataset)

#Split into train and test set 80/20
intrain=createDataPartition(Dataset$Yn,p=0.8,list = FALSE,times = 1)

TRAIN=Dataset[intrain,]
TRAIN.ori=DS[intrain,] # Unstandardized TRAIN set
set.seed(3)
#Shuffle the data set
shuff=sample(nrow(TRAIN))
TRAIN=TRAIN[shuff,]
TRAIN.ori=TRAIN.ori[shuff,]

TEST=Dataset[-intrain,]
TEST.ori=DS[-intrain,] # Unstandardized TRAIN set
set.seed(5)
#Shuffle the data set
shuff=sample(nrow(TEST))
TEST=TEST[shuff,]
TEST.ori=TEST.ori[shuff,]

#get the value of poly function for the set TRAIN and TEST
pol_TRAIN=NULL
for (i in 1:4000){
  pol_TRAIN[i]=polyx(data.matrix(TRAIN.ori[i,]))
}
plot(pol_TRAIN,TRAIN$Yn,main = 'Poly(TRAIN) vs True class')

pol_TEST=NULL
for (i in 1:1000){
  pol_TEST[i]=polyx(data.matrix(TEST.ori[i,]))
}
plot(pol_TEST,TEST$Yn,main='Poly(TEST) vs True class')

# Check the variation of dataset

################ A function to calculate the error of estimation
std_p = function(acc,n) {sqrt(acc*(1-acc)/n)*100}


############## A function to generate a nice confusion matrix
generate_cfm=function(predi,truec,cap,prt){
  
  cfm=table(truec, predi) # Frequency table
  nocc=rowSums(cfm) #Total cases of each class 
  pocp=cfm/nocc #Confusion matrix
  acc=(cfm[1,1]+cfm[2,2])/length(truec) # the global accuracy
  
  # Std for three percentages of correct prediction : global and 2 classes
  std=std_p(acc,length(truec))
  std_1=std_p(pocp[1,1],nocc[1])
  std_2=std_p(pocp[2,2],nocc[2])
  
  #Prepare for the Prediction table
  ac=c(acc,pocp[1,1],pocp[2,2])*100
  st=c(std,std_1,std_2)
  pred_table=data.frame(ac,st,ac-1.96*st,ac+1.96*st) # Upper and lower limit for the estimates
  rownames(pred_table) = c('Global accuracy','CL-1','CL1')
  colnames(pred_table) = c('Accuracy','Standard deviation','LL','UL')
  
  if (prt == TRUE) {
    print(cap)
    print(cfm)
    print(pocp)
    print(acc)
    print(pred_table,digits = 5 )
  }
  return(pred_table)
}

#################  Question 2 ###########

# A function to generate the report
generate_model_report= function(model,cap,prt){
  
  summary(model)
  S=sum(model$nSV) # total number of support vectors
  s=S/4000
  
  #Prediction for TRAIN and TEST sets
  PredTRAIN =predict(model,dat[,1:4])
  PredTEST =predict(model,TEST.dat[,1:4])
  
  # Check accuracy:
  pre_train=generate_cfm(PredTRAIN, TRAIN$Yn,'Confusion matrix for the set TRAIN',prt)
  pre_test=generate_cfm(PredTEST, TEST$Yn,'Confusion matrix for the set TEST',prt)
  if (prt==TRUE){
    print(paste('Number S of support vectors is ',S))
    print(paste('And the ratio s = ',s))
    
    prediction=bind_rows(pre_train,pre_test)
    write.xlsx(prediction,paste(cap,'.xlsx'))
    
    PredTRAIN=as.numeric(as.character(PredTRAIN))
    jpeg(paste(cap,'on TRAIN.jpg'),units = 'in',width=8, height=4, res=240)
    plot(pol_TRAIN,PredTRAIN,main='Poly(TRAIN) vs Prediction')
    abline(v=0)
    dev.off()
    
    PredTRAIN=as.numeric(as.character(PredTRAIN))
    jpeg(paste(cap,'on TEST.jpg'),units = 'in',width=8, height=4, res=240)
    plot(pol_TEST,PredTEST,main='Poly(TEST) vs Prediction')
    abline(v=0)
    dev.off()
  }
  return(c(s,pre_train[1,1],pre_test[1,1]))
}

# Convert the Yn to factor type
x=TRAIN[,1:4]
y=TRAIN$Yn
dat = data.frame(x=x, y=as.factor(y)) # must convert to factor to avoid the svm regression
TEST.dat=data.frame(x=TEST[,1:4], y=as.factor(TEST$Yn)) 
#Alternative way to use svm without specify the type is to conver Y to factor
#model_2 =  svm(Yn~., cost=5, kernel='linear',type='C-classification',scale=FALSE,data=TRAIN)
sta_tim=Sys.time()
# run svm
model_2 =  svm(y~., cost=5, kernel='linear',scale=FALSE,data=dat)
generate_model_report(model_2,'2. Arbitrary linear svm model',prt = TRUE)

#Build linear separator
#coef_m2=coef(model_2)
#pol_svm2=data.matrix(TRAIN[,1:4])%*%coef_m2[2:5]
#plot(pol_svm2,PredTRAIN,main='SVM(TRAIN) vs Prediction class')

end_tim=Sys.time()
print(paste('Computing time for question 2 is ',round(end_tim-sta_tim,2),' second'))


# Question 3 ####


set.seed(1)
tune.linear = tune(svm ,y~., data=dat , kernel ="linear",scale=FALSE,
                ranges =list (cost=c(0.001, 0.01, 0.1, 1, 10,100) ))
print(summary(tune.linear))
plot(tune.linear,main='Performance of linear svm')

sta_tim=Sys.time()
tune.linear2 = tune(svm ,y~., data=dat , kernel ="linear",scale=FALSE,
                   ranges =list (cost=c(0.0001, 0.0005, 0.001, 0.005, 0.01,0.05) ))
end_tim=Sys.time()
print(paste('Computing time for question 3 is ',round(end_tim-sta_tim,2),' second'))
print(summary(tune.linear2))
plot(tune.linear2,main='Performance of linear svm 2')
output=NULL
cost=c(0.0001, 0.0005, 0.001, 0.005, 0.01,0.05)
for (i in 1:6){
  model=svm(y~., cost=cost[i], kernel='linear',scale=FALSE,data=dat)
  val=generate_model_report(model,'2. Arbitrary linear svm model',prt = FALSE)
  output=cbind(output,val)
}
output=t(output)
colnames(output)=c('s','Performance on TRAIN','Performance on TEST')
print(output)
plot(cost,type='l',output[,1],ylab = "Ratio s")
plot(cost,type='l',output[,2],col='blue',ylim=c(0.72, 0.74),ylab = "Percentage of Performance")
lines(cost,output[,3],col='red')
legend(0.035,0.725,c("Performance on TRAIN","Performance on TEST"),lwd=c(2,2), col=c("blue","red"))


model_3.best=tune.linear$best.model
generate_model_report(model_3.best,'3.Best linear svm model',prt=TRUE)

model_3.best=svm(y~., cost=0.05, kernel='linear',scale=FALSE,data=dat)
generate_model_report(model_3.best,'Best linear svm model',prt = TRUE)

plot(model_3.best,dat,x.X1~x.X2,svSymbol = "x", dataSymbol = ".")
title(main = "Best linear")



# Question 4 ####

sta_tim=Sys.time()

model_4 =  svm(y~., cost=0.05, kernel='radial',gamma=1,scale=FALSE,data=dat)
generate_model_report(model_4,'4.Arbitrary radial svm',prt=TRUE)

end_tim=Sys.time()
print(paste('Computing time for question 4 is ',round(end_tim-sta_tim,2),' second'))

# A function to generate a heatmap #1
generate_heatmap = function(x,caption){
  x=cast(x,cost~gamma,value=colnames(x)[3])
  row.names(x)=x[,1]
  x=data.matrix(x[,2:6])
  heatmap.2(x,trace = "none", Colv=NA, Rowv=NA,colsep,
            rowsep,
            sepcolor="black",
            sepwidth=c(0.1,0.1),
            xlab = 'Gamma',ylab = 'Cost',main = caption )
}
  
# Question 5 ####




set.seed(1)
tune.radial= tune(svm ,y~., data=dat , kernel ="radial",scale=FALSE,
                ranges =list (cost=c(0.001 , 0.01 , 0.1, 1 ,100),
                              gamma=c(0.001 , 0.01 , 0.1, 1 ,100)))
print(summary(tune.radial))
write.xlsx(tune.radial$performances, "tune.radial.per.xlsx")

sta_tim=Sys.time()
set.seed(1)
tune.radial.2= tune(svm ,y~., data=dat , kernel ="radial",scale=FALSE,
                  ranges =list (cost=c(60 , 80 , 100, 120 ,140),
                                gamma=c(0.06 , 0.08 , 0.1, 0.12 ,0.14)))
end_tim=Sys.time()
print(paste('Computing time for question 5 is ',round(end_tim-sta_tim,2),' second'))

print(summary(tune.radial.2))
write.xlsx(tune.radial.2$performances, "tune.radial.2.perf.xlsx")


generate_heatmap(tune.radial.2$performances[,1:3],"The error for TRAIN set of difference radial svm models")
model_5.best = tune.radial.2$best.model
generate_model_report(model_5.best,'5.Best radial svm',prt=TRUE)

#par(mfrow=c(1,2))
#plot(model_5,dat,x.X1~x.X2)
#plot(model_5,dat,x.X2~x.X3)

opt.radial=NULL
cost=c(60 , 80 , 100, 120 ,140)
gamma=c(0.06 , 0.08 , 0.1, 0.12 ,0.14)
for (i in 1:5){
  for (j in 1:5){
    model=svm(y~., cost=cost[i], kernel='radial',,gamma=gamma[j],scale=FALSE,data=dat)
    val=generate_model_report(model,'5.test radial svm model',prt = FALSE)
    opt.radial=cbind(opt.radial,val)
  }
  
}
opt.radial=t(opt.radial)
cost.list=rep(cost,each=5)
gamma.list=rep(gamma,5)
opt.radial=cbind(cost.list,gamma.list,opt.radial)
colnames(opt.radial)=c("cost","gamma",'s','Performance on TRAIN','Performance on TEST')
opt.radial=data.frame(opt.radial)
#write.xlsx(opt.radial, "opt.radial.perf.xlsx")
generate_heatmap(opt.radial[,1:3],"The ratio s of difference radial svm models")

generate_heatmap(opt.radial[,c(1:2,4)],"The Performance for Train set of radial svm models")

delta=round(opt.radial[,4]-opt.radial[,5],3)
opt.radial.dat=cbind(opt.radial[,1:2],delta)
generate_heatmap(opt.radial.dat,"The Difference between performances for TRAIN and SET")

plot(model_5.best,dat,x.X1~x.X2,svSymbol = "x", dataSymbol = ".")
title(main = "Radial")

end_tim=Sys.time()
print(paste('Computing time for question 5 is ',round(end_tim-sta_tim,2),' second'))

# A function to generate a heatmap #2
generate_heatmap.2 = function(x,caption){
  x=cast(x,cost~coef0,value=colnames(x)[3])
  row.names(x)=x[,1]
  x=data.matrix(x[,2:6])
  heatmap.2(x,trace = "none", Colv=NA, Rowv=NA,colsep,
            rowsep,
            sepcolor="black",
            sepwidth=c(0.1,0.1),
            xlab = 'Coef0',ylab = 'Cost',main = caption )
}
# Question 6 ####




model_6 =  svm(y~., cost=1, kernel="polynomial",degree=4,gamma=1,coef0=1,scale=FALSE,data=dat)
generate_model_report(model_6,'6.Arbitrary polynomial svm',prt = TRUE)
#plot(model_6,TRAIN,X1~X2)
model_6.2 =  svm(y~., cost=0.08, kernel="polynomial",degree=4,gamma=1,coef0=100,scale=FALSE,data=dat)
generate_model_report(model_6.2,'6.Arbitrary polynomial svm 2',prt = TRUE)

model_6.3 =  svm(y~., cost=0.01, kernel="polynomial",degree=4,gamma=1,coef0=120,scale=FALSE,data=dat)
generate_model_report(model_6.3,'6.Arbitrary polynomial svm 3',prt = TRUE)

set.seed(1)
tune.poly = tune(svm ,y~., data=dat , kernel ="polynomial",degree=4,gamma=1,scale=FALSE,
                ranges =list (cost=c(0.01, 0.1, 1  ,10 ,100),
                              coef0=c(0.01, 0.1, 1 ,10 ,100)))
print(summary(tune.poly))
write.xlsx(tune.poly$performances, "tune.poly.perf.xlsx")

sta_tim=Sys.time()
set.seed(5)
tune.poly.2 = tune(svm ,y~., data=dat , kernel ="polynomial",degree=4,gamma=1,scale=FALSE,
                 ranges =list (cost=c(0.06, 0.08, 0.01  ,0.12 ,0.14),
                               coef0=c(60, 80, 100 ,120 ,140)))
end_tim=Sys.time()
print(paste('Computing time for question 6 is ',round(end_tim-sta_tim,2),' second'))

print(summary(tune.poly.2))
write.xlsx(tune.poly.2$performances, "tune.poly.2.perf.xlsx")

generate_heatmap.2(tune.poly.2$performances[,1:3],"The error for TRAIN set of difference polynomial models")

model_6.best=tune.poly.2$best.model
generate_model_report(model_6.best,'6.The best polynomial svm',prt = TRUE)

opt.poly=NULL
cost=c(0.06, 0.08, 0.01  ,0.12 ,0.14)
coef0=c(60, 80, 100 ,120 ,140)
for (i in 1:5){
  for (j in 1:5){
    model=svm(y~., cost=cost[i], kernel='polynomial',degree=4,gamma=1,coef0=coef0[j],scale=FALSE,data=dat)
    val=generate_model_report(model,'6.test polynomial svm model',prt = FALSE)
    opt.poly=cbind(opt.poly,val)
  }
  
}
opt.poly=t(opt.poly)
#

cost.list=rep(cost,each=5)
coef0.list=rep(coef0,5)
opt.poly=cbind(cost.list,coef0.list,opt.poly)
colnames(opt.poly)=c("cost","coef0",'s','Performance on TRAIN','Performance on TEST')
opt.poly=data.frame(opt.poly)
generate_heatmap.2(opt.poly[,1:3],"The ratio s of difference polynomial svm models")

generate_heatmap.2(opt.poly[,c(1:2,4)],"The Performance for Train set of Polynomial models")

delta=opt.poly[,4]-opt.poly[,5]
opt.poly.dat=cbind(opt.poly[,1:2],delta)
generate_heatmap.2(opt.poly.dat,"The Difference between performances for TRAIN and SET")

plot(model_6.best,dat,x.X1~x.X2,svSymbol = "x", dataSymbol = ".")
title(main = "Best Polynomial")

