#Titanic Dataset with 	1310 obs. of  14 variables:
rm(list=ls())                             
ls()
data=read.csv("titanic_data.csv",header=TRUE,stringsAsFactors = T) #Loading data
nrow(data)
str(data)
sum(is.na(data))                          #total 1459 NA values
#I will use CARET package for preprocessing of data:
library(caret)
preprocvalues=preProcess(data,method=c("medianImpute","center","scale")) #taking median for all NA with respective variables & adjusting scale
library(RANN)
data_pro=predict(preprocvalues,data)
sum(is.na(data_pro))                          #total 0 NA values
dv=dummyVars("~.",data_pro,fullRank = T)      # creating dummy variable to handle factors
data_tran=data.frame(predict(dv,data_pro))
data_tran$survived=as.factor(data_tran$survived)  # converting response variable in factor
set.seed(5)
index <- createDataPartition(data_tran$survived, p=0.75, list=FALSE) #data partition
train <- data_tran[ index,]       #Traning data=75%
test<- data_tran[-index,]         #Test data=25%
#############################################Decision Tree#############################
set.seed(3)
library(rpart)
m=rpart(survived~.,data=train,method="class",control=rpart.control(minsplit=20,
                             minbucket=7,maxdepth=10,usesurrogate = 2,xval=10))#pre-proned method
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(m)
printcp(m)                      
bestcp=m$cptable[which.min(m$cptable[,"xerror"]),"CP"]
bestcp                             #Evaluting best cp
pruned=prune(m,cp=bestcp) 
fancyRpartPlot(pruned)
t=table(train$survived,predict(pruned,type="class"))
prop.table(table(train$survived,predict(pruned,type="class")))
rownames(t)=paste("Actual",rownames(t),sep=":")
colnames(t)=paste("predicted",colnames(t),sep=":")
t
prop.table(t)
accuracy=sum(diag(t))/sum(t)
accuracy                       ###Accuracy on traning data=0.9287894
t=predict(m,test,type="class")
s=prop.table(table(t,test$survived))
s
accuracy=sum(diag(s))/sum(s) 
accuracy                       ### Accuracy on test data=0.8899083
#####################      ROC         #######################
for_auc=predict(pruned,test,type="prob")
library(pROC)
a=auc(test$survived,for_auc[,2])
a                                      #Area under the curve: 0.8977
#Ex:90-100,Good:80-90,fair:70-80,poor:60-70,Fail:50-60
plot(roc(test$survived,for_auc[,2]),main="Decsion Tree")
gini_coeff=2*a-1
gini_coeff                          # Gini Coeff=0.7954851
######################################## Random Forest ############################
library(randomForest)
set.seed(7) 
rf=randomForest(survived~.,train,ntree=60)  ## wait for few seconds
importance(rf)
varImpPlot(rf)
q=predict(rf,test,type="prob")
library(pROC)
x=auc(test$survived,q[,2])
x                               ##Area under the curve: 0.95
plot(roc(test$survived,q[,2]))
gini_coeff=2*x-1
gini_coeff                          # # Gini Coeff=0.90
###################################################################################

