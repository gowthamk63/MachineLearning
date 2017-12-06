#####Libraries####
library(rpart)
library(ada)
library(adabag)

##Read Data from the csv files######
Digits<-read.csv("optdigits_raining.csv",header=FALSE)
test<-read.csv("optdigits_test.csv",header=FALSE)
Digits$V65=factor(Digits$V65)
test$V65=factor(test$V65)


##Calculating the attributes,means and standard deviation####
no_attributes=ncol(Digits)
no_observations=length(Digits$V65)
sds<-as.data.frame(apply(test[,1:ncol(Digits)], 2, function(x) sd(x, na.rm=FALSE)),header=FALSE)
Means<-as.data.frame(apply(test[,1:ncol(Digits)], 2, function(x) mean(x, na.rm=FALSE)),header=FALSE)


##rpart with pruning  based on cp values##
rpart_control<-rpart.control(minsplit = 5,minbucket = 1,cp=0)

##rpart algorithms for decision tree classification
rpart_nopruning<-rpart(Digits$V65~.,Digits,method = "class")
rpart_pruning<-rpart(Digits$V65~.,Digits,method = "class",control=rpart_control)

## Predicting the output####
testout3<-predict(rpart_nopruning,test,type="class")    #prediction for rpart DT without pruning      
testout4<-predict(rpart_pruning,test,type="class")      #prediction for rpart DT with pruning

### Calculating Accuracy for the Decision Trees###
rpart_nop_accuracy=sum(test$V65==testout3)/length(testout3)
rpart_p_accuracy=sum(test$V65==testout4)/length(testout4)

## Training the dataset using boosting algorithm
i=15
j=1
accuracy=NULL
while(i<300){
boosting_trian=boosting(V65~.,Digits,control = rpart_control,mfinal = i,coeflearn = "Breiman")
pred=predict.boosting(boosting_trian,test)
accuracy[j]=sum(diag(pred$confusion))/length(test[[1]])
i=i+30
j=j+1
}
c=seq(from=15,to=300,by=15)
plot(c,accuracy,type = 'b',xlab = "No. Iterations")
title("Accuracy Vs Iterations")
max(accuracy)


## rpart Alogorithm for the Decision Tree and boosting using the Adapackage for Crossvalidation#### 
j=9
l=1
accuracy=NULL
p=NULL
while(j>5){
p=c(1:floor(j/10*length(Digits$V1)))
boosting_trian=boosting(V65~.,Digits[p,],control = rpart_control,mfinal = 100,coeflearn = "Breiman")
c=predict.boosting(boosting_trian,Digits[-p,])
accuracy_cv[l]=sum(diag(c$confusion))/length(Digits[-p,][[1]])
j=j-1
l=l+1
}

##Maximum accuracy##
max(accuracy_cv)

##plotting decision tree##
k=plot(l$trees[[2]])

##error evaluation plots
e_test=errorevol(pred,test)
plot.errorevol(e_test)




