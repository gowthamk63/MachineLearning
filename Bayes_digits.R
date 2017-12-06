library(e1071)
library(caret)


digits=read.csv("optdigits_raining.csv",header = FALSE)
test1=read.csv("optdigits_test.csv",header = FALSE)


##Calculating the attributes,means and standard deviation####
no_attributes=ncol(Digits)
no_observations=length(Digits$V65)
sds<-as.data.frame(apply(test[,1:ncol(Digits)], 2, function(x) sd(x, na.rm=FALSE)),header=FALSE)
Means<-as.data.frame(apply(test[,1:ncol(Digits)], 2, function(x) mean(x, na.rm=FALSE)),header=FALSE)



normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
digits=cbind(normalize(digits[1:64]),V65=digits$V65)
test1_norm=cbind(normalize(test1[1:64]),V65=test1$V65)




train=naiveBayes(as.factor(digits_norm$V65)~.,digits_norm,threshold=1,laplace = 3,type='raw')
tr=predict(train,test1_norm)
(sum(tr==test1_norm$V65)/length(test1_norm$V1))

-
confusionMatrix(tr,test1_norm$V65)

acc
t=NaiveBayes(as.factor(digits$V65)~.,digits)
#################Algorithms
bayes_acc=NULL
cv=NULL
i=1
m=1
l=1
while(i<100){
train=naiveBayes(as.factor(digits$V65)~.,digits,threshold=i,laplace = 1)
tr=predict(train,digits)
bayes_acc[m]=c(sum(tr==digits$V65)/length(digits$V1))
j=7
while(j<10){
  p=c(1:floor(j/10*length(digits$V1)))
  train2=naiveBayes(as.factor(digits[p,]$V65)~.,digits[p,],threshold=1)
  tr=predict(train2,digits[-p,])
  cv[l]=c(sum(tr==digits[-p,]$V65)/length(digits[-p,]$V1))
  j=j+1
  l=l+1
}
i=i+10
m=m+1
}

#####Ploting Acuracy graphs

plot(cv)
plot(bayes_acc)
cv_acc=cv*100
m=c(1:9)
cv_plot=NULL
cv_plot=as.data.frame(cbind(cv90=cv_acc[seq(1,36,4)],cv80=cv_acc[seq(2,36,4)],
                            cv70=cv_acc[seq(3,36,4)],cv60=cv_acc[seq(4,36,4)]))
plot.ts(cv_plot,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for cross validated linear kernel")


plot(1:length(l[1:64]),l)
mmetric(as.factor(test1$V65),tr,c("ACC"))
l=confusionMatrix(tr,test1$V65)


cv_plot=as.data.frame(cbind(cv90=cv[seq(1,40,4)],cv80=cv[seq(2,40,4)],
                                   cv70=cv[seq(3,40,4)],cv60=cv[seq(4,40,4)]))
plot.ts(cv_plot,xlab="laplace values",ylab="Accuracy",type = 'b')
