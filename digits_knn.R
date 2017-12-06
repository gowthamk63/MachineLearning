#######Libraries Required#######
library(class)
library(kknn)
library(caret)


########Read the csv files#######
digits=read.csv("optdigits_raining.csv",header=FALSE)
test1=read.csv("optdigits_test.csv",header=FALSE)

##Calculating the attributes,means and standard deviation####
no_attributes=ncol(digits)
no_observations=length(digits$V65)
sds<-as.data.frame(apply(test1[,1:ncol(digits)], 2, function(x) sd(x, na.rm=FALSE)),header=FALSE)
Means<-as.data.frame(apply(test1[,1:ncol(digits)], 2, function(x) mean(x, na.rm=FALSE)),header=FALSE)

######Normalization#######
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
digits_norm=cbind(normalize(digits[1:64]),digits$V65)
test1_norm=cbind(normalize(test1[1:64]),test1$V65)

###Knn Algorithms####
cv_ecld_acc=NULL
cv_weight_acc=NULL
knn_ecld_acc=NULL
knn_weight_acc=NULL
i=1
j=1
while (i<20){ 
  ####Knn Algorithms with euclidiean distance and weighted differences#####
  knn_ecld=knn(train = digits,test=test1,cl=digits$V65,k=i,l=0,prob = FALSE)
  knn_weight=kknn(train = digits,test=test1,formula = digits$V65~.,k=i)
  knn_weight=fitted(knn_weight)
  knn_ecld_acc[i]=c(sum(test1$V65==knn_ecld)/length(knn_ecld))
  knn_weight_acc[i]=c(sum(test1$V65==knn_weight)/length(knn_weight))
  ####Knn algorithms for cross validation#######
  l=9
  while(l>5){
    p=c(1:floor(l/10*length(digits$V1)))
    cv_ecld=knn(train = digits[p,],test=digits[-p,],cl=digits[p,]$V65,k=i,l=0,prob = FALSE)
    cv_weight=kknn(train = digits[p,],test=digits[-p,],formula = digits[p,]$V65~.,k=i)
    cv_weight=fitted(cv_weight)
    cv_ecld_acc[j]=c(sum(digits[-p,]$V65==cv_ecld)/length(cv_ecld))
    cv_weight_acc[j]=c(sum(digits[-p,]$V65==cv_weight)/length(cv_weight))
    l=l-1
    j=j+1
  }
  i=i+1
}


####Knn Algorithms with euclidiean distance and weighted differences for the normalized inputs#####
i=1
knn_ecld_acc_norm=NULL
knn_weight_acc_norm=NULL
while (i<20) {
  knn_ecld_norm=knn(train = digits_norm,test=test1_norm,cl=digits_norm$`digits$V65`,k=i,l=0,prob = FALSE)
  knn_weight_norm=kknn(train = digits_norm,test=test1_norm,formula = digits_norm$`digits$V65`~.,k=i)
  knn_weight_norm=fitted(knn_weight_norm)
  knn_ecld_acc_norm[i]=c(sum(test1_norm$`test1$V65`==knn_ecld_norm)/length(knn_ecld_norm))
  knn_weight_acc_norm[i]=c(sum(test1_norm$`test1$V65`==knn_weight_norm)/length(knn_weight_norm))
  
  i=i+1
}


cv_ecld_acc=cv_ecld_acc*100
cv_weight_acc=cv_weight_acc*100
knn_ecld_acc=knn_ecld_acc*100
knn_weight_acc=knn_weight_acc*100
knn_ecld_acc_norm=knn_ecld_acc_norm*100
knn_weight_acc_norm=knn_weight_acc_norm*100


####plotting the accuracy for the different k values#####
plot(1:length(knn_ecld_acc),knn_ecld_acc,xlab="vaues of k",ylab="Accuracy",type = 'b')
title(main="Accuracy graph for Euclidean Distance")
plot(1:length(knn_weight_acc),knn_weight_acc,xlab="vaues of k",ylab="Accuracy",type = 'b')
title(main="Accuracy graph for Weighted Distance")

####plotting the accuracy for the different Cross validation values#####
cv_ecld_plot=NULL
cv_weight_plot=NULL
cv_ecld_plot=as.data.frame(cbind(cv90=cv_ecld_acc[1:7],cv80=cv_ecld_acc[8:14],
                                 cv70=cv_ecld_acc[15:21],cv60=cv_ecld_acc[22:28]))
cv_weight_plot=as.data.frame(cbind(cv90=cv_weight_acc[1:7],cv80=cv_weight_acc[8:14],
                                   cv70=cv_weight_acc[15:21],cv60=cv_weight_acc[22:28]))
plot.ts(cv_ecld_plot,xlab="vaues of k",ylab="Accuracy",type = 'b')
title(" knn for Crossvalidation with differing k values")
plot.ts(cv_weight_plot,xlab="vaues of k",ylab="Accuracy",type = 'b')
title("Weighted knn for Crossvalidation with differing k values")


###plotting the accuray for the normalized data
plot(1:length(knn_ecld_acc_norm),knn_ecld_acc_norm,xlab="vaues of k",ylab="Accuracy",type = 'b')
title(main="KNN for normalized data")
plot(1:length(knn_weight_acc_norm),knn_weight_acc_norm,xlab="vaues of k",ylab="Accuracy",type = 'b')
title(main="Weighted KNN for normalized data")

##maximum accuracy for differnt methods#####
cat("maximum accuracy for Ecludiean distacne=",max(knn_ecld_acc))
cat("maximum accuracy for weighted knn=",max(knn_weight_acc))
cat("maximum accuracy for crossvalidation Ecludiean distacne=",max(cv_ecld_acc))
cat("maximum accuracy for cross validation weighted knn=",max(cv_weight_acc))
cat("maximum accuracy for Ecludiean distacne Normalized data=",max(knn_ecld_acc_norm))
cat("maximum accuracy for weighted knn normalized data=",max(knn_weight_acc_norm))

confusionMatrix(knn_ecld)
