### Packages needed for this session 
require( 'kernlab' )
library(caret)

########Read the csv files#######
digits=read.csv("optdigits_raining.csv",header=FALSE)
test1=read.csv("optdigits_test.csv",header=FALSE)

##Calculating the attributes,means and standard deviation####
no_attributes=ncol(digits)
no_observations=length(digits$V65)
sds<-as.data.frame(apply(test1[,1:ncol(digits)], 2, function(x) sd(x, na.rm=FALSE)),header=FALSE)
Means<-as.data.frame(apply(test1[,1:ncol(digits)], 2, function(x) mean(x, na.rm=FALSE)),header=FALSE)

###SVM Algorithms####
cv_acc=NULL
svm_acc=NULL
i=1
j=1
m=1
while (i<=360){ 
  ####SVM Algorithms #####
  svm=ksvm( digits$V65~., data=digits, type='C-svc', kernel='vanilladot',C=i )
  svm.prediction=predict( svm, test1 )
  svm_acc[m]=c(sum(test1$V65==svm.prediction)/length(test1$V65))
  ####SVM algorithms for cross validation#######
  l=9
  while(l>5){
    p=c(1:floor(l/10*length(digits$V1)))
    cv_svm=ksvm( digits[p,]$V65 ~ ., data=digits[p,], type='C-svc', kernel='vanilladot',C=i, scale=c() )
    cv_svm.prediction=predict( cv_svm, digits[-p,] )
    cv_acc[j]=c(100*sum( cv_svm.prediction == digits[-p,]$V65)/length(digits[-p,]$V65))
    l=l-1
    j=j+1
  }
  i=i+40
  m=m+1
}

###Confusion matrix
c=confusionMatrix(svm.prediction,test1$V65)


###SVM Algorithms####
cv_gauss_acc=NULL
svm_gauss_acc=NULL
i=1
j=1
m=1
while (i<=360){ 
  ####SVM Algorithms  gaussian model #####
  svm_gauss=ksvm( digits$V65~., data=digits, type='C-svc', kernel='rbfdot',C=i )
  svm_gauss.prediction=predict( svm_gauss, test1 )
  svm_gauss_acc[m]=c(sum(test1$V65==svm_gauss.prediction)/length(test1$V65))
  ####SVM algorithms for cross validation#######
  l=9
  while(l>5){
    p=c(1:floor(l/10*length(digits$V1)))
    cv_svm_gauss=ksvm( digits[p,]$V65 ~ ., data=digits[p,], type='C-svc', kernel='rbfdot',C=i, scale=c() )
    cv_svm_gauss.prediction=predict( cv_svm_gauss, digits[-p,] )
    cv_gauss_acc[j]=c(100*sum( cv_svm_gauss.prediction == digits[-p,]$V65)/length(digits[-p,]$V65))
    l=l-1
    j=j+1
  }
  i=i+40
  m=m+1
}

###Plotting graphs
m=seq(1,360,40)
n=seq(1,360,40)
cv_plot=NULL
cv__gauss_plot=NULL
cv_plot=as.data.frame(cbind(cv90=cv_acc[seq(1,36,4)],cv80=cv_acc[seq(2,36,4)],
                            cv70=cv_acc[seq(3,36,4)],cv60=cv_acc[seq(4,36,4)]))
cv_gauss_plot=as.data.frame(cbind(cv90=cv_gauss_acc[seq(1,36,4)],cv80=cv_gauss_acc[seq(2,36,4)],
                                  cv70=cv_gauss_acc[seq(3,36,4)],cv60=cv_gauss_acc[seq(4,36,4)]))
plot(m,svm_acc,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for linear kernel")
plot(n,svm_gauss_acc,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for gaussian kernel")
plot.ts(cv_plot,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for cross validated linear kernel")
plot.ts(cv_gauss_plot,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for cross validated gaussian kernel")

##maximum accuracy for differnt methods#####
cat("maximum accuracy for linear=",max(svm_acc))
cat("maximum accuracy for gaussian=",max(svm_gauss_acc))
cat("maximum accuracy for crossvalidation linear=",max(cv_acc))
cat("maximum accuracy for cross validation gaussian=",max(cv_gauss_acc))

