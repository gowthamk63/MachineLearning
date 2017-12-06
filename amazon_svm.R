##Libraries Required 
require( 'kernlab' )
library(tm)
library(SnowballC)

##Read csv files
train=read.csv("amazon_baby_train.csv")
test=read.csv("amazon_baby_test.csv")

#Removing numbers, stop words and white spaces
preprocess<-function(review_corpus){
  review_corpus = tm_map(review_corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus = tm_map(review_corpus, stripWhitespace)
}

#Converting dataframe to corpus
train_review_corpus = Corpus(VectorSource(train$review))
test_review_corpus = Corpus(VectorSource(test$review))

train_review_corpus=preprocess(train_review_corpus)
test_review_corpus=preprocess(test_review_corpus)

#Transforming to document term matrix by performing TF-IDF
train_review_dtm <- DocumentTermMatrix(train_review_corpus, control = list(weighting = weightTfIdf, minWordLength=2, minDocFreq=5))
test_review_dtm <- DocumentTermMatrix(test_review_corpus, control = list(weighting = weightTfIdf, minWordLength=2, minDocFreq=5))

#Setting the sparsity ratio to 0.95
train_review_dtm = removeSparseTerms(train_review_dtm, 0.95)

##Conversion of the data into matrix
train_review_dtm<-as.matrix(train_review_dtm)
test_review_dtm<-as.matrix(test_review_dtm)

##conversion into data frame 
test_review_dtm<- as.data.frame(test_review_dtm[,intersect(colnames(train_review_dtm),colnames(test_review_dtm))] )
train_review_dtm<- as.data.frame(train_review_dtm )

##combining the rating column to the training and testing dataframes
train_df=cbind(train_review_dtm,rating=train$rating)
test_df=cbind(test_review_dtm,rating=test$rating)


###SVM Algorithms####
cv_acc=NULL
cv_acc2=NULL
svm_gauss_acc=NULL
svm_acc=NULL
i=1
m=1
while (i<=100){
  ####SVM Algorithms #####
  svm=ksvm( train_df$rating~., data=train_df, type='C-svc', kernel='vanilladot',C=i )
  svm.prediction=predict( svm, test_df )
  svm_acc[m]=c(sum(test_df$rating==svm.prediction)/length(test_df$rating))
  
  ####SVM Algorithms  gaussian #####
  svm_gauss=ksvm( train_df$rating~., data=train_df, type='C-svc', kernel='rbfdot',C=i )
  svm_gauss.prediction=predict( svm_gauss, test_df )
  svm_gauss_acc[m]=c(sum(test_df$rating==svm_gauss.prediction)/length(test_df$rating))
  
  ####SVM algorithms for cross validation#######
  p=c(1:floor(7/10*length(train_df$rating)))
  cv_svm=ksvm( train_df[p,]$rating ~ ., data=train_df[p,], type='C-svc', kernel='vanilladot',C=i, scale=c())
  cv_svm.prediction=predict( cv_svm, train_df[-p,])
  cv_acc1[j]=c(100*sum( cv_svm.prediction == train_df[-p,]$rating)/length(train_df[-p,]$rating) )
  q=c(1:floor(8/10*length(train_df$rating)) )
  cv_svm=ksvm( train_df[q,]$rating ~ ., data=train_df[q,], type='C-svc', kernel='vanilladot',C=i, scale=c())
  cv_svm.prediction=predict( cv_svm, train_df[-q,])
  cv_acc2[m]=c(100*sum( cv_svm.prediction == train_df[-q,]$rating)/length(train_df[-q,]$rating) )
  i=i+30 
  m=m+1
} 
   


###Plotting graphs
m=seq(1,100,30)
plot(m,svm_acc,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for linear kernel")
plot(m,svm_gauss_acc,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for gaussian kernel")
plot(m,cv_gauss_plot,xlab="vaues of c",ylab="Accuracy",type = 'b')
title(main="SVM accuracy for cross validated gaussian kernel")

##maximum accuracy for differnt methods#####
cat("maximum accuracy for linear=",max(svm_acc))
cat("maximum accuracy for gaussian=",max(svm_gauss_acc))
cat("maximum accuracy for cross validation gaussian=",max(cv_gauss_acc))

