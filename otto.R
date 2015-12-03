# redaing in data
train<-read.csv("train.csv",header=TRUE)
test<-read.csv("test.csv",header=TRUE)

str(train)
#removing id feature
train <- train[,-1]
test = test[,-1]

library(randomForest)
library(ggplot2)
library(gbm)
install.packages("xgboost")
library(xgboost)

# simple RF model
M1=randomForest(target~.,data=train, ntree=500,importance=TRUE)

pred1=predict(M1, test,type="prob")
summary(pred1)
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- pred1
head(submission)
write.csv(submission, "Prediction.csv", row.names=TRUE)
#multiclass loss=0.55897
imp1<-importance(M1,type=1)
imp1<-data.frame(Feature=row.names(imp1),Importance=imp1[,1])
imp1
ggplot(imp1,aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() +
  ylab("Importance") +
  xlab("") + 
  ggtitle("M1 Feature Importance")

# model removing least important features
imp1strip<-subset(imp1,imp1$Importance<25)
imp1strip$Feature
M2=randomForest(target~.-feat_2+feat_3+feat_4+feat_6+feat_7+feat_27+feat_28+feat_35+feat_46+feat_49+feat_51+feat_54+feat_61+feat_80+feat_82+feat_83,data=train, ntree=500,importance=TRUE)
pred2=predict(M2, test,type="prob")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- pred2
head(submission)
write.csv(submission, "Prediction.csv", row.names=TRUE)
#multiclass loss=0.55975

# RF model using tuneRF function
mtry <- tuneRF(train[,1:93], train[,94])
mtry

M3=randomForest(target~.,data=train, ntree=500,mtry=18,importance=TRUE)
pred3=predict(M3, test,type="prob")
summary(pred3)
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- pred3
head(submission)
write.csv(submission, "Prediction.csv", row.names=TRUE)
#multiclass loss=0.53941
imp3<-importance(M3,type=1)
imp3<-data.frame(Feature=row.names(imp3),Importance=imp3[,1])
imp3
ggplot(imp3,aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill=blue) +
  coord_flip() +
  ylab("Importance") +
  xlab("") + 
  ggtitle("M3 Feature Importance")


#boosted trees: xgboost
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 50
M4 = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(M4,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

#0.50995

# Run Cross Valication
cv.nround = 250
best.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)
best.cv

# Train the model
nround = 250
M5 = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(M5,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
0.47512


pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)


