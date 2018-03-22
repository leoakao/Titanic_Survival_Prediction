directory = "C:/Projects/titanic"
setwd(directory)

gendersub <- read.csv("gender_submission.csv", stringsAsFactors = FALSE) # Sample Submission
dataset     <- read.csv("train.csv", stringsAsFactors = FALSE)
datapred<- read.csv("test.csv", stringsAsFactors = FALSE)

sapply(dataset,class)

library(caTools)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tidyverse)
library(dummies)
library(e1071)
library(rpart)
library(class)
library(randomForest)
library(xgboost)
library(h2o)
library(caret)
library(e1071)
library(ElemStatLearn)
library(pROC)
library(ROCR)

# Creating Datasets --------------------------------------------------------------------------------

# Find Columns with NA or Blanks
# colnames(dataset)[colSums(is.na(dataset)) > 0]
# colnames(dataset)[colSums(dataset == "") > 0]
apply(dataset, 2, function(x) any(is.na(x)))
apply(dataset, 2, function(x) any(x == ""))

apply(datapred, 2, function(x) any(is.na(x)))
apply(datapred, 2, function(x) any(x == ""))

# Missing Values (Age, Fare, Embarked)
# Impute Mean Age by Passenger Class, Sex for Missing Values
dataset <- dataset %>% 
  group_by(Pclass, Sex) %>% 
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age)) %>%
  ungroup()

datapred <- datapred %>% 
  group_by(Pclass, Sex) %>% 
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age)) %>%
  ungroup()
 
# Impute Mean Fare by Passenger Class for Missing Values
datapred %>% 
  group_by(Pclass, Embarked) %>% 
  count()

datapred <- datapred %>% 
  group_by(Pclass, Embarked) %>% 
  mutate(Fare = ifelse(is.na(Fare), mean(Fare, na.rm = TRUE), Fare)) %>%
  ungroup()

# Impute Most Common Emarked Location for Missing Values
dataset %>% group_by(Embarked) %>% count()
dataset %>% group_by(Embarked, Pclass) %>% count()
dataset$Embarked <- ifelse(dataset$Embarked=="","S",dataset$Embarked)

# Converting data type
dataset$Survived <- as.factor(dataset$Survived)
datapred$Pclass <- as.factor(datapred$Pclass)

dataset <- as.data.frame(dataset)
dataset2 <- dataset
datapred <- as.data.frame(datapred)
datapred <- datapred

# Removing Unused Features
dataset2$Name <- NULL
dataset2$Ticket <- NULL
dataset2$Cabin <- NULL

datapred$Name <- NULL
datapred$Ticket <- NULL
datapred$Cabin <- NULL

# Creating Dummy Variables
dataset2 <- dummy.data.frame(dataset2, names = c('Sex','Embarked'),sep = "_")
dataset2$Sex_female <- NULL
dataset2$Embarked_S <- NULL
dataset2$Pclass <- as.numeric(dataset2$Pclass)

datapred <- dummy.data.frame(datapred, names = c('Sex','Embarked'),sep = "_")
datapred$Sex_female <- NULL
datapred$Embarked_S <- NULL
datapred$Pclass <- as.numeric(datapred$Pclass)

# Feature Engineering
# Create Family Size Variable
dataset2$famsize <- dataset2$SibSp + dataset2$Parch + 1
datapred$famsize <- datapred$SibSp + datapred$Parch + 1

dataset2$SibSp <- NULL
dataset2$Parch <- NULL

datapred$SibSp <- NULL
datapred$Parch <- NULL

# Splitting Dataset
set.seed(123)
split = sample.split(dataset2$Survived, SplitRatio = 0.8)
train = subset(dataset2, split == TRUE)
test = subset(dataset2, split == FALSE)

# Feature Scaling
trainfs <- train
testfs <- test
trainfs[c(-1,-2)] <- scale(trainfs[c(-1,-2)])
testfs[c(-1,-2)] <- scale(testfs[c(-1,-2)])

# Dataset 3 is scaled data, Dataset 1 is unmodified
dataset3 <- dataset2
dataset3$Pclass <- as.numeric(dataset3$Pclass)
dataset3[c(-1,-2)] <- scale(dataset3[c(-1,-2)])

train$Pclass <- as.factor(train$Pclass)
test$Pclass <- as.factor(test$Pclass)
dataset2$Pclass <- as.factor(dataset2$Pclass)
datapred$Pclass <- as.factor(datapred$Pclass)

# For Python TPOT 
write.csv(dataset2,"traincleaned.csv", row.names = FALSE) 
write.csv(datapred,"testcleaned.csv", row.names = FALSE) 

# Data Exploration---------------------------------------------------------------
sapply(train, class)

dataset %>%
  group_by(Survived, Sex) %>%
  count()

# temp <- train %>%
#   group_by(Survived, Pclass) %>%
#   count() %>%
#   mutate(SurvivedClass = paste("Survived: ",Survived,", PClass: ",Pclass) )  %>%
#   ungroup() %>%
#   select(SurvivedClass, n)

ggplot(data= dataset2) + 
  geom_bar(mapping = aes(x = Survived, fill = Pclass)) + 
  coord_flip()

# Histograms / Distributions
ggplot(data = dataset2) +
  geom_histogram(mapping = aes(x = Age))

ggplot(data = dataset2) +
  geom_histogram(mapping = aes(x = Fare))

ggplot(data = dataset) +
  geom_bar(mapping = aes(x = SibSp))

ggplot(data = dataset) +
  geom_bar(mapping = aes(x = Parch))

ggplot(data = dataset2) +
  geom_bar(mapping = aes(x = famsize))

ggplot(data = dataset2) +
  geom_bar(mapping = aes(x = Pclass))

ggplot(data = dataset) +
  geom_bar(mapping = aes(x = Embarked))

# Survival Comparisons
ggplot(data = dataset) + 
  geom_bar(mapping = aes(x = SibSp, fill = Survived), position = "fill")

ggplot(data = dataset2) + 
  geom_histogram(aes(x = Age, fill = Survived))

ggplot(data = dataset2) + 
  geom_bar(mapping = aes(x = Pclass, fill = Survived)) + 
  coord_flip()

ggplot(data = dataset2) + 
  geom_point(mapping = aes(x = Age, y = Fare, color = Survived))

bar <- ggplot(data = train) + 
  geom_bar(
    mapping = aes(x = Pclass, fill = Survived), 
    show.legend = FALSE,
    width = 1
  ) + 
  theme(aspect.ratio = 1) +
  labs(x = NULL, y = NULL)
bar + coord_flip()
bar + coord_polar()

bp <- ggplot(data = dataset) + 
  geom_bar(mapping = aes(x=Pclass, fill=Survived), position = "dodge")
bp + facet_grid(Embarked ~ .) + coord_flip()

# Predicting -----------------------------------------------------------------------------------------
# Regression -----------------------------------------------------------------------------------------
# Unrestricted Logit Model
lr1 <- glm(formula = Survived ~ .,
                 family = binomial,
                 data = train[,2:9])

prob_pred_lr1 <- predict(lr1, type = 'response', newdata = test[,3:9])
y_pred_lr1 <- ifelse(prob_pred_lr1 > 0.5, 1, 0)

summary(lr1)

# Making the Confusion Matrix
cm_lr1 <- table(test[,2], y_pred_lr1)

acc_lr1 <- (cm_lr1[2,2] + cm_lr1[1,1])/count(test)

# Restricted Logit Model
lr2 <- glm(formula = Survived ~ Pclass + Sex_male + Age + famsize + Embarked_Q + Embarked_C,
         family = binomial,
         data = train[,2:9])

prob_pred_lr2 <- predict(lr2, type = 'response', newdata = test[,3:9])
y_pred_lr2 <- ifelse(prob_pred_lr2 > 0.5, 1, 0)

summary(lr2)

# Making the Confusion Matrix
cm_lr2 <- table(test[,2], y_pred_lr2)

acc_lr2 <- (cm_lr2[2,2] + cm_lr2[1,1])/count(test)

# K-NN ------------------------------------------------------------------------------------
# Feature scaling done since algorithm is based on Eucleadian distance
y_pred_knn <- knn(train = trainfs[, c(-1,-2)],
             test = testfs[,c(-1,-2)],
             cl = trainfs[, 2],
             k = 20)

# Making the Confusion Matrix
cm_knn = table(testfs[, 2], y_pred_knn)

acc_knn = (cm_knn[2,2] + cm_knn[1,1])/count(testfs)

# SVM -------------------------------------------------------------------------------------
# Linear SVM
# Feature scaling included in E1071 package
psvm = svm(formula =  Survived ~ .,
                 data = train[-1],
                 type = 'C-classification',
                 kernel = 'linear')

y_pred_svm = predict(psvm, newdata = test[c(-1,-2)])

# Making the Confusion Matrix
cm_svm = table(test[,2], y_pred_svm)

acc_svm = (cm_svm[2,2] + cm_svm[1,1])/count(test)

# Kernal SVM
# Feature scaling included in E1071 package
ksvm <- svm(formula = Survived ~ .,
                 data = train[-1],
                 type = 'C-classification',
                 kernel = 'radial')

y_pred_ksvm <- predict(ksvm, newdata = test[c(-1,-2)])

# Making the Confusion Matrix
cm_ksvm <- table(test[, 2], y_pred_ksvm)

acc_ksvm <- (cm_ksvm[2,2] + cm_ksvm[1,1])/count(test)

# Naive Bayes --------------------------------------------------------------------------------------
# Naive Bayes 1
# No feature scaling used
nb1 <- naiveBayes(x = train[c(-1,-2)],
                 y = train$Survived)

y_pred_nb1 <- predict(nb1, newdata = test[c(-1,-2)])

# Making the Confusion Matrix
cm_nb1 <- table(test[, 2], y_pred_nb1)

acc_nb1 <- (cm_nb1[2,2] + cm_nb1[1,1])/count(test)

# Naive Bayes 2
# Feature scaling lowers accuracy
nb2 <- naiveBayes(x = trainfs[c(-1,-2)],
                        y = trainfs$Survived)

y_pred_nb2 <- predict(nb2, newdata = testfs[c(-1,-2)])

# Making the Confusion Matrix
cm_nb2 <- table(testfs[, 2], y_pred_nb2)

acc_nb2 <- (cm_nb2[2,2] + cm_nb2[1,1])/count(testfs)

# Trees ----------------------------------------------------------------------------------------------
# Decision Tree 
# Does not use Euclidean Distance so feature scaling optional
dt <- rpart(formula = Survived ~ .,
                   data = train[-1])

y_pred_dt <- predict(dt, newdata = test[c(-1,-2)], type = 'class')

# Making the Confusion Matrix
cm_dt <- table(test[,2], y_pred_dt)

acc_dt <- (cm_dt[2,2] + cm_dt[1,1])/count(test)

# Random Forest 1
set.seed(123)
rf1 <- randomForest(x = train[c(-1,-2)],
                          y = train$Survived,
                          ntree = 100)

# Predicting the Test set results
y_pred_rf1 <- predict(rf1, newdata = test[c(-1,-2)])

# Making the Confusion Matrix
cm_rf1 <- table(test[, 2], y_pred_rf1)

acc_rf1 <- (cm_rf1[2,2] + cm_rf1[1,1])/count(test)

# Random Forest 2
set.seed(123)
rf2 <- randomForest(x = train[c(-1,-2)],
                    y = train$Survived,
                    ntree = 500)

# Predicting the Test set results
y_pred_rf2 <- predict(rf2, newdata = test[c(-1,-2)])

# Making the Confusion Matrix
cm_rf2 <- table(test[, 2], y_pred_rf2)

acc_rf2 <- (cm_rf2[2,2] + cm_rf2[1,1])/count(test)

# xgboost
train_2 <- train
test_2 <- test

str(train_2)
str(test_2)

train_2$Survived <- as.integer(train_2$Survived)-1
train_2$Pclass <- as.integer(train_2$Pclass)
test_2$Pclass <- as.integer(test_2$Pclass)

xgb = xgboost(data = as.matrix(train_2[c(-1,-2)]), label = train_2$Survived, nrounds = 250, objective = "binary:logistic")

# Accuracy Comparison# Predicting the Test set results
y_pred_xgb = predict(xgb, newdata = as.matrix(test_2[c(-1,-2)]))
y_pred_xgb = ifelse(y_pred_xgb >= 0.5,1,0)

# Making the Confusion Matrix
cm_xgb <- table(test_2[, 2], y_pred_xgb)

acc_xgb <- (cm_xgb[2,2] + cm_xgb[1,1])/count(test_2)

# Neural Network ---------------------------------------------------------------------------------------
# ANN 1
h2o.init(nthreads = -1)
annh1 = h2o.deeplearning(y = 'Survived',
                        training_frame = as.h2o(train_2[-1]),
                        activation = 'Rectifier',
                        hidden = c(5,5), # (number of hidden layers, number of nodes in hidden layer [(1+9)/2]) 
                        epochs = 100,
                        train_samples_per_iteration = -2)

# Predicting the Test set results
prob_pred_ann1 = h2o.predict(annh1, newdata = as.h2o(test_2[c(-1,-2)]))
y_pred_ann1 = (prob_pred_ann1 > 0.5)
y_pred_ann1 = as.vector(y_pred_ann1)

# Making the Confusion Matrix
cm_ann1 = table(test_2[,2], y_pred_ann1)
acc_ann1 <- (cm_ann1[2,2] + cm_ann1[1,1])/count(test_2)

# ANN 2
annh2 = h2o.deeplearning(y = 'Survived',
                        training_frame = as.h2o(train_2[-1]),
                        activation = 'Rectifier',
                        hidden = c(6,6),
                        epochs = 100,
                        train_samples_per_iteration = -2)

# Predicting the Test set results
prob_pred_ann2 = h2o.predict(annh2, newdata = as.h2o(test_2[c(-1,-2)]))
y_pred_ann2 = (prob_pred_ann2 > 0.5)
y_pred_ann2 = as.vector(y_pred_ann2)

# Making the Confusion Matrix
cm_ann2 = table(test_2[,2], y_pred_ann2)
acc_ann2 <- (cm_ann2[2,2] + cm_ann2[1,1])/count(test_2)

# ANN 3
annh3 = h2o.deeplearning(y = 'Survived',
                         training_frame = as.h2o(train_2[-1]),
                         activation = 'Rectifier',
                         hidden = c(100,100),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
prob_pred_ann3 = h2o.predict(annh3, newdata = as.h2o(test_2[c(-1,-2)]))
y_pred_ann3 = (prob_pred_ann3 > 0.5)
y_pred_ann3 = as.vector(y_pred_ann3)

# Making the Confusion Matrix
cm_ann3 = table(test_2[,2], y_pred_ann3)
acc_ann3 <- (cm_ann3[2,2] + cm_ann3[1,1])/count(test_2)

h2o.shutdown()

modeltype <- c("Unrestricted Logit", "Restricted #1 Logit", "k-NN", "Linear SVM", "Kernal SVM", "Naive Bayes 1", "Naive Bayes 2","Decision Tree", "Random Forest 1", "Random Forest 2", "XGBoost", "ANN 1", "ANN 2", "ANN 3")
accuracy <- c(acc_lr1[1,1], acc_lr2[1,1], acc_knn[1,1], acc_svm[1,1], acc_ksvm[1,1], acc_nb1[1,1], acc_nb2[1,1], acc_dt[1,1], acc_rf1[1,1], acc_rf2[1,1], acc_xgb[1,1], acc_ann1[1,1], acc_ann2[1,1], acc_ann3[1,1])

acc_df <- data.frame("model_type" =  modeltype, "accuracy" = accuracy)

# Receiver Operating Characteristic Curve (ROC) ----------------------------------------------------------------
# Area Under Curve Calc
aucc <- c(auc(test$Survived,y_pred_lr1), auc(test$Survived,y_pred_lr2), auc(test$Survived,as.integer(y_pred_knn)-1),
          auc(test$Survived,as.integer(y_pred_svm)-1), auc(test$Survived,as.integer(y_pred_ksvm)-1), auc(test$Survived,as.integer(y_pred_nb1)-1),
          auc(test$Survived,as.integer(y_pred_nb2)-1), auc(test$Survived,as.integer(y_pred_dt)-1), auc(test$Survived,as.integer(y_pred_rf1)-1),
          auc(test$Survived,as.integer(y_pred_rf2)-1), auc(test$Survived,y_pred_xgb), auc(test$Survived,y_pred_ann1),
          auc(test$Survived,y_pred_ann2), auc(test$Survived,y_pred_ann3))
modeltype2 <- c("Unrestricted Logit", "Restricted #1 Logit", "k-NN", "Linear SVM", "Kernal SVM", "Naive Bayes 1", "Naive Bayes 2","Decision Tree", "Random Forest 1", "Random Forest 2", "XGBoost", "ANN 1", "ANN 2", "ANN 3")
auctab <- data.frame("model_type" = modeltype2, "aucc" = aucc)

rm(modeltype2)
rm(aucc)

tab <- inner_join(acc_df, auctab, by=c("model_type", "model_type"))
tab <- tab[order(tab$accuracy),]

# ROC Curves 
# ANN#3
predauc <- prediction(y_pred_ann3, test$Survived)
roc <- performance(predauc,"tpr","fpr")
plot(roc, lwd=2, colorize=TRUE)
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

# Random Forest 1
predauc <- prediction(as.integer(y_pred_rf1)-1, test$Survived)
roc <- performance(predauc,"tpr","fpr")
plot(roc, lwd=2, colorize=TRUE)
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

# Naive Bayes
predauc <- prediction(as.integer(y_pred_nb1)-1,test$Survived)
roc <- performance(predauc,"tpr","fpr")
plot(roc, lwd=2, colorize=TRUE)
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

# Variable Importance --------------------------------------------------------------------------------
# names <- dimnames(data.matrix(train_2[c(-1,-2)]))[[2]]
names <- colnames(train_2[c(-1,-2)])

imatrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix = imatrix)

print(imatrix)

# k-folds Cross Validation -----------------------------------------------------------------------------
folds2 = createFolds(dataset2$Survived, k = 10)
folds3 = createFolds(dataset3$Survived, k = 10)
folds4 = createFolds(dataset4$Survived, k = 10)

# Restricted Logit Model  Cross validation
cv = lapply(folds2, function(x) {
  training_fold = dataset2[-x, ]
  test_fold = dataset2[x, ]
  classifier = glm(formula = Survived ~ Pclass + Sex_male + Age + famsize + Embarked_Q + Embarked_C,
                   family = binomial,
                   data = training_fold[,2:9])
  y_pred = predict(classifier, newdata = test_fold[c(-1,-2)])
  y_pred = ifelse(y_pred > 0.5, 1, 0)
  cm = table(test_fold[, 2], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
a_rlmcv = mean(as.numeric(cv))

# k-nn Cross Validation
cv = lapply(folds3, function(x) {
  training_fold = dataset3[-x,]
  test_fold = dataset3[x,]
  y_pred = knn(train = training_fold[,c(-1,-2)],
                   test = test_fold[,c(-1,-2)],
                   cl = training_fold[, 2],
                   k = 20)
   cm = table(test_fold[, 2], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
a_knncv = mean(as.numeric(cv))

# Naive Bayes 1 Cross validation
cv = lapply(folds2, function(x) {
  training_fold = dataset2[-x, ]
  test_fold = dataset2[x, ]
  classifier = naiveBayes(x = training_fold[c(-1,-2)], y = training_fold$Survived)
  y_pred = predict(classifier, newdata = test_fold[c(-1,-2)])
  cm = table(test_fold[, 2], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
a_nb1cv = mean(as.numeric(cv))

# Random Forest 1 Cross validation
set.seed(123)

cv = lapply(folds2, function(x) {
  training_fold = dataset2[-x, ]
  test_fold = dataset2[x, ]
  classifier = randomForest(x = training_fold[c(-1,-2)],
                            y = training_fold$Survived,
                            ntree = 100)
  y_pred = predict(classifier, newdata = test_fold[c(-1,-2)])
  cm = table(test_fold[, 2], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
a_rfcv = mean(as.numeric(cv))

# ANN 1 Cross Validation
rm(dataset) # Cleanup
dataset4 <- dataset2[-1]
dataset4$Survived <- as.integer(dataset4$Survived)-1
dataset4$Pclass <- as.integer(dataset$Pclass)

h2o.init(nthreads = -1)

cv = lapply(folds4, function(x) {
  training_fold = dataset4[-x, ]
  test_fold = dataset4[x, ]
  classifier = h2o.deeplearning(y = 'Survived',
                                training_frame = as.h2o(training_fold),
                                activation = 'Rectifier',
                                hidden = c(100,100),
                                epochs = 100,
                                train_samples_per_iteration = -2)
  prob_pred = h2o.predict(classifier, newdata = as.h2o(test_fold[-1]))
  y_pred = (prob_pred > 0.5)
  y_pred = as.vector(y_pred)
  cm = table(test_fold[,1], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
a_anncv = mean(as.numeric(cv))

h2o.shutdown()

modeltype2 <- c("Restricted #1 Logit", "k-NN",  "Naive Bayes 1", "Random Forest 1", "ANN 1")
accuracy2 <- c(a_rlmcv, a_knncv, a_nb1cv, a_rfcv, a_anncv)

acc_cv_df <- data.frame("model_type" =  modeltype2, "accuracy_cv" = accuracy2)

# Grid Search ---------------------------------------------------------------------------------------
set.seed(123)
classifier3 = train(form = Survived ~ ., data = dataset2[-1], method = 'rf')
classifier3
classifier3$bestTune

ggplot(classifier3) + scale_x_log10()
getTrainPerf(classifier3) # Similar as typing classifier

nrow(datapred[complete.cases(datapred), ])

dpred <- predict(classifier3, datapred[,-1], type = "prob")
dpred$Survived<- ifelse(dpred[,1] >0.5,0,1)  
dpred2 <- cbind(PassengerId=datapred[,1],dpred['Survived'])

write.csv(dpred2,file="pred.csv", row.names=FALSE)

# PCA -----------------------------------------------------------------------------------------------
set.seed(123)
# Random Forest
train$Pclass<- as.integer(train$Pclass)
test$Pclass<- as.integer(test$Pclass)
pcarf = preProcess(x = train[c(-1,-2)], method = 'pca', pcaComp = 2)
trainpca = predict(pcarf, train)
testpca = predict(pcarf, test)

# Fitting Random Forest to the PCA dataset
rfpca <- randomForest(x = trainpca[c(-1,-2)],
                         y = trainpca$Survived,
                         ntree = 100)

# Predicting the Test set results
y_pred_rfpca = predict(rfpca, newdata = testpca[c(-1,-2)])

# Making the Confusion Matrix
cm_rfpca = table(testpca[,2], y_pred_rfpca)
acc_rfpca <- (cm_rfpca[2,2] + cm_rfpca[1,1])/count(testpca)

# Visualising the Training set results
set <- trainpca[,-1]
set <- set[c(2,3,1)] 
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(rfpca, newdata = grid_set)
plot(set[, -3],
     main = 'Random Forest (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
set = testpca[,-1]
set = set[c(2,3,1)] 
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(rfpca, newdata = grid_set)
plot(set[, -3], main = 'Random Forest (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
