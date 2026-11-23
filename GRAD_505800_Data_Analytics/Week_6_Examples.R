##############################################################
# AI USAGE DISCLAIMER
# ------------------------------------------------------------
# This code was generated with assistance from GPT-based AI.
# Tools used include Microsoft 365 Copilot and Google Gemini.
#
# Users should independently review, validate,
# test, and modify the code before use in
# production environments.
#
# Final review, validation, testing, and approval
# remain the responsibility of the user.
#
# The author assumes no responsibility for its use.
##############################################################

##############################################################
# DATASET USAGE DISCLAIMER
# ------------------------------------------------------------
# This example uses public, open-source, synthetic,
# educational datasets, and/or pre-trained models.
#
# Users are responsible for:
#
# - Reviewing dataset licenses
# - Providing required attribution
# - Complying with usage restrictions
# - Verifying data quality and completeness
# - Ensuring privacy compliance
# - Ensuring regulatory compliance
# - Validating model performance
#
# Dataset ownership remains with the original
# dataset publishers and maintainers.
#
# The author assumes no responsibility for misuse of datasets, 
# models, predictions, analysis results, or licensing
# violations.
##############################################################

##############################################################
# WEEK 6 EXAMPLES
##############################################################
# Topics Covered:
# 1. Decision Tree
# 2. Random Forest
# 3. Support Vector Machine (SVM)
# 4. Ensemble Learning
##############################################################

# install.packages("caret")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("class")
# install.packages("neuralnet")
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("ggplot2")

library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(class)
library(neuralnet)
library(tm)
library(SnowballC)
library(ggplot2)

data(iris)

set.seed(42)
train_index <- createDataPartition(iris$Species,p=0.8,list=FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

cat("
DECISION TREE
")
tree_model <- rpart(Species ~ ., data=train_data, method="class")
tree_predictions <- predict(tree_model,test_data,type="class")
tree_accuracy <- mean(tree_predictions==test_data$Species)
cat("Accuracy:",tree_accuracy,"
")

cat("
RANDOM FOREST
")
rf_model <- randomForest(Species ~ ., data=train_data, ntree=100)
rf_predictions <- predict(rf_model,test_data)
rf_accuracy <- mean(rf_predictions==test_data$Species)
cat("Accuracy:",rf_accuracy,"
")

cat("
SVM
")
svm_model <- svm(Species ~ ., data=train_data)
svm_predictions <- predict(svm_model,test_data)
svm_accuracy <- mean(svm_predictions==test_data$Species)
cat("Accuracy:",svm_accuracy,"
")

cat("
ENSEMBLE LEARNING
")
ensemble_votes <- data.frame(Tree=as.character(tree_predictions),Forest=as.character(rf_predictions),SVM=as.character(svm_predictions))
majority_vote <- apply(ensemble_votes,1,function(x) names(sort(table(x),decreasing=TRUE))[1])
ensemble_accuracy <- mean(majority_vote==test_data$Species)
cat("Ensemble Accuracy:",ensemble_accuracy,"
")

results <- data.frame(Model=c("Decision Tree","Random Forest","SVM"),Accuracy=c(tree_accuracy,rf_accuracy,svm_accuracy))
print(results)

ggplot(results,aes(Model,Accuracy)) + geom_bar(stat="identity",fill="steelblue") + ggtitle("Model Accuracy Comparison")

cat("
WEEK 6 EXAMPLES COMPLETE
")

