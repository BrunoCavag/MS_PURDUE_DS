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
# WEEK 7 EXAMPLES
##############################################################
# Topics Covered:
# 1. K-Nearest Neighbors (KNN)
# 2. K-Means Clustering
# 3. Social Network Recommendation
# 4. Sentiment Analysis
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
KNN
")
train_x <- train_data[,1:4]
test_x <- test_data[,1:4]
train_y <- train_data$Species
knn_predictions <- knn(train=train_x,test=test_x,cl=train_y,k=5)
knn_accuracy <- mean(knn_predictions==test_data$Species)
cat("Accuracy:",knn_accuracy,"
")

cat("
NEURAL NETWORK
")
iris_nn <- iris
iris_nn$Species <- ifelse(iris_nn$Species == "setosa", 1, 0)

# Split data into Train (80%) and Test (20%) for proper evaluation
set.seed(123) # For reproducibility
train_index <- createDataPartition(iris_nn$Species, p = 0.8, list = FALSE)
train_data  <- iris_nn[train_index, ]
test_data   <- iris_nn[-train_index, ]

# Train the model on training data
nn_model <- neuralnet(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train_data, 
  hidden = 3, 
  linear.output = FALSE
)

# Generate Predictions on New (Test) Data
# The model outputs continuous probabilities between 0 and 1
nn_predictions_prob <- predict(nn_model, test_data[, 1:4])

# Convert Probabilities to Binary Classes (0 or 1)
# Using a standard 0.5 decision threshold
nn_predictions_class <- ifelse(nn_predictions_prob > 0.5, 1, 0)

# Align Data Types for Evaluation
# Both vectors must be factors with identical levels for the confusion matrix
predicted_factors <- factor(nn_predictions_class, levels = c(0, 1))
actual_factors    <- factor(test_data$Species, levels = c(0, 1))

# Build and Display the Confusion Matrix
conf_matrix <- confusionMatrix(predicted_factors, actual_factors, positive = "1")
print(conf_matrix)

cat("
K-MEANS CLUSTERING
")
cluster_data <- iris[,1:4]
kmeans_model <- kmeans(cluster_data, centers=3)

social_users <- data.frame(Age=c(25,30,22,40),Activity=c(8,5,9,3),Interest=c(7,6,8,4))
user_names <- c("UserA","UserB","UserC","UserD")
new_user <- c(24,9,7)
distances <- apply(social_users,1,function(x) sqrt(sum((x-new_user)^2)))
recommended_user <- user_names[which.min(distances)]
cat("Recommended User:",recommended_user,"
")

cat("
SENTIMENT ANALYSIS
")
texts <- c("I love this product","Amazing experience","I hate this item","Worst purchase ever","Fantastic quality","Very disappointing")
labels <- factor(c("Positive","Positive","Negative","Negative","Positive","Negative"))
corpus <- VCorpus(VectorSource(texts))
corpus <- tm_map(corpus, content_transformer(tolower))
dtm <- DocumentTermMatrix(corpus)
text_matrix <- as.matrix(dtm)
sentiment_df <- as.data.frame(text_matrix)
sentiment_df$Sentiment <- labels
sentiment_model <- naiveBayes(Sentiment ~ ., data=sentiment_df)
predictions <- predict(sentiment_model,sentiment_df)
cat("Sentiment Accuracy:",mean(predictions==labels),"
")

cat("
WEEK 7 EXAMPLES COMPLETED
")

