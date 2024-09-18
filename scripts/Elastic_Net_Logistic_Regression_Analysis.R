# Install and Load the packages
required_packages <- c("glmnet", "pROC", "caret", "dplyr")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
}

library(glmnet)
library(caret)
library(dplyr)
library(pROC)


data(iris)

# Convert Species into a binary variable: setosa vs. not setosa
iris$Species <- ifelse(iris$Species == "setosa", 1, 0)

head(iris)
table(iris$Species)


# Create a 70-30 train-test split
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data  <- iris[-train_index, ]

# Separate predictors and response variables
x_train <- as.matrix(train_data[, -5])
y_train <- train_data$Species

x_test <- as.matrix(test_data[, -5])
y_test <- test_data$Species

# Fit Elastic Net model with alpha = 0.5 (equal weight for Lasso and Ridge)
set.seed(42)
elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5)

print(elastic_net_model)

# Perform cross-validation to find the best lambda
set.seed(42)
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, type.measure = "class")

plot(cv_model)

best_lambda <- cv_model$lambda.min
cat("Optimal Lambda:", best_lambda, "\n")

# Predict on the test set using the best lambda
test_predictions <- predict(cv_model, s = best_lambda, newx = x_test, type = "class")

# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(test_predictions), as.factor(y_test))
print(conf_matrix)

# Calculate AUC (Area Under the Curve)
test_probabilities <- predict(cv_model, s = best_lambda, newx = x_test, type = "response")
roc_curve <- roc(y_test, as.numeric(test_probabilities))
auc(roc_curve)








