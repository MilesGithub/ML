# Install and Load the packages
required_packages <- c("e1071", "caret", "ggplot2", "dplyr")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
}

library(e1071)
library(caret)
library(ggplot2)
library(dplyr)

data(iris)
head(iris)
summary(iris)

# Create a 70-30 train-test split
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data  <- iris[-train_index, ]

# Verify the distribution of species in both sets
prop.table(table(train_data$Species))
prop.table(table(test_data$Species))

# Train the SVM model with a linear kernel
svm_model <- svm(Species ~ ., data = train_data, kernel = "linear", cost = 1, scale = TRUE)

summary(svm_model)

# Make predictions on the test set
test_predictions <- predict(svm_model, newdata = test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(test_predictions, test_data$Species)
print(conf_matrix)

# Set up the tuning grid
tune_grid <- expand.grid(C = 2^(-2:2), sigma = 2^(-2:2))

# Set up cross-validation parameters
control <- trainControl(method = "cv", number = 5)

# Train the SVM model with tuning
set.seed(42)
svm_tuned <- train(Species ~ ., data = train_data, method = "svmRadial",
                   metric = "Accuracy", tuneGrid = tune_grid, trControl = control)

print(svm_tuned)
plot(svm_tuned)

best_parameters <- svm_tuned$bestTune

cat("Optimal Parameters: Cost =", best_parameters$C, ", Sigma =", best_parameters$sigma, "\n")

# Train a simple SVM model with two features
svm_simple <- svm(Species ~ Sepal.Length + Sepal.Width, data = train_data, kernel = "linear", cost = 1, scale = TRUE)

plot(svm_simple, train_data, Sepal.Length ~ Sepal.Width,
     main = "SVM Decision Boundary with Sepal Length and Width",
     slice = list(Petal.Length = mean(train_data$Petal.Length), Petal.Width = mean(train_data$Petal.Width)))



