# Install and Load the packages
required_packages <- c("randomForest", "caret", "ggplot2", "dplyr")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
}

library(randomForest)
library(caret)    # For data splitting and evaluation
library(ggplot2)  # For visualization
library(dplyr)    # For data manipulation

data(iris)

head(iris)
summary(iris)

# Create a 70-30 train-test split
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data  <- iris[-train_index, ]

prop.table(table(train_data$Species))
prop.table(table(test_data$Species))

rf_model <- randomForest(Species ~ ., data = train_data, ntree = 500, mtry = 2, importance = TRUE)

print(rf_model)

# Make predictions on the test set
test_predictions <- predict(rf_model, newdata = test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(test_predictions, test_data$Species)
print(conf_matrix)

var_imp <- importance(rf_model)
print(var_imp)

varImpPlot(rf_model, main = "Variable Importance Plot")

# Define the tuning grid
tune_grid <- expand.grid(.mtry = 1:4)

# Set up cross-validation parameters
control <- trainControl(method = "cv", number = 5, search = "grid")

# Train the model with tuning
rf_tuned <- train(Species ~ ., data = train_data, method = "rf",
                  metric = "Accuracy", tuneGrid = tune_grid, trControl = control, ntree = 500)

print(rf_tuned)
plot(rf_tuned)

# Best mtry value
best_mtry <- rf_tuned$bestTune$mtry
cat("Optimal mtry:", best_mtry, "\n")

# Train the final model with the optimal mtry
rf_final <- randomForest(Species ~ ., data = train_data, ntree = 500, mtry = best_mtry, importance = TRUE)

# Evaluate on the test set
final_predictions <- predict(rf_final, newdata = test_data)
final_conf_matrix <- confusionMatrix(final_predictions, test_data$Species)
print(final_conf_matrix)



