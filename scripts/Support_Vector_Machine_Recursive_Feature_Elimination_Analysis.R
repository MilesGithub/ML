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

prop.table(table(train_data$Species))
prop.table(table(test_data$Species))


# Define the control using the rfeControl function
control <- rfeControl(functions = caretFuncs, method = "cv", number = 10)

# Train the SVM-RFE model
set.seed(42)
svm_rfe <- rfe(train_data[, 1:4], train_data$Species,
               sizes = c(1:4),  # Number of features to select
               rfeControl = control,
               method = "svmRadial")

print(svm_rfe)

# Best subset of features
selected_features <- predictors(svm_rfe)
cat("Selected Features:", selected_features, "\n")

# Train the final SVM model using the selected features
final_svm_model <- svm(Species ~ ., data = train_data[, c(selected_features, "Species")],
                       kernel = "radial", cost = 1, gamma = 0.1)

summary(final_svm_model)


# Make predictions on the test set using the selected features
test_predictions <- predict(final_svm_model, newdata = test_data[, selected_features])

# Confusion Matrix
conf_matrix <- confusionMatrix(test_predictions, test_data$Species)
print(conf_matrix)












