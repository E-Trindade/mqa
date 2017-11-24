# PCA

# Importing the dataset
ds = read.csv('300k.csv')

levels(ds$continent) = list(
  America=c('America/Argentina','America/Indiana','America/Kentucky'),
  Africa=c('Africa'),
  Asia=c('Asia'),
  Oceania=c('Atlantic', 'Australia'),
  Europe=c('Europe', 'Indian', 'Pacific')
)

ds$isRare <- ifelse(
  ds$pokemonId == 1 |
    ds$pokemonId == 2 |
    ds$pokemonId == 3 |
    ds$pokemonId == 4 |
    ds$pokemonId == 5 |
    ds$pokemonId == 6 |
    ds$pokemonId == 7 |
    ds$pokemonId == 8 |
    ds$pokemonId == 9 |
    ds$pokemonId == 25 |
    ds$pokemonId == 26 |
    ds$pokemonId == 27 |
    ds$pokemonId == 28 |
    ds$pokemonId == 35 |
    ds$pokemonId == 36 |
    ds$pokemonId == 37 |
    ds$pokemonId == 38 |
    ds$pokemonId == 39 |
    ds$pokemonId == 50 |
    ds$pokemonId == 51 |
    ds$pokemonId == 52 |
    ds$pokemonId == 53 |
    ds$pokemonId == 54 |
    ds$pokemonId == 55 |
    ds$pokemonId == 56 |
    ds$pokemonId == 63 |
    ds$pokemonId == 64 |
    ds$pokemonId == 65 |
    ds$pokemonId == 66 |
    ds$pokemonId == 67 |
    ds$pokemonId == 68 |
    ds$pokemonId == 72 |
    ds$pokemonId == 73 |
    ds$pokemonId == 74 |
    ds$pokemonId == 75 |
    ds$pokemonId == 76 |
    ds$pokemonId == 77 |
    ds$pokemonId == 78 |
    ds$pokemonId == 79 |
    ds$pokemonId == 80 |
    ds$pokemonId == 81 |
    ds$pokemonId == 82 |
    ds$pokemonId == 86 |
    ds$pokemonId == 87 |
    ds$pokemonId == 90 |
    ds$pokemonId == 91 |
    ds$pokemonId == 95 |
    ds$pokemonId == 100 |
    ds$pokemonId == 101 |
    ds$pokemonId == 102 |
    ds$pokemonId == 103 |
    ds$pokemonId == 104 |
    ds$pokemonId == 105 |
    ds$pokemonId == 109 |
    ds$pokemonId == 110 |
    ds$pokemonId == 111 |
    ds$pokemonId == 112 |
    ds$pokemonId == 114 |
    ds$pokemonId == 116 |
    ds$pokemonId == 117 |
    ds$pokemonId == 118 |
    ds$pokemonId == 119 |
    ds$pokemonId == 123 |
    ds$pokemonId == 124 |
    ds$pokemonId == 125 |
    ds$pokemonId == 126 |
    ds$pokemonId == 128 |
    ds$pokemonId == 134 |
    ds$pokemonId == 135 |
    ds$pokemonId == 136 |
    ds$pokemonId == 138 |
    ds$pokemonId == 139 |
    ds$pokemonId == 140 |
    ds$pokemonId == 141 |
    ds$pokemonId == 143 |
    ds$pokemonId == 147 |
    ds$pokemonId == 148 |
    ds$pokemonId == 149,
  TRUE, FALSE)

# Muito raros
ds$isVeryRare <- ifelse(
  ds$pokemonId == 88 |
    ds$pokemonId == 89 |
    ds$pokemonId == 106 |
    ds$pokemonId == 107 |
    ds$pokemonId == 108 |
    ds$pokemonId == 113 |
    ds$pokemonId == 130 |
    ds$pokemonId == 137 |
    ds$pokemonId == 142 |
    ds$pokemonId == 83 |
    ds$pokemonId == 132 |
    ds$pokemonId == 144 |
    ds$pokemonId == 145 |
    ds$pokemonId == 146 |
    ds$pokemonId == 150 |
    ds$pokemonId == 151 |
    ds$pokemonId == 115 |
    ds$pokemonId == 122 |
    ds$pokemonId == 131,
  TRUE, FALSE)

ds$classif <- ifelse(ds$isRare == TRUE, "rare", "NONE")
ds$classif <- ifelse(ds$isVeryRare == TRUE, "very rare", ds$classif)
ds$classif <- ifelse(ds$classif == "NONE", "common", ds$classif)


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(ds$classif, SplitRatio = 0.8)
training_set = subset(ds, split == TRUE)
test_set = subset(ds, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))