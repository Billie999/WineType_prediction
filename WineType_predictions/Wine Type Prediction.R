
# title: "Wine Type Prediction With Supervised And Unsupervised Learning"
# author: "Biljana Simonovikj"
# date: "06/05/2020"


#------------------------------------------------------
# Install and load the liberaries used in this project:
#------------------------------------------------------
library(GGally)
library(gridExtra)
library(RColorBrewer)
library(factoextra)
library(FactoMineR)
library(caret)
library(pander)
library(reshape2)
library(dbscan)
library(dplyr)
library(stats)
library(graphics)
library(reshape)
library(tidyverse)
library(naivebayes)
library(dendextend)
library(MASS)
library(ggpubr)

#=======================
# Create metadata table:
#========================
# First, we assign the columns: row_id, var_names, type, class and description, define the column names and connect them into dataframe
# that is that is converted into table with pander package:
row_id <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "Output", 1)
var_names <- c("fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide",
               "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol", "Name", "quality")
type <- c("continuous","continuous","continuous","continuous","continuous","continuous","continuous","continuous",
          "continuous","continuous", "continuous", "Type", "categorical")
class <- c("measure", "measure", "measure", "measure", "measure", "measure", "measure", "measure", "measure", "measure",
           "measure", "Class", "category")
description <- c("acidity concentration, (tartaric acid) g/dm3", "volatile acidity concentraion, g/dm3",
                 "citric acid concentration, g/dm3", "residual sugar concentration, g/dm3", "chlorides concentration, g/dm3",
                 "free sulfur dioxide concentration, mg/dm3", "total sulfur dioxide concentration, mg/dm3", "density concentration, g/dm3",
                 "estimation of pH level", "sulphates concentration, g/dm3", "alcohol percentage in absolute units",
                 "Description", "score of wine quality between 0 and 10")
col_names <- c("Input", "Name", "Type", "Class", "Description")
metadata_df <- cbind(row_id, var_names, type, class, description)
colnames(metadata_df) <- col_names
set.alignment("left", row.names = "left")
pander(metadata_df, style = 'rmarkdown',
       caption = "Data dictionary for red and white wine datasets", split.table = Inf)
#---------------------------------------
# Download and import the datasets
#---------------------------------------
red_wine <- read.csv(file = "/Users/Biljana/Datasets/Data 7/winequality-red.csv", sep = ";", header = T)
white_wine <- read.csv(file = "/Users/Biljana/Datasets/Data 7/winequality-white.csv", sep = ";", header = T)
dim(red_wine) # retrieve dimensions of the data frame
dim(white_wine) # retrieve dimensions of the data frame

# Introduce new variable wine_type:
#----------------------------------
red_wine$wine_type = rep("red", nrow(red_wine))
white_wine$wine_type = rep("white", nrow(white_wine))

# Connect the two datasets:
#--------------------------
wine_data <- rbind(red_wine, white_wine)
dim(wine_data)
summary(wine_data) # summary statistics of the variables
str(wine_data) # compactly display the structure of an arbitrary R object
map_dbl(wine_data, function(.x) {sum(is.na(.x))}) # determine the number of missing values in each variable
table(wine_data$wine_type) # frequency of the class label
# Define column names:
#---------------------
column_names <- c("fixed_acidity", "volatile_acidity", "citric_acid",
                  "residual_sugar", "chlorides", "free_sulfur_dioxide",
                  "total_sulfur_dioxide", "density", "pH",
                  "sulphates", "alcohol", "quality", "wine_type")

colnames(wine_data) <- column_names # apply column names

# Remove quality variable from the dataset:
#------------------------------------------
wine_data <- wine_data[ ,-12]

# Challange with unbalanced dataset:
#==================================
'%ni%' <- Negate('%in%') # define "not in" function
options(scipen = 999) # prevent printing scientific notations

# # Down sample of class label:
#-----------------------------
set.seed(0)
data_class_balance <- downSample(x = wine_data[, colnames(wine_data) %ni% factor(wine_data$wine_type)],
                                  y = factor(wine_data$wine_type))
data_class_balance <- data_class_balance[ ,-13] # removing the extra Class variable created
wine_classe <- data_class_balance[ ,-c(1:11)] # select the class label column
wine_predictors <- data_class_balance[ ,1:11] # select predictor variables by column number
summary(wine_predictors) # statistical summary of wine predictors
wine_classe <- data_class_balance$wine_type # assign the class label

# Define categorical class label as a factor with 2 levels: "red" and "white"
#----------------------------------------------------------------------------
wine_classe <- factor(data_class_balance$wine_type, levels = c("red","white"), ordered = FALSE)
# Equal distribution of red and white samples in training data subset:
#--------------------------------------------------------------------
table(wine_classe)

# Normalization of the dataset:
#==============================
wine_predictors_scaled <- scale(wine_predictors)
wine_data_scaled <- cbind(wine_predictors, wine_classe) # connect the predictors with class label
wine_data_scaled$wine_classe <- factor(wine_data_scaled$wine_classe, levels = c("red","white"), ordered = FALSE) # assigning class label as factor
#=================================================================
# Split the data into train and test set according to caret package
#==================================================================
set.seed(0)
train_data_index <- createDataPartition(y = wine_data_scaled$wine_classe,
                                        p = 0.80,
                                        list = FALSE)
# Train and test sets for wine type:
#----------------------------------
test_set <- wine_data_scaled[-train_data_index, ]
train_set <- wine_data_scaled[train_data_index, ]
wine_class <- factor(train_set$wine_classe, levels = c("red","white"), ordered = FALSE)

# Class distribution of train data:
#---------------------------------
table(wine_class)

#===================================
# Exploratory Analysis of Train Data:
#===================================
fixed_acidity <- train_set$fixed_acidity
volatile_acidity <- train_set$volatile_acidity
citric_acid <- train_set$citric_acid
residual_sugar <- train_set$residual_sugar
chlorides <- train_set$chlorides
free_sulfur_dioxide <- train_set$free_sulfur_dioxide
total_sulfur_dioxide <- train_set$total_sulfur_dioxide
density <- train_set$density
pH <- train_set$pH
sulphates <- train_set$sulphates
alcohol <- train_set$alcohol

# Feature plots of training dataset:
#----------------------------------
caret::featurePlot(x = train_set[ ,c("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                     "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                                     "pH", "sulphates","alcohol")], y = train_set$wine_classe,
                   plot = "density", scales = list(x = list(relation = "free"),
                                                   y = list(relation = "free")),
                   adjust = 1.5,pch = "|",layout = c(3,4),auto.key = list(columns = 2))

caret::featurePlot(train_set[ ,c("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                 "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                                 "pH", "sulphates","alcohol")], y = train_set$wine_classe,
                   plot = "ellipse",auto.key = list(columns = 2))

caret::featurePlot(train_set[ ,c("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                 "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                                 "pH", "sulphates","alcohol")], y = train_set$wine_classe,
                   plot = "box", scales = list(y = list(relation = "free"), x = list(rot = 90)), layout = c(4,3))

# Visually explore correalations of training data subset  - Correlation plot:
#============================================================================
ggcorr(train_set[ , -12], label = TRUE)

# Check normality of training dataset - QQ - plots:
#==================================================
plot_1 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "fixed_acidity", color = c("#00AFBB"), y = "fixed_acidity")

plot_2 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "volatile_acidity", color = c("#00AFBB"), y = "volatile_acidity")

plot_3 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "citric_acid", color = c("#00AFBB"), y = "citric_acid")

plot_4 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "residual_sugar", color = c("#00AFBB"), y = "residual_sugar")

plot_5 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "chlorides", color = c("#00AFBB"), y = "chlorides")

plot_6 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "free_sulfur_dioxide", color = c("#00AFBB"), y = "alcohol")

plot_7 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "total_sulfur_dioxide", color = c("#00AFBB"), y = "alcohol")

plot_8 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "density", color = c("#00AFBB"), y = "density")

plot_9 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "pH", color = c("#00AFBB"), y = "pH")

plot_10 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "sulphates", color = c("#00AFBB"), y = "sulphates")

plot_11 <- ggqqplot(train_set, title = "NORMAL Q-Q PLOT", x = "alcohol", color = c("#00AFBB"), y = "alcohol")

grid.arrange(plot_1, plot_2, plot_3, plot_4, plot_5, plot_6, plot_7, plot_8, plot_9, plot_10, plot_11, ncol = 3) # side-by-side Q-Q plots

# Removing Sulphur Dioxide and Total Sulphur Dioxide:
#====================================================
train_set <- train_set[ ,-c(6:7)]
test_set <- test_set[ , -c(6:7)]

#========================
# PCA Component Analysis
#========================
# Applying PCA to 9-dimensional space of predictor variables:
#===========================================================
pca_wine <- prcomp(wine_predictors, scale = TRUE, center = TRUE) # apply prcomp() from stats package

# Compute hierarchical clustering on principal components

# Inspect PCA results with get_eigenvalue() from factoextra package:
#------------------------------------------------------------------
# Present the PCA results in a tabular form:
#------------------------------------------
get_eigenvalue(pca_wine) # get eigenvalues, variance and cummulative variance
eig <- get_eigenvalue(pca_wine) # assign PCA results into a variable
df_eig  <- data.frame(eigenvalue = eig[ ,1], variance = eig[ ,2], cumvariance = eig[ ,3]) # create a dataframe
pcs <- c("PC1","PC2","PC3","PC4","PC5","PC6", "PC7", "PC8", "PC9", "PC10", "PC11") # create a vector of PCs names
eig_df <- cbind(pcs, df_eig) # connect to the datafame
column_names <- c("PCs", "Eigenvalue", "Variance %", "Cummulative Variance %") # define column names
colnames(eig_df) <- column_names # set column names
# Present the table with pander package:
pander(eig_df, style = 'rmarkdown',
                      caption = "Proportion of variance explained (PVE) and accumulated sum of proportion
       of variance for each of the eleven principal components.", split.table = Inf)

# Extract the results for variables from the PCA output:
#------------------------------------------------------
var <- get_pca_var(pca_wine) # assign the results in a variable
var$coord # extract coordinates (loadings) of the variables
var$contrib # extract variables contribution
var$cos2 # extract quality of representaion of the variables on factor map

# Create dataframe of loadings of variables to first 2 Principal Components:
#----------------------------------------------------------------------------
loading_var <- var$coord # loading of the variables
loading_var <-  as.data.frame(loading_var) # assign as dataframe
loading_var <- loading_var[ , 1:2] # isloate loading on the first two PCs
col_names <- c("PC1", "PC2") # define row names
colnames(loading_var) <- col_names # set row names
# Present the table with pander package:
pander(loading_var, style = 'rmarkdown',
        caption = "Variable loadings (coordinates) on Principal Components 1 and 2.", split.table = Inf)

# Variable Correlation Plot with class labels:
#--------------------------------------------
ind_var <- fviz_pca_biplot(pca_wine, label = "var", habillage = wine_classe,
                           palette = c("#BB0C00", "#E7B800"),
                           addEllipses = TRUE, ellipse.level = 0.95,
                           legend.title = "Wine Type")
ggpubr::ggpar(ind_var,
              title = "Principal Component Analysis",
              subtitle = "Correlations of the variables and class labels (red and white wine type) with PCA Components 1 and 2",
              caption = "Data source: UCI Maschine Learning Repository,
              https://archive.ics.uci.edu/ml/datasets/wine",
              xlab = "PC1", ylab = "PC2",
              legend.title = "Wine Type", legend.position = "top",
              ggtheme = theme_gray())

# Correlation plots of cos2 of variables on all the PCs and for the most contributing variables:
#------------------------------------------------------------------------------------------------
par(mfrow = c(2,1))
plot1 <- corrplot(var$cos2, is.corr = FALSE)
plot2 <- corrplot(var$contrib, is.corr = FALSE)

# Hierachical Clustering:
#========================
# Create a data frame of first 3 PCA components:
#-----------------------------------------------
pca_3 = data.frame(pca_wine$x[, 1:3])

pc_1 <- pca_3$pc_1 # assign PC1
pc_2 <- pca_3$pc_2 # assign PC2
pc_3 <- pca_3$pc_3 # assign PC3

wine_matrix <- dist(pca_3, method = "euclidean", diag = FALSE, upper = FALSE, p = 2) # distance matrix computation of class "dist"
# object by using the specified distance measure (euclidean) that computes the dissimilarity distances between the rows of a data matrix
head(as.matrix(wine_matrix)) # conversion to conventional distance matrix

# Use the distance matrix as input to call the Single-Linkage clustering algorithm available from the base R package stats and plot the resulting dendrogram.
wine_sl <- hclust(wine_matrix, method = "single") # hierarchical cluster analysis with the Single-Linkage clustering algorithm

plot(wine_sl, xlab = "", sub = "", cex = 0.6, hang = -1, col = "red3", labels = FALSE,
     main = "Cluster Dendrogram with Single Linkage Method") # plot the dendogram with Single-Linkage method

# Use the distance matrix as input to call the Complete-Linkage clustering algorithm available from the base R package stats and plot the resulting dendrogram.
wine_cl <- hclust(wine_matrix, method = "complete") # hierarchical cluster analysis with the Complete-Linkage clustering algorithm

plot(wine_cl, xlab = "", sub = "", cex = 0.6, hang = -1, col = "turquoise3", labels = FALSE,
     main = "Cluster Dendrogram with Complete Linkage Method") # plot the dendrogram with Complete-Linkage method

# Use the distance matrix as input to call the Average-Linkage clustering algorithm available from the base R package stats and plot the resulting dendrogram.
wine_al <- hclust(wine_matrix, method = "average") # hierarchical cluster analysis with the Average-Linkage clustering algorithm

plot(wine_al, xlab = "", sub = "", cex = 0.6, hang = -1, col = "slateblue2", labels = FALSE,
     main = "Cluster Dendrogram with Average Linkage Method") # plot the dendrogram with Average-Linkage method

# Use the distance matrix as input to call Ward’s clustering algorithm available from the base R package stats and plot the resulting dendrogram.
wine_wl <- hclust(wine_matrix, method = "ward.D2") # hierarchical cluster analysis with Ward’s 2 clustering algorithm

plot(wine_wl, xlab = "", sub = "", cex = 0.6, hang = -1, col = "maroon3", labels = FALSE,
     main = "Cluster Dendrogram with Ward 2 Linkage Method") # plot the dendrogram with Ward's 2 Linkage method

# Plot cluster dendrograms with Ward's 2 Linkage method along with class labels:
#===============================================================================
dendrogram_wl_wine <- as.dendrogram(wine_wl) %>%
  set("branches_lty", 1) %>%
  set("branches_k_color", value = c("black", "red"), k = 2)
  colours_to_use <- as.numeric(wine_classe)
  colours_to_use <- colours_to_use[order.dendrogram(dendrogram_wl_wine)]
  labels_colors(dendrogram_wl_wine) <- colours_to_use
  dend_list_wl <- as.character(wine_classe)
  labels(dendrogram_wl_wine) <- dend_list_wl[order.dendrogram(dendrogram_wl_wine)]
  plot(dendrogram_wl_wine, main = "Cluster Dendrogram with Ward 2 Linkage Method", ylab = "Height")
dendrogram_wl_rect_wine <- rect.dendrogram(dendrogram_wl_wine, k = 2, lty = 5, lwd = 0, x = 1, col = rgb(0.1, 0.2, 0.4, 0.1))
legend("topright",
       legend = c("Red Wine","White Wine"),
       col = c("black", "red"),
       title = "Cluster Labels",
       pch = c(20,20), bty = "n", pt.cex = 1.5, cex = 0.8,
       text.col = c("black"), horiz = F, inset = c(0,0.1))

# Build Naive_Bayes Model:
#=========================
nb_fit = naive_bayes(train_set$wine_classe~ fixed_acidity + volatile_acidity  +
                       residual_sugar + density +
                       pH + alcohol + sulphates + citric_acid + chlorides, train_set[ ,-10])
nb_fit

# Measures of predicted classes for NB model on training data:
#=============================================================
head(predict(nb_fit, data = train_set[ ,-10], type = "prob"))
predmodel_train_nb = predict(nb_fit, data = train_set[ ,-10], type = "class")
table(predicted = predmodel_train_nb, observed = train_set$wine_classe)
confusionMatrix(predmodel_train_nb, train_set$wine_classe,
                mode = "everything", positive = "red")

# Measures of predicted classes for NB model on testing data:
#============================================================
predmodel_test_nb = predict(nb_fit, newdata = test_set[ ,-10])
table(predicted = predmodel_test_nb, observed = test_set$wine_classe)
confusionMatrix(predmodel_test_nb, test_set$wine_class,
                mode = "everything", positive = "red")

# Build LDA Model:
#=================
lda_fit <- lda(train_set$wine_classe~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides
               + density + pH + sulphates + alcohol, data = train_set[, -10])
lda_fit

# Measures of predicted classes for LDA model on training data:
#============================================================
predmodel_train_lda = predict(lda_fit, data = train_set[ ,-10])
predmodel_train_lda = predict(lda_fit, data = train_set[ ,-10])
predmodel_train_lda$x[,1] # contains the values for the first discriminant function
table(predicted = predmodel_train_lda$class, observed = train_set$wine_classe)
confusionMatrix(predmodel_train_lda$class, train_set$wine_classe,
                mode = "everything", positive = "red")

# Measures of predicted classes for LDA model on testing data:
#============================================================
predmodel_test_lda = predict(lda_fit, newdata = test_set[ ,-10])
table(predicted = predmodel_test_lda$class, observed = test_set$wine_classe)
confusionMatrix(predmodel_test_lda$class, test_set$wine_classe,
                mode = "everything", positive = "red")


# Build QDA Model:
#=================

qda_fit <- qda(train_set$wine_classe~ fixed_acidity + volatile_acidity  +
                 residual_sugar + density +
                 pH + alcohol + sulphates + citric_acid + chlorides, data = train_set[ ,-10])
qda_fit

#  Measures of predicted classes for QDA model on training data:
#===============================================================
predmodel_train_qda = predict(qda_fit, data = train_set[ ,-10], type = "class")
table(predicted = predmodel_train_qda$class, observed = train_set$wine_classe)
confusionMatrix(predmodel_train_qda$class, train_set$wine_classe,
                mode = "everything", positive = "red")

# Measures of predicted classes for QDA model on testing data:
#============================================================
predmodel_test_qda = predict(qda_fit, newdata = test_set[ ,-10])
table(predicted = predmodel_test_qda$class, observed = test_set$wine_classe)
confusionMatrix(predmodel_test_qda$class, test_set$wine_classe,
                mode = "everything", positive = "red")

# Calculate metrics performance of the models:
#=============================================
metrics_classification = function(predicted, observed){
  (confusion_table = table(predicted, observed)) # create the confusion matrix
  TP = confusion_table[1,1]
  TN = confusion_table[2,2]
  FN = confusion_table[2,1]
  FP = confusion_table[1,2]
  accuracy = round((TP + TN) / sum(TP,FP,TN,FN), 2)
  error_rate = round((FP + FN) / sum(TP,FP,TN,FN),2)
  precision = round(TP / sum(TP, FP), 2)
  recall = round(TP / sum(TP, FN), 2)
  sensitivity = round(TP / sum(TP, FN), 2)
  specificity = round(TN / sum(TN, FP), 2)
  f1_score = round((2 * precision * sensitivity) / (precision + sensitivity), 2)
  metrics = c(accuracy, error_rate, precision, recall, sensitivity, specificity, f1_score)
  names(metrics) = c("Accuracy", "Error_Rate", "Precision", "Recall", "Sensitivity", "Specificity", "F1 score")
  return(metrics)
}
# Calculate model performance metrics on training data:
#=====================================================
lda_train <- metrics_classification(predmodel_train_lda$class,train_set$wine_classe)
qda_train <- metrics_classification(predmodel_train_qda$class, train_set$wine_classe)
nb_train <- metrics_classification(predmodel_train_nb, train_set$wine_classe)

# Calculate model performance metrics on testing data:
#=====================================================
lda_test <- metrics_classification(predmodel_test_lda$class,test_set$wine_classe)
qda_test <- metrics_classification(predmodel_test_qda$class, test_set$wine_classe)
nb_test <- metrics_classification(predmodel_test_nb, test_set$wine_classe)

# Table for performance metrics on train data of 3 classifiers:
#==============================================================
col_names <- c("Parameters", "LDA", "QDA", "NB")
parameters <- c("Accuracy", "Error_Rate", "Precision", "Recall", "Sensitivity", "Specificity", "F1 score")
metrics_df_train <- cbind(parameters, lda_train, qda_train, nb_train)
colnames(metrics_df_train) <- col_names
row.names(metrics_df_train) <- NULL
set.alignment("left", row.names = "left")
pander(metrics_df_train, style = 'rmarkdown',
       caption = " Performances metrics of naive_Bayes, LDA and QDA classifiers on train wine data", split.table = Inf)

# Table for performance metrics on test data of 3 classifiers:
#=============================================================
col_names <- c("Parameters", "LDA", "QDA", "NB")
parameters <- c("Accuracy", "Error_Rate", "Precision", "Recall", "Sensitivity", "Specificity", "F1 score")
metrics_df_test <- cbind(parameters, lda_test, qda_test, nb_test)
colnames(metrics_df_test) <- col_names
row.names(metrics_df_test) <- NULL
set.alignment("left", row.names = "left")
pander(metrics_df_test, style = 'rmarkdown',
       caption = "Performances metrics of naive_Bayes, LDA and QDA classifiers on test wine data", split.table = Inf)

# Overall metrics performance of 3 classifiers on train and test data:
#=====================================================================
metrics_df_train <- as.data.frame(metrics_df_train)
Parameters <- metrics_df_train$Parameters
metrics_df <- merge(metrics_df_train, metrics_df_test, by = "Parameters")
col_names <- c("Parameters", "LDA.trn", "QDA.trn", "NB.trn", "LDA.tst", "QDA.tst", "NB.tst")
colnames(metrics_df) <- col_names
row.names(metrics_df) <- NULL
set.alignment("left", row.names = "left")
pander(metrics_df, style = 'rmarkdown',
       caption = "Performances metrics of naive_Bayes, LDA and QDA classifiers on train and test wine data", split.table = Inf)

