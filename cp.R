############################################################
## G L O B A L # V A R I A B L E S #########################
############################################################
downloadUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
trainingDataFile <- "pml-training.csv"
testingDataFile <- "pml-testing.csv"

############################################################
## F U N C T I O N S #######################################
############################################################
downloadFile <- function(url, file, destination="./") {
  if (!file.exists(paste(destination, file, sep=""))) {
      download.file(url=paste(url, file, sep=""), 
                    dest=paste(destination, file, sep=""), method="curl")
  }
}

loadOrInstallPackage <- function(packageName, ret=TRUE) {
  findPackage <- function(packageName)
    packageName %in% rownames(installed.packages())
  
  if (!findPackage(packageName)) install.packages(packageName)
  if (findPackage(packageName)) 
    suppressWarnings(suppressMessages(library(packageName, character.only=TRUE)))
  if (ret) found
}

log <- function(..., seperator=" ", show=FALSE) {
  if (show) print(paste(..., sep=seperator))
}

purgeNAColumns <- function(dataFrame, purgableThreshold=0.8) {
  dataFrame[, ! colSums(is.na(dataFrame)) / nrow(dataFrame) >= purgableThreshold]
}

obviousColumnReduction <- function(dataFrame, columns) {
  dataFrame[, ! (names(dataFrame) %in% columns)]
}

rationalizeFactorColumns <- function(dataFrame1, dataFrame2) {
  for (colName in names(dataFrame1)) {
    if (is.factor(dataFrame1[, colName])) {
      dataFrame2[, colName] <- as.factor(dataFrame2[, colName])
    }
  }
  dataFrame2
}

readCsv <- function(pathToFile, headerFlag=TRUE, naStrings=c("NA")) {
  read.csv(file=pathToFile, header=headerFlag, na.strings=naStrings)
}

############################################################
## L I B R A R Y ###########################################
############################################################
loadOrInstallPackage("caret", FALSE)
loadOrInstallPackage("rpart", FALSE)
loadOrInstallPackage("randomForest", FALSE)
loadOrInstallPackage("parallel", FALSE)
loadOrInstallPackage("doParallel", FALSE)

############################################################
## S E E D #################################################
############################################################
set.seed(212121)

############################################################
## G E T # S O U R C E #####################################
############################################################
downloadFile(downloadUrl, trainingDataFile)
downloadFile(downloadUrl, testingDataFile)

############################################################
## P A R S I N G ###########################################
############################################################
naStrings <- c("NA", "#DIV/0!", " ", "")
trainingDataFrame <- readCsv(trainingDataFile, TRUE, naStrings)
testingDataFrame <- readCsv(testingDataFile, TRUE, naStrings)

trainingDataFrame <- purgeNAColumns(trainingDataFrame)
testingDataFrame <- purgeNAColumns(testingDataFrame)

obviousReducibleColumns <- c("X", "user_name", "raw_timestamp_part_1",
                             "raw_timestamp_part_2", "cvtd_timestamp",
                             "new_window", "num_window")

trainingDataFrame <- obviousColumnReduction(trainingDataFrame, 
                                             obviousReducibleColumns)
testingDataFrame <- obviousColumnReduction(testingDataFrame, 
                                           obviousReducibleColumns)

############################################################
## T R A I N I N G # N T E S T I N G # D A T A S E T #######
############################################################
inTrain <- createDataPartition(y=trainingDataFrame$classe, p=0.6, list=FALSE)
training <- trainingDataFrame[inTrain, ]
testing <- trainingDataFrame[-inTrain, ]

############################################################
## M O D E L S #############################################
############################################################

# PARALLEL PROCESSING
cluster <- makeCluster(detectCores() -1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number=5, allowParallel=TRUE)

modelFitRandomForest <- train(classe ~ ., method="rf", data=training, trControl=fitControl)
predictionRandomForest <- predict(modelFitRandomForest, testing)
cmRandomForest <- confusionMatrix(testing$classe, predictionRandomForest)

modelFitDecisionTree <- train(classe ~ ., method="rpart", data=training, trControl=fitControl)
predictionDecisionTree <- predict(modelFitDecisionTree, testing)
cmDecisionTree <- confusionMatrix(testing$classe, predictionDecisionTree)

stopCluster(cluster)

############################################################
## E V A L U A T I O N #####################################
############################################################
names(cmDecisionTree)
cmDecisionTree$overall
computeFunc <- function (value) {
  round((1 - as.numeric(value))*100, 2)
}
percentagize <- function(value, func) {
  paste(func(value), "%")
}
percentagize(cmDecisionTree$overall[1], computeFunc)

############################################################
## P R E D I C T ###########################################
############################################################
pred <- predict(modelFitRandomForest, testingDataFrame)
pred

############################################################
## A P P E N D I X #########################################
############################################################
loadOrInstallPackage("corrplot", FALSE)
corrPlot <- cor(trainingDataFrame[, -length(names(trainingDataFrame))])
corrplot(corrPlot, method="color")

loadOrInstallPackage("rpart.plot", FALSE)
prp(modelFitDecisionTree$finalModel)
