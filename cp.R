############################################################
## G L O B A L # V A R I A B L E S #########################
############################################################
downloadUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
trainingDataFile <- "pml-training.csv"
testingDataFile <- "pml-testing.csv"
predicteeColumn <- "classe"

############################################################
## F U N C T I O N S #######################################
############################################################
downloadFile <- function(url, file, destination="./") {
  if (file.exists(paste(destination, file, sep=""))) return
  
  download.file(url=paste(url, file, sep=""), 
                dest=paste(destination, file, sep=""), method="curl")
}

loadOrInstallPackage <- function(packageName, ret=TRUE) {
  findPackage <- function(packageName)
    packageName %in% rownames(installed.packages())
  
  # If not found then install
  if (!findPackage(packageName)) install.packages(packageName)
  
  # If found then load to environment
  if (findPackage(packageName)) library(packageName, character.only=TRUE)
  
  # TRUE - installed / loaded, FASLE - possibly wrong package name
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
## G E T # S O U R C E #####################################
############################################################
downloadFile(downloadUrl, trainingDataFile)
# downloadFile(downloadUrl, testingDataFile) # Not required

############################################################
## P A R S I N G ###########################################
############################################################
naStrings <- c("NA", "#DIV/0!", " ", "")
trainingDataFrame <- readCsv(trainingDataFile, TRUE, naStrings)
# testingDataFrame <- readCsv(testingDataFile, TRUE, naStrings)

trainingDataFrame_ <- purgeNAColumns(trainingDataFrame)
# testingDataFrame_ <- purgeNAColumns(testingDataFrame)

obviousReducibleColumns <- c("X", "user_name", "raw_timestamp_part_1",
                             "raw_timestamp_part_2", "cvtd_timestamp", "new_window", 
                             "num_window")

trainingDataFrame_ <- obviousColumnReduction(trainingDataFrame_, obviousReducibleColumns)
# testingDataFrame_ <- obviousColumnReduction(testingDataFrame_, obviousReducibleColumns)

testingDataFrame_ <- rationalizeFactorColumns(obviousColumnReduction(trainingDataFrame_, c(predicteeColumn)), testingDataFrame_)

nzv <- nearZeroVar(trainingDataFrame_, saveMetrics=TRUE)
trainingDataFrame__ <- trainingDataFrame_[, !nzv$nzv] # No further reduction possible

############################################################
## M A I N #################################################
############################################################
loadOrInstallPackage("caret", FALSE)

set.seed(19622)

sampleIndexes <- sample(1:nrow(trainingDataFrame_), 100)
sampleData <- trainingDataFrame_[sampleIndexes, ]
sampleTest <- trainingDataFrame_[c(100:200), ]
sum(complete.cases(sampleData)) == nrow(sampleData)

# PARALLEL PROCESSING
loadOrInstallPackage("parallel", FALSE)
loadOrInstallPackage("doParallel", FALSE)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number=3, allowParallel=TRUE)

modelFit <- train(classe ~ ., method="rf", data=sampleData, trControl=fitControl)
stopCluster(cluster)

predictions<-predict(modelFit, sampleTest)
confusionMatrix(sampleTest$classe, predictions)
