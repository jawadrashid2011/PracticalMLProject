#Read in the dataset
#Data Source http://groupware.les.inf.puc-rio.br/har
#Include Citation
library(caret)
library(kernlab)
library(rattle)
library(rpart)
library(gbm)
library(randomForest)
actualAnswer <- c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B")

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=paste("predictions",filename,sep = "/"),quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}


allData <- read.csv("pml-training.csv")
cases <- read.csv("pml-testing.csv")
actualAnswers <- c("B", "A", )


allNAs <- which(sapply(cases, function(x) all(is.na(x))))
removeCols <- union(allNAs, c(1:5))
cases <- cases[, -union(removeCols,ncol(cases))]
filteredData <- allData[, -removeCols]

set.seed(1)
inTrain <- createDataPartition(y = filteredData$classe, p=0.75, list=FALSE)
training <- filteredData[inTrain, ]
testing <- filteredData[-inTrain, ]

#BASE
set.seed(2)
baseOutput <- factor(rep("A", nrow(testing)), levels=c("A","B","C","D","E"))
baseTraining <- factor(rep("A", nrow(training)), levels=c("A","B","C","D","E"))
baseTrainConfusion <- confusionMatrix(training$classe, baseTraining)
baseConfusion <- confusionMatrix(testing$classe, baseOutput)
print(baseTrainConfusion)
print(baseConfusion)
casesConfusionBase <- (confusionMatrix(actualAnswer, factor(rep("A", nrow(cases)), levels=c("A","B","C","D","E"))))
print(casesConfusionBase)

#RPart Without Any Modification
set.seed(3)
ptm <- proc.time()
rPartSimpleModel <- train(classe ~ ., data=training, method="rpart")
rPartSimpleTime <- proc.time() - ptm
print(rPartSimpleTime)

print(rPartSimpleModel$finalModel)
fancyRpartPlot(rPartSimpleModel$finalModel)

rPartSimpleTrain <- predict(rPartSimpleModel, newdata=training)
rPartSimpleTest <- predict(rPartSimpleModel, newdata=testing)

rPartSimpleTrainConfusion <- confusionMatrix(training$classe, rPartSimpleTrain)
rPartSimpleTestConfusion <- confusionMatrix(testing$classe, rPartSimpleTest)
print(rPartSimpleTrainConfusion)
print(rPartSimpleTestConfusion)
casesConfusionSimplerPart <- confusionMatrix(actualAnswer, predict(rPartSimpleModel, cases))
print(casesConfusionSimplerPart)

#RPart With CV
set.seed(4)
ptm <- proc.time()
rPartSimpleModelCV <- train(classe ~ ., data=training, method="rpart", trControl=trainControl(method="cv"))
rPartSimpleTimeCV <- proc.time() - ptm
print(rPartSimpleTimeCV)

print(rPartSimpleModelCV$finalModel)
fancyRpartPlot(rPartSimpleModelCV$finalModel)

rPartSimpleTrainCV <- predict(rPartSimpleModelCV, newdata=training)
rPartSimpleTestCV <- predict(rPartSimpleModelCV, newdata=testing)

rPartSimpleTrainConfusionCV <- confusionMatrix(training$classe, rPartSimpleTrainCV)
rPartSimpleTestConfusionCV <- confusionMatrix(testing$classe, rPartSimpleTestCV)
print(rPartSimpleTrainConfusionCV)
print(rPartSimpleTestConfusionCV)
casesConfusionSimplerPartCV <- confusionMatrix(actualAnswer, predict(rPartSimpleModelCV, cases))
print(casesConfusionSimplerPartCV)


#PRPart Scaled without PCA
set.seed(5)
ptm <- proc.time()
trainingPreProc <- preProcess(training[,-c(1,ncol(training))], 
                              method=c("BoxCox", "center", 
                                       "scale"))
trainingAfterProcess <- predict(trainingPreProc, training[,-c(1,ncol(training))])
testingAfterProcess <- predict(trainingPreProc, testing[,-c(1,ncol(testing))])
trainingAfterProcess$new_window = training$new_window
testingAfterProcess$new_window = testing$new_window
# rPartSimpleModelRCVScaled <- train(classe ~ ., data=training, method="rpart", trControl=trainControl(method="cv"), preProcess = c("BoxCox", "center", "scale"))
rPartSimpleModelRCVScaled <- train(training$classe ~ ., data=trainingAfterProcess, method="rpart", trControl=trainControl(method="cv"))
rPartSimpleTimeRCVScaled <- proc.time() - ptm
print(rPartSimpleTimeRCVScaled)

print(rPartSimpleModelRCVScaled$finalModel)
fancyRpartPlot(rPartSimpleModelRCVScaled$finalModel)

rPartSimpleTrainRCVScaled <- predict(rPartSimpleModelRCVScaled, newdata=trainingAfterProcess)
rPartSimpleTestRCVScaled <- predict(rPartSimpleModelRCVScaled, newdata=testingAfterProcess)

rPartSimpleTrainConfusionRCVScaled <- confusionMatrix(training$classe, rPartSimpleTrainRCVScaled)
rPartSimpleTestConfusionRCVScaled <- confusionMatrix(testing$classe, rPartSimpleTestRCVScaled)
print(rPartSimpleTrainConfusionRCVScaled)
print(rPartSimpleTestConfusionRCVScaled)
casesScaled <- predict(trainingPreProc, cases[,-c(1)])
casesScaled$new_window = cases$new_window
casesConfusionSimplerPartRCVScaled <- confusionMatrix(actualAnswer, predict(rPartSimpleModelRCVScaled, casesScaled))
print(casesConfusionSimplerPartRCVScaled)


###Random Forest Simple
set.seed(3)
ptm <- proc.time()
rPartSimpleModel <- train(classe ~ ., data=training, method="rpart")
rPartSimpleTime <- proc.time() - ptm
print(rPartSimpleTime)

print(rPartSimpleModel$finalModel)
fancyRpartPlot(rPartSimpleModel$finalModel)

rPartSimpleTrain <- predict(rPartSimpleModel, newdata=training)
rPartSimpleTest <- predict(rPartSimpleModel, newdata=testing)

rPartSimpleTrainConfusion <- confusionMatrix(training$classe, rPartSimpleTrain)
rPartSimpleTestConfusion <- confusionMatrix(testing$classe, rPartSimpleTest)
print(rPartSimpleTrainConfusion)
print(rPartSimpleTestConfusion)
casesConfusionSimplerPart <- confusionMatrix(actualAnswer, predict(rPartSimpleModel, cases))
print(casesConfusionSimplerPart)

#Random Forest
set.seed(40)
ptm <- proc.time()
rfSimpleModel <- train(classe ~ ., data=training, method="rf", trControl=trainControl(method="cv"))
rfSimpleTime <- proc.time() - ptm
print(rfSimpleTime)

print(rfSimpleModel$finalModel)
fancyRpartPlot(rfSimpleModel$finalModel)

rfSimpleTrain <- predict(rfSimpleModel, newdata=training)
rfSimpleTest <- predict(rfSimpleModel, newdata=testing)

rfSimpleTrainConfusion <- confusionMatrix(training$classe, rfSimpleTrain)
rfSimpleTestConfusion <- confusionMatrix(testing$classe, rfSimpleTest)
print(rfSimpleTrainConfusion)
print(rfSimpleTestConfusion)
rfCasesConfusion <- confusionMatrix(actualAnswer, predict(rfSimpleModel, cases))
print(rfCasesConfusion)

#GBM
set.seed(50)
ptm <- proc.time()
gbmSimpleModel <- train(classe ~ ., data=training, method="gbm", trControl=trainControl(method="cv"))
gbmSimpleTime <- proc.time() - ptm
print(gbmSimpleTime)

print(gbmSimpleModel$finalModel)

gbmSimpleTrain <- predict(gbmSimpleModel, newdata=training)
gbmSimpleTest <- predict(gbmSimpleModel, newdata=testing)

gbmSimpleTrainConfusion <- confusionMatrix(training$classe, gbmSimpleTrain)
gbmSimpleTestConfusion <- confusionMatrix(testing$classe, gbmSimpleTest)
print(gbmSimpleTrainConfusion)
print(gbmSimpleTestConfusion)
gbmCasesConfusion <- confusionMatrix(actualAnswer, predict(gbmSimpleModel, cases))
print(gbmCasesConfusion)


save.image()
