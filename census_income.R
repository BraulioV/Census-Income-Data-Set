## ------------------------------------------------------------------------
leer_datos <- function(fichero = "./Data/adult.data") {
    adult.train <- read.csv(fichero, header=FALSE, col.names = c("age","workclass", 
            "fnlwgt","education","education-num","marital-status","occupation","relationship",
            "race","sex","capital-gain","capital-loss","hours-per-week","country","income"), 
            na.strings = c(" ?", "?", ""), stringsAsFactors = F)
}

adult.train <- leer_datos()
adult.test <- leer_datos(fichero = "./Data/adult.test")

## ------------------------------------------------------------------------
# Añadimos una columna a los datos para indicar 
# cuales pertenecen a datos de train y cuales a
# datos de test

adult.test$trainTest = rep(1,nrow(adult.test))
adult.train$trainTest = rep(0,nrow(adult.train))

# Reconstruimos el conjunto de datos al completo,
# uniendo los datos de train y test

fullSet <- rbind(adult.test,adult.train)

# Cada una de las variables categóricas, pasarán 
# de ser cadenas de caracteres a tener un valor 
# numérico o un factor, con lo que 
# tanto datos de train como datos de test, 
# obtendrán el mismo factor

fullSet$workclass = as.factor(fullSet$workclass)
fullSet$country = as.factor(fullSet$country)
fullSet$education = as.factor(fullSet$education)
fullSet$marital.status = as.factor(fullSet$marital.status)
fullSet$sex = as.factor(fullSet$sex)
fullSet$relationship = as.factor(fullSet$relationship)
fullSet$occupation = as.factor(fullSet$occupation)
fullSet$income = as.factor(fullSet$income)
fullSet$race = as.factor(fullSet$race)

# Reconstruimos los datos de train y test originales

adult.train = data.frame(fullSet[fullSet$trainTest == 0,])
adult.test = data.frame(fullSet[fullSet$trainTest == 1,])

# Y eliminamos la columna auxiliar

adult.test$trainTest = NULL
adult.train$trainTest = NULL

## ------------------------------------------------------------------------
apply(X=adult.train, MARGIN=2, FUN=function(columna) length(is.na(columna)[is.na(columna)==T]))

## ------------------------------------------------------------------------
getRowsNA <- function(datos = adult.train) {
    aux = is.na(datos)*1
    rowsMissingValues = apply(X=aux, MARGIN=1,
             FUN = function(fila) sum(fila))
}

rowsMissingValues.train = getRowsNA()
length(rowsMissingValues.train[rowsMissingValues.train > 0])

## ------------------------------------------------------------------------
adult.train.clean = adult.train[rowsMissingValues.train == 0,]
adult.test.clean = adult.test[getRowsNA(datos = adult.test) == 0,]
fullSet.clean = data.frame(fullSet[getRowsNA(datos = fullSet) == 0,])

## ------------------------------------------------------------------------
levels(adult.train.clean$workclass)

## ------------------------------------------------------------------------
head(c(adult.train.clean$workclass))
head(adult.train.clean$workclass)

## ------------------------------------------------------------------------
adult.train.clean[,"education.num"] = NULL
adult.test.clean[,"education.num"] = NULL
adult.train[,"education.num"] = NULL
adult.test[,"education.num"] = NULL

## ------------------------------------------------------------------------
predictorGLM <- function(model, pintar = T){
  ypred = predict(model, adult.test.clean, type="response")
  ypred[ypred <= 0.5] = ">50K"
  ypred[ypred > 0.5] = "<=50K"

  if(pintar){
    print("Matriz de confusión datos de test")
    print(table(predict=ypred, truth=(adult.test.clean$income)))
  }
  cat("Eout = ",mean((ypred != adult.test.clean$income)*1))
  ypred
}
trainingIndex=which(fullSet.clean$trainTest==0)
fullSet.clean$trainTest = NULL
fullSet.clean$education.num = NULL
set.seed(1)
glmModel = glm(income ~ ., data = fullSet.clean, 
    subset = trainingIndex, family = binomial(logit))
glmPred = predictorGLM(glmModel, pintar=T)

## ------------------------------------------------------------------------
cat("Error obtenido en la clase <=50K: ", 
    0/length(adult.test.clean$income[adult.test.clean$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    370000/length(adult.test.clean$income[adult.test.clean$income==">50K"]))

## ------------------------------------------------------------------------
plot(glmModel, which=c(1))

## ------------------------------------------------------------------------
library(randomForest)

outOfSampleError <- function(model, newdata = adult.test.clean,  printTable = T, ...){
  set.seed(1)
  ypred = predict(object = model, newdata = newdata, ...)
  if(printTable)
    print(table(predict=ypred, truth=newdata$income))
  cat("Eout = ",mean((ypred != newdata$income)*1))
}
set.seed(1)
rf.clean = randomForest(income ~ ., data = adult.train.clean, importance = T)
print(rf.clean)
outOfSampleError(rf.clean)

## ------------------------------------------------------------------------
length(adult.train.clean$income[adult.train.clean$income == "<=50K"])
length(adult.train.clean$income[adult.train.clean$income == ">50K"])

## ------------------------------------------------------------------------
set.seed(1)
rf.tuned = tuneRF(x=subset(adult.train.clean, select=-income), 
      y=adult.train.clean$income, doBest=T)

## ------------------------------------------------------------------------
outOfSampleError(rf.tuned)
plot(rf.clean$predicted, col = c(3,2), main = "Nº de datos predichos para cada clase por Random Forest")
cat("Error obtenido en la clase <=50K: ", 
    2700/length(adult.test.clean$income[adult.test.clean$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    269900/length(adult.test.clean$income[adult.test.clean$income==">50K"]))

## ------------------------------------------------------------------------
length(adult.test.clean$income[adult.test.clean$income == "<=50K"])
length(adult.test.clean$income[adult.test.clean$income == ">50K"])

## ------------------------------------------------------------------------
library(e1071)
set.seed(1)
svmModel = svm(income ~ ., data = adult.train.clean, kernel = "radial")
outOfSampleError(svmModel)

## ------------------------------------------------------------------------
set.seed(1)
svmModelReg_0001 = svm(income ~ ., data = adult.train.clean, kernel = "radial", cost = 0.001)
outOfSampleError(svmModelReg_0001)
set.seed(1)
svmModelReg_001 = svm(income ~ ., data = adult.train.clean, kernel = "radial", cost = 0.01)
outOfSampleError(svmModelReg_001)
set.seed(1)
svmModelReg_01 = svm(income ~ ., data = adult.train.clean, kernel = "radial", cost = 0.1)
outOfSampleError(svmModelReg_01)
set.seed(1)
svmModelReg_09 = svm(income ~ ., data = adult.train.clean, kernel = "radial", cost = 0.9)
outOfSampleError(svmModelReg_09)
set.seed(1)
svmModelReg_25 = svm(income ~ ., data = adult.train.clean, kernel = "radial", cost = 2.5)
outOfSampleError(svmModelReg_25)

## ------------------------------------------------------------------------
plot(svmModelReg_09$fitted, col = c(3,2), main = "Nº de datos predichos para cada clase por SVM")
cat("Error obtenido en la clase <=50K: ", 
    71400/length(adult.test.clean$income[adult.test.clean$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    151300/length(adult.test.clean$income[adult.test.clean$income==">50K"]))

## ------------------------------------------------------------------------
normalizar_maxmin <- function(data=adult.train.clean, numbercols = c("age","fnlwgt","capital.gain",
                            "capital.loss","hours.per.week"), data_test = adult.test.clean) {
    # nos quedamos sólo con las columnas numéricas
    cols_numericas = subset(data, select=numbercols)
    colstest_numericas = subset(data_test, select=numbercols)
    # calculamos el máximo y el mínimo de cada columna de los datos de train
    maxs = apply(X=cols_numericas, MARGIN=2, FUN=max)
    mins = apply(X=cols_numericas, MARGIN=2, FUN=min)
    # aplicamos el escalado a los datos de train
    datos_normalizados_numericos = scale(x=cols_numericas, center=mins, scale=maxs)
    # aplicamos los valores de normalización de train sobre los de test
    test_normalizado = as.data.frame(scale(x = colstest_numericas, 
                    center = attr(datos_normalizados_numericos, "scaled:center"),
                    scale = attr(datos_normalizados_numericos, "scaled:scale")))
    # juntamos los valores normalizados con el resto de columnas
    datos_normalizados_numericos = as.data.frame(datos_normalizados_numericos)
    for (c in numbercols) {
        data[c] = datos_normalizados_numericos[c]
        data_test[c] = test_normalizado[c]
    }
    list(data, data_test)
}

norm = normalizar_maxmin()
adult.train.clean.norm = norm[[1]]
adult.test.clean.norm = norm[[2]]
norm = NULL

## ---- message=FALSE, warning=FALSE---------------------------------------
library(kknn)
set.seed(1)
best_model_knn <- train.kknn(formula = income ~ ., data = adult.train.clean.norm, 
  kmax = 2*ncol(adult.train), kernel = c("gaussian", "inversion"))
best_model_knn

## ------------------------------------------------------------------------
set.seed(1)
model_knn <- kknn(formula = income ~ ., train = adult.train.clean.norm, 
    test = adult.test.clean.norm, k = best_model_knn$best.parameters$k, 
    kernel = best_model_knn$best.parameters$kernel)
print(table(predict = model_knn$fitted.values, truth = adult.test.clean$income))
cat("Eout = ",mean((model_knn$fitted.values != adult.test.clean$income)*1))

## ------------------------------------------------------------------------
plot(model_knn$fitted.values, col = c(3,2), 
     main = "Nº de datos predichos para cada clase por KNN")
cat("Error obtenido en la clase <=50K: ", 
    94700/length(adult.test.clean$income[adult.test.clean$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    154100/length(adult.test.clean$income[adult.test.clean$income==">50K"]))

## ------------------------------------------------------------------------
library(nnet)
set.seed(1)
model.nnet = nnet(formula = income ~ ., maxit=10000, 
    data = adult.train.clean.norm, size = 10, decay=0.1,trace=F)
outOfSampleError(model.nnet, adult.test.clean.norm, type="class")

## ------------------------------------------------------------------------
cat("Error obtenido en la clase <=50K: ", 
    94700/length(adult.test.clean$income[adult.test.clean$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    154100/length(adult.test.clean$income[adult.test.clean$income==">50K"]))

## ------------------------------------------------------------------------
library(ROCR)

getPerfomance <- function(model, newdata = adult.test.clean, svmPred = F, 
                          nnetOrGlm=F, calculatePred = T,...){
    set.seed(1)
    if(calculatePred) {
        if(!nnetOrGlm && !svmPred)
            preds = predict(object = model, newdata = newdata, ...)[,2]
        else if (!svmPred && nnetOrGlm)
            preds = predict(object = model, newdata = newdata, ...)
        else{
            preds = attributes(predict(model, newdata, decision.values=T))$decision.values
            preds = preds*-1
        }
    } else
        preds = as.numeric(model)
    
    pred = prediction(preds, newdata$income)
    # tpr --> True Positive Rate
    # fpr --> False Positive Rate
    performance(pred, "tpr", "fpr")
}
plot(getPerfomance(model = glmModel, type="response", nnetOrGlm = T),
     col=2,lwd=2,main="Curvas ROC Para los distintos modelos estudiados") 
plot(getPerfomance(model = rf.tuned, type = "prob"),col=3,lwd=2,add=T)
plot(getPerfomance(model = svmModelReg_09, svmPred = T),lwd=2,col=4,add=T)
plot(getPerfomance(model_knn$prob[,2], newdata = adult.test.clean.norm, 
                   calculatePred = F),lwd=2,col=5,add=T)
plot(getPerfomance(model = model.nnet, newdata = adult.test.clean.norm,
  type="raw", nnetOrGlm = T),lwd=2,col=6,add=T)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottomright",col=c(2:6),lwd=2,legend=c("Regresión logística",
            "Random Forest","SVM","KNN","Red Neuronal"),bty='n')

## ------------------------------------------------------------------------
set.seed(1)
# normalizamos todos los datos de la clase
norm = normalizar_maxmin(data = adult.train, data_test = adult.test)
adult.train.norm = norm[[1]]
adult.test.norm = norm[[2]]
norm = NULL

# debemos quitar las variables con NA para poder predecir correctamente
predice_na <- function(attr, noselectvars, formula, test_dataset = adult.test.norm,
                       train_dataset = adult.train.norm, dataset = adult.train.clean.norm) {
    # el vector no selectvars siempre tendrá dos variables string
    model <- nnet(formula = formula, maxit = 100,
                  data = subset(dataset, select=c(-which(colnames(dataset) == noselectvars[1]),
                                                  -which(colnames(dataset) == noselectvars[2]))),
                  size = 8, decay = 1, trace = F)
    pred_train <- predict(object = model, newdata = subset(train_dataset[is.na(train_dataset[,attr]),],
                         select=c(-which(colnames(train_dataset) == noselectvars[1]),
                                  -which(colnames(train_dataset) == noselectvars[2]))), type = "class")
    pred_test <- predict(object = model, newdata = subset(test_dataset[is.na(test_dataset[,attr]),],
                         select=-which(colnames(test_dataset) == noselectvars)), type = "class")
    list(pred_train, pred_test)
}

workclass = predice_na(attr = "workclass", noselectvars = c("country", "occupation"), 
                       formula = workclass ~ .)
adult.train$workclass[is.na(adult.train$workclass)] = workclass[[1]]
adult.train.norm$workclass[is.na(adult.train.norm$workclass)] = workclass[[1]]
adult.test$workclass[is.na(adult.test$workclass)] = workclass[[2]]
adult.test.norm$workclass[is.na(adult.test.norm$workclass)] = workclass[[2]]

occupation = predice_na(attr = "occupation", noselectvars = c("country", "workclass"),
                        formula = occupation ~ .)
adult.train$occupation[is.na(adult.train$occupation)] = occupation[[1]]
adult.train.norm$occupation[is.na(adult.train.norm$occupation)] = occupation[[1]]
adult.test$occupation[is.na(adult.test$occupation)] = occupation[[2]]
adult.test.norm$occupation[is.na(adult.test.norm$occupation)] = occupation[[2]]

country = predice_na(attr = "country", noselectvars = c("occupation", "workclass"),
                     formula = country ~ .) 
adult.train$country[is.na(adult.train$country)] = country[[1]]
adult.train.norm$country[is.na(adult.train.norm$country)] = country[[1]]
adult.test$country[is.na(adult.test$country)] = country[[2]]
adult.test.norm$country[is.na(adult.test.norm$country)] = country[[2]]

## ------------------------------------------------------------------------
apply(X=adult.train, MARGIN=2, FUN=function(columna) length(is.na(columna)[is.na(columna)==T]))
apply(X=adult.test, MARGIN=2, FUN=function(columna) length(is.na(columna)[is.na(columna)==T]))

## ------------------------------------------------------------------------
library(nnet)
set.seed(1)
model.nnet = nnet(formula = income ~ ., maxit=10000, 
    data = adult.train.norm, size = 10, decay=0.1,trace=F)
outOfSampleError(model.nnet, adult.test.norm, type="class")

## ------------------------------------------------------------------------
cat("Error obtenido en la clase <=50K: ", 
    97200/length(adult.test$income[adult.test$income=="<=50K"]))
cat("Error obtenido en la clase >50K: ", 
    143800/length(adult.test$income[adult.test$income==">50K"]))

