################################################################################################################################################
#In Search of Return Predictability: Application of Machine Learning Algorithms in Tactical Allocation
################################################################################################################################################
#written by Kris Boudt (Ghent University, Vrije Universiteit Brussel/Amsterdam),
#Muzafer Cela (Vrije Universiteit Brussel; corresponding author; muzafer.cela@vub.be) and Majeed Simaan (Stevens Institute of Technology).

#The following script contains the implementation of a tactical asset allocation strategy as presented in the book chapter.
#The strategy consists of taking a daily position in both a risky (S&P 500 ETF) -and riskless (7-10 years US threasury bonds) asset proportional 
#to a Machine learning algorithm output. The remaining of the script is organised as follows. The section "the data" contains all codes to 
#download all needed input features for the analysis and arrange it in a proper format. In the section "Machine Learning Algorithms " we will use 
#the input features to form an output probability for the analyzed machine learning algorithms (ML) over the different seeds. 
#An average over the different seeds will be formed in the section "Results" as well as weighted versions of the outputs and Faber's 
#tactical asset allocation strategy. The performance of the presented strategy will be analyzed form both a financial and statistical perspective
#in the section "Performance". As last a slightly adapted version of the LIME framework will be presented in the section "LIME". 
#For further details on the implementation, we refer to the book chapter.
#Note that the code was developed under the following system.
#R version 3.5.0 (2018-04-23)
#Platform: x86_64-w64-mingw32/x64 (64-bit)
#Running under: Windows >= 8 x64 (build 9200)

# install the packages with the used version
install.packages("https://cran.r-project.org/src/contrib/Archive/caret/caret_6.0-80.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/quantmod/quantmod_0.4-13.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/lubridate/lubridate_1.7.4.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/PerformanceAnalytics/PerformanceAnalytics_1.5.2.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/xts/xts_0.10-2.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/pROC/pROC_1.13.0.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/iml/iml_0.9.0.tar.gz", repos=NULL)
install.packages("https://cran.r-project.org/src/contrib/Archive/rstudioapi/rstudioapi_0.9.0.tar.gz", repos=NULL)

library(caret) 
library(quantmod) 
library(lubridate) 
library(PerformanceAnalytics)
library(xts) 
library(pROC) 
library(iml) 
library(rstudioapi)

#set the working directory to the file location
current_path <- getActiveDocumentContext()$path
wd <- paste(strsplit(as.character(current_path),"/")[[1]][1:(length(strsplit(as.character(current_path),"/")[[1]])-1)], collapse="/")
setwd(wd)

################################################################################################################################################
# the Data
################################################################################################################################################
#download the data
t1 <- "1990-01-01"  #starting date 
t2 <- "2019-01-31"  #ending date
v <- c("SPY","^VIX","GLD","IEF","XLF")

P.list <- lapply(v, function(sym) get(getSymbols(sym,from = t1, to= t2)) ) #download the data
P.list5 <- lapply(P.list, function(x) x[,5])  #extract the volumes
P.list6 <- lapply(P.list, function(x) x[,6])  #extract the adjusted price

P5 <- na.omit(Reduce(function(...) merge(...),P.list5 )) #merge the volumes together
P6 <- na.omit(Reduce(function(...) merge(...),P.list6 )) #merge the adjusted close price together 

names(P5) <- names(P6) <- c("SPY","VIX","GLD","IEF","XLF") #assign names to the adjusted close price
names(P5) <- paste(names(P5),"vol",sep = "_")
P5$VIX_vol <- NULL
R6 <- P6/lag(P6)-1

# add rolling difference 
R6_roll <- R6 - rollapply(R6,25,mean)
names(R6_roll) <- paste(names(R6_roll),"_roll",sep="")
R <- na.omit(merge(R6,R6_roll,P5)) #merge the prices, volume and rolling prices and - volumes

#create the response variable
tau <- -0.01 # threshold value for assigning instances to a class
R$CHANGE_next <- 1
R$CHANGE_next[which(lag(R$SPY,-1) < tau )] <- -1

#convert to dataframe
ds <- data.frame(date = date(R),R)
rownames(ds) <- NULL

# the response variable Change next must be a factor so that it can be predicted with the caret package 
ds$CHANGE_next <- factor(ds$CHANGE_next,labels = c("dn","up"))
features <- names(ds)[!names(ds) %in% c("date","CHANGE_next")]

rm(list=setdiff(ls(), c("ds", "features", "P.list6")))

################################################################################################################################################
# Machine Learning Algorithms 
################################################################################################################################################
#This section contains the code that will output the out of sample probabilities of a downward movement of the SPY ETF. The main function 
# is constituted of 3 loops. The first one performs the mapping of the input features to the target variable on a rolling basis. The second 
# performs the first analysis over the different models. The last one performs the analysis of the different models over different seeds.

#vector containing the model names (caret compatible models can be tested as well)
#models <-  c("rf","C5.0","mlp","knn","glm","glmnet","lda","svmRadial")
models <-  c("glm","lda")
#includes all the months of the whole period
month <- date(unique(floor_date(ds$date,"month")))
M <- 12 #m is the amount of months used for training 

#generate different seeds
am <- 2 #am is the amount of seeds that are taken into consideration
ss <- combn(c(1:8),4, simplify = T)
seeds <- as.numeric(apply(ss, 2, function(x)paste(x, collapse = "")))

for (ii in c(1:am)) {
  #loop for all the seeds
  myseed <- seeds[ii]
  MIT <- data.frame() #the dataframe Will contain the mean squared error for the in sample period on a rolling basis for all models
  DS_PR <- data.frame() #the dataframe will contain all the out of sample predictions on a rolling basis for all models
  
  for(mo in models){
    #loop for all models 
    ds_predict <- data.frame() # this dataframe will contain all out of sample predictions  
    mit <- c()
    
    for(m in M:(length(month)-2)) {
      #prepare the data to be analyzed
      # training period is constituted of 12 months prior to the prediction month
      train.month <- month[(m-M+1):(m+1)]
      train.index <- which((ds$date > train.month[1]) & (ds$date < train.month[M+1]))
      test.month <-  month[m+1:2]
      test.index <- which((ds$date > test.month[1]) & (ds$date < test.month[2]))
      DS <- ds[train.index,-1]
      ######################################################
      # ML ALGO
      # Run algorithms using 10-fold cross validation with 3 repeats
      trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs= TRUE, seeds= set.seed(myseed))
      set.seed(myseed)
      rfDefault <- train(CHANGE_next~., data=DS, method= mo, metric=c("Accuracy"), preProc=c("range"), trControl=trainControl)
      ######################################################
      #make predictions for the testing period
      DS_predict <- as.data.frame(predict(rfDefault, ds[test.index,-1], type = "prob"))
      names(DS_predict) <- c("up","dn")
      DS_predict$date <- ds[test.index,"date"]
      ds_predict <- rbind(ds_predict,DS_predict)
      ######################################################
      #make predictions for the training period to derive the mean squared error
      DS_predict_tr <- as.data.frame(predict(rfDefault, ds[train.index,-1], type = "prob"))
      names(DS_predict_tr) <- c("up","dn")
      DS_predict_tr$date <- as.Date(ds[train.index,"date"]) #add the dates to the predictions
      updnr <- cbind(ifelse(DS$CHANGE_next == "up", c(1),c(0) ),ifelse(DS$CHANGE_next == "dn", c(1),c(0) ))
      mit[(m+1)-M] <- sum((updnr[,1]-DS_predict_tr$up)^2)
      ######################################################
      print(m)
    }
    if(mo== models[1]){ 
      DS_PR <-  cbind.data.frame(as.Date(ds_predict$date),ds_predict$dn)
      MIT <- mit
    }else{
      DS_PR <- cbind(DS_PR, ds_predict$dn )
      MIT <- cbind(MIT, mit)
    }
    print(mo)
  }
  colnames(DS_PR) <- c("Date", models)
  write.csv(DS_PR, paste("DS_PR",myseed, ".csv", sep = ''))
  write.csv(MIT, paste("MIT",myseed, ".csv", sep = ''))
}
rm(list=setdiff(ls(), c("ds", "features", "P.list6", "models", "am","seeds", "month","M")))

################################################################################################################################################
# Results
################################################################################################################################################
#The following section contains the code to process the output probabilities of the analysed ML- algorithms over the different seeds. 
#This output probabilities will then be averaged over the different seeds in the matrix mydatam. Based on the output probabilities of the different
#ML- algorithms two additional predictions will be formed as weighted versions of the existing outputs. The first one will assign weights to 
#each individual output probabilities proportional to the in sample mean squared error. The second one will be a simple average of all different 
#outputs which corresponds to equally weighted output. At last, an output based on Faber's tactical asset allocation strategy will be formed.
#The latter is formed by analyzing the simple moving average over the last 200 days with daily price. A buy signal of the SPY ETF is formed 
#whenever the daily price exceeds the simple moving average over the last 200 days. In the opposite case a position in the IEF ETF will be taken.
#Note that Faber's strategy consists of doing a monthly rebalancing of the taken positions.
############################################################################################################
#average the output of the different seeds
############################################################################################################
#load the output in an array
my.names.pr <- paste('DS_PR',seeds,'.csv', sep = '') #create a vector of all prediction names
mydata <- lapply(my.names.pr[1:am], (read.csv), header= TRUE, dec = '.', sep = ',') #load the data
mydataa <- array(NA, dim = c(nrow(mydata[[1]]), ncol(mydata[[1]]),am)) #create an array that will contain all the data
mydataa[,,1:am]<- sapply(mydata,function(x)(as.matrix(x)), simplify = T) # place the data in the created array
mydatam <-  apply(mydataa[,-c(1,2),1:am],c(1,2), function(x)mean(as.numeric(x))) #calculate mean of each ML over the seeds
mydatam <- xts(mydatam, order.by = as.Date( mydataa[,2,1])) #convert the matrix in a xts

#create the final data frame
final <- as.data.frame(xts(matrix(0,nrow = nrow(mydatam),ncol = (length(models)+5)),order.by = as.Date( mydataa[,2,1]) ))
final[,c(1:2)] <- na.locf(ds[which(as.Date(ds$date) %in% as.Date(mydataa[,2,1]) == TRUE),c("SPY", "IEF")]) #place SPY and IEF into the final dataframe
colnames(final) <- c("SPY", "IEF", models, "weighted","average", "faber" )
final[,3:(2+ncol(mydatam))] <- mydatam

############################################################################################################
#calculate the weighted function
############################################################################################################
my.names.mit <- paste('MIT',seeds,'.csv', sep = '')
mydatamit <- lapply(my.names.mit[1:am], (read.csv), header= TRUE, dec = '.', sep = ',')
mydatamita <- array(NA, dim = c(nrow(mydatamit[[1]]), ncol(mydatamit[[1]]),am)) #create an array that will contain all the data
mydatamita[,,1:am]<- sapply(mydatamit,function(x)(as.matrix(x)), simplify = T) # place the data in the created array
MIT <-  apply(mydatamita[,-1,1:am],c(1,2), function(x)mean(as.numeric(x))) #calculate the average error
wit <- apply(MIT,c(1), function(x)( 1/x/sum((1/x)))) #calculate the weights from the errors
wit <- xts(t(wit), order.by = as.Date(month[(M):(length(month)-2)]))
test <- cbind.xts(mydatam, wit)
test[,c((ncol(mydatam)+1):ncol(test))] <- na.locf(test[,c((ncol(mydatam)+1):ncol(test))]) #Use the weigths for the comming month
test <- test[-which(is.na(test[,1])),]
weighted <- apply((test[,c(1:ncol(mydatam))]*test[,c((ncol(mydatam)+1):ncol(test))]), 1,sum) #weighted represents the output of the weighted function
final[,(3+ncol(mydatam))] <- weighted

############################################################################################################
#calculate the equally weighted function
############################################################################################################
wm <- (apply(mydatam,1, mean)) # calculate the average of the mydatam
final[,(4+ncol(mydatam))] <- wm

############################################################################################################
#Faber
############################################################################################################
SPYroll <- na.omit(rollapply(P.list6[[1]],200,mean)) #calculate the simple moving average over the last 200 days
SPYroll$buy <- as.numeric(P.list6[[1]] > SPYroll) # convert the SMA into a to be taken position
SPYroll$buy <- lag(SPYroll$buy) #avoid lookahead bias
SPYroll$bm <- SPYroll$buy[(month),] # extract the monthly to be taken position
SPYroll$bm <- na.locf(SPYroll$bm) #keep the same position for one month
SPYroll <- SPYroll[rownames(final),]
SPYroll$Faber <-  ifelse(SPYroll$bm==1,0,1) #convert probability of going up to probability down
final[,(5+ncol(mydatam))] <- SPYroll$Faber

rm(list=setdiff(ls(), c("ds", "features", "P.list6", "models", "am","final")))

################################################################################################################################################
# Performance
################################################################################################################################################
# In this section all previously formed outputs will be tested in the asset allocation framework as presented in the book chapter In Search of 
#Return Predictability: Application of Machine Learning Algorithms in Tactical Allocation. The analysis of the results will be performed on 
# both financial- and statistical basis. Note that the all results are highly dependent on the level of taken risks(= threshold a). Therefore, 
#four different threshold values will be analyzed namely 0.5, 0.85, 0.9, 0.95.
############################################################################################################
# Financial performance
############################################################################################################
#those are returns without sma of the probabilities
returns <- matrix(0, nrow= nrow(final), ncol= (ncol(final)-2 ))
returns <- xts(returns, order.by = as.Date(rownames(final[,0])))
colnames(returns) <- colnames(final)[-c(1,2)]

TC_f <- function(TC,i) { # creat the function outputs the annualized returns of a ML algorithm with transaction costs included 
  R_i <- returns[-1,i] - dummyfinal.ch[,i]*TC
  return(Return.annualized(R_i))
}

for(a in c(0.5, seq(0.85,.95,0.05))){
  a <- 0.5
  dummyfinal <- matrix(NA, nrow= nrow(final), ncol= (ncol(final)-2))# dummyfinal is a matrix that will contain all the changes in the positions that are taken
  i <- 3:ncol(final); dummyfinal <- ifelse(((1-final[,i])> a), 1,0) # display the amount of times the output probability exceeds the risk aversion threshold
  ii <- 1: ncol(dummyfinal); dummyfinal.ch <- abs(diff(dummyfinal[,ii])) # extract the amount of changes in the taken postions  
  amount.tr <- apply(dummyfinal.ch, 2, sum) # compute a sum of the changes in the taken positions 
  amount.tr <- round(amount.tr/14, 3) # annualize the changes in the taken positions 
  
  for(i in c(3:ncol(final))){
    returns[,i-2] <- ((((1-final[,i]) >= a)*(((1-final[,i])*final[,1])+((final[,i])*final[,2])))+(((1-final[,i]) < a)* final[,2]))
  }
  anreport <- table.AnnualizedReturns(returns)
  sk <- skewness(returns)
  ku <- kurtosis(returns)
  drisk <- table.DownsideRisk(returns, MAR =0.01/252 )
  
  # solve for TC that makes it equal to the benchmark
  TC <- c()
  for(i in 1:10){
    TC[i] <- ifelse(is.numeric(try(uniroot(function(TC) TC_f(TC,c(i)) - TC_f(TC,11) ,c(-0.1,0.1))$root ,outFile = print("Error")))== T, 
                    uniroot(function(TC) TC_f(TC,c(i)) - TC_f(TC,11) ,c(-0.1,0.1))$root, NA)
    #Note that the ifelse(try()) part of the code has been added to avoid that the loop would stop if the sollution 
    #do not exist
    print(i)
  }
  TC <- TC*100 #display the results in %
  
  final_caret <- rbind(anreport, sk, ku, drisk, amount.tr, c(TC, NA))# Note that the maximum transaction cost equivalent can take any value for the benchmark itself leading to (TC, NA) 
  plot(cumsum(returns), type = 'l')
  print(a)
  write.csv(final_caret, paste("final",a*100, ".csv", sep = ""))
}

############################################################################################################
# Statistical performance
############################################################################################################

for( a in c(0.5,0.85, 0.9, 0.95)){
  acckappaauc <- matrix(NA, nrow = length(c(3:ncol(final))), ncol = 5)
  colnames(acckappaauc) <- c("Accuracy","Kappa","Sensitivity", "Specificity", "AUC")
  rownames(acckappaauc) <- colnames(final)[-c(1,2)]
  i=1
  for(ii in c(3:ncol(final))){
    pred <- xts(ifelse((1-final[,ii])>a ,"up","dn"), order.by = as.Date(rownames(final)))
    chnext <- xts(ds$CHANGE_next, order.by = as.Date(ds$date))
    rocdata <- cbind.xts(pred,chnext)
    rocdata <- rocdata[-which(is.na(rocdata[,1])),]
    rocdata[,1:2] <- as.factor(rocdata[,1:2]) #convert to factor
    rocdata  <- cbind.data.frame(factor(rocdata[,1] ,c(2,1),labels = c("1","-1")),
                                 factor(rocdata[,2] ,c(2,1),labels = c("1","-1"))) #assign specific values to the factors
    res <- confusionMatrix(table(rocdata[,1],rocdata[,2])) #form a confusion matrix
    au <-ifelse(strsplit(as.character(try(auc(as.numeric(rocdata[,1]), as.numeric(rocdata[,2]))))," ")[[1]][1]=="Error", c(NA), 
                auc(as.numeric(rocdata[,1]), as.numeric(rocdata[,2]))) #calculate the area under the curve with if then else stucture in case of error
    acckappaauc[i,] <- c(res$overall[1:2],res$byClass[1:2],au) #form the vector containing all the statistics
    i=i+1
  }
}

################################################################################################################################################
# LIME
################################################################################################################################################
#This section contains the code to implement the LIME framework. The code starts by standardizing the input features manually, train a model and 
#the threshold probability to a given level. The presented model with the given threshold probability corresponds to the best performing model under
#the best performing conditions. A lime object is then created to perform the linear approximation (with a generalized linear model) of the 
#instances in the testing data. All indivuidual coefficients will then be averaged to form the final explanation of the analyzed model. Note that 
#the same analysis can be performed for different models under various threshold probabilities.

#pre-process the data
idx <- createDataPartition(ds$SPY, p = 0.9, list = FALSE, times = 1) #split the training and testing data
dssc <- scale(ds[,-c(1,16)],scale = T, center =T)  #center and scale the data
dssc <- cbind.data.frame(dssc, as.factor(ds$CHANGE_next)) #bind the target variable with the standardized inputs
colnames(dssc) <- c(features,"CHANGE_next")
ds_tr <- dssc[ idx,]
ds_te  <- dssc[-idx,]

#train a model
set.seed(1234)
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs= TRUE, seeds= set.seed(1234))
rfDefault <- train(CHANGE_next~., data=ds_tr, method= models[3], metric=c("Accuracy"), trControl=trainControl) #train the model

#choose  the threshold probability for assignment to a class
rfDefault$modelInfo$predict <-
  function(modelFit, newdata, submodels = NULL) {
    out <- predict(modelFit, newdata)
    if(modelFit$problemType == "Classification")
    {
      out <- modelFit$obsLevels[ifelse(out[,2]>0.9,2,1 )]
    } else out <- out[,1]
    out
  }

#create a lime object
predictor.rf <- Predictor$new(
  model = rfDefault, 
  data = as.data.frame(ds_tr[,features]), 
  y = ds_tr$CHANGE_next
)

#average the results over the testing period
for(i in 1:nrow(ds_te)){
  lime_explain <- LocalModel$new(predictor.rf , x.interest = ds_te[i,features], k=13)$results[, c(1,2,3)]
  if(i == 1){
    tot <- lime_explain$results
  }else{
    tot[,c(1,2,3)] <- (tot[,c(1,2,3)]) + (lime_explain$results[,c(1,2,3)])
    tot[,4] <- as.numeric(tot[,4]) + as.numeric(lime_explain$results[,4])
  }
  if(i== nrow(ds_te)){
    tot[,c(1,2,3,4)] <- tot[,c(1,2,3,4)]/nrow(ds_te)
  }
  print(i)
}

#form an average input feature for the testing period 
value <- as.data.frame(apply(ds_te[,features],2, mean)) #calculate the mean of the input features
value <- value[which((rownames(value) %in%  tot$feature[1:(length(features)-1)]) == TRUE),] #keep the input features held by the model

#prepare the data to plac it in the lime object
tot$x.recoded <- tot$x.original <- c(value, value) 
tot$effect <- as.numeric(tot$beta*tot$x.recoded)
tot$feature.value <- as.character(paste(tot$feature, "=" ,round(tot$x.recoded,3 ) ,sep = ""))
tot[,4:7] <- as.character(tot[,4:7])

#place the data inside of the object
lime_explain$x.interest <- as.data.frame(apply(ds_te[,features],2, mean))
lime_explain$results <- tot
plot(lime_explain)
