# http://hamelg.blogspot.in/2016/09/kaggle-home-price-prediction-tutorial.html

# Set working directory
setwd('/media/satya/backupdata/Hackathons/certace')

# -------------------------------------------Exploratory Data Analysis------------------------------------------ #

#
# Differing factor levels could cause problems with predictive modeling later on so we need to resolve these issues before going 
# further. We can make sure the train and test sets have the same factor levels by loading each data set again without converting 
# strings to factors, combining them into one large data set, converting strings to factors for the combined data set and then
# separating them. Let's change any NA values we find in the character data to a new level called "missing" while we're at it:

# Load the dataset
train <- read.csv("/media/satya/backupdata/Hackathons/certace/housing_price/train.csv", stringsAsFactors = FALSE)
test <- read.csv("/media/satya/backupdata/Hackathons/certace/housing_price/test.csv", stringsAsFactors = FALSE)

# Check the dimension of test and training set
dim(train)
dim(test)

# structure of train dataset
str(train)

# structure of test datset
str(test)

# Remove the target variable not found in test set from train set
SalePrice = train$SalePrice 
train$SalePrice = NULL

# Combine data sets
full_data = rbind(train,test)

typeof(full_data) 
typeof(full_data[,"Id"])
typeof(full_data[,"Id"])

# colnames(full_data)
# Convert character columns to factor, filling NA values with "missing"
for (col in colnames(full_data)){
  if (typeof(full_data[,col]) == "character"){
    new_col = full_data[,col]
    new_col[is.na(new_col)] = "missing"
    full_data[col] = as.factor(new_col)
  }
}

# Separate out our train and test sets
train = full_data[1:nrow(train),]
train$SalePrice = SalePrice  
test = full_data[(nrow(train)+1):nrow(full_data),]

# Now the factor levels should be identical across both data sets. Let's continue our exploration by looking at a summary of the 
# training data.
summary(train)

# Summary output gives us a basic sense of the distribution of each variable, but it also reveals another issue: 
# some of the numeric columns contain NA values. None of the numeric variables contain negative values so encoding the NA's 
# as a negative number is a simple way to convert them to numeric while making it easy to tell which values are actually NA. 
# We will be using a tree-based model in this example so the scale of our numbers shouldn't affect our model and assigning the 
# NA's to -1 will essentially allow -1 to act as a numeric flag for NA values. If we were using a model that scales numeric 
# variables by a learned parameter like linear regression, we might want to use a different solution such as imputing missing values 
# and we'd also want to consider centering, scaling and normalizing the numeric features so that they are on the same scale and have 
# distributions that are roughly normal.

# Fill remaining NA values with -1
train[is.na(train)] = -1
test[is.na(test)] = -1

# Now the data should be clean with no NA values, so we can start exploring how home features affect home sales prices. 
# For one, it could be useful to know whether any of the variables are highly correlated with SalePrice. 
# Let's determine whether any variables have a correlation with SalePrice with an absolute value of 0.5 or higher:
for (col in colnames(train)){
  if(is.numeric(train[,col])){
    if( abs(cor(train[,col],train$SalePrice)) > 0.5){
      print(col)
      print( cor(train[,col],train$SalePrice) )
    }
  }
}

# The output shows a handful of variables have relatively strong correlations with sale price, with "OverallQual" being the 
# highest at 0.7909816. These variables are likely important for predicting sale prices. Now let's investigate some which 
# numeric variables have low correlations with sales prices:

for (col in colnames(train)){
  if(is.numeric(train[,col])){
    if( abs(cor(train[,col],train$SalePrice)) < 0.1){
      print(col)
      print( cor(train[,col],train$SalePrice) )
    }
  }
}

# The year and month sold don't appear to have much of a connection to sales prices. Interestingly, "overall condition" doesn't 
# have a strong correlation to sales price, while "overall quality" had the strongest correlation

# Next, let's determine whether any of the numeric variables are highly correlated with one another.
sapply(train, is.numeric) # show the columns names value as TRUE if numeric, otherwise false
typeof(sapply(train, is.numeric))

train[ , sapply(train, is.numeric)] # return the train dataset having only numeric column data
typeof(train[ , sapply(train, is.numeric)])

cors = cor(train[ , sapply(train, is.numeric)])

high_cor = which(abs(cors) > 0.6 & (abs(cors) < 1))

rows = rownames(cors)[((high_cor-1) %/% 38)+1]
cols = colnames(cors)[ifelse(high_cor %% 38 == 0, 38, high_cor %% 38)]
vals = cors[high_cor]

cor_data = data.frame(cols=cols, rows=rows, correlation=vals)
cor_data

# # Note that since the table above was constructed from a symmetric correlation matrix, each pair appears twice.
# # The table shows that 11 variables have correlations above 0.6, leaving out the target variable SalesPrice. 
# The highest correlation is between GarageCars and GarageArea, which makes sense because we'd expect a garage that can park more 
# cars to have more area. Highly correlated variables can cause problems with certain types of predictive models but since no 
# variable pairs have a correlations above 0.9 and we will be using a tree-based model, let's keep them all.

# Now let's explore the distributions of the numeric variables with density plots. This can help us get identify outlines 
# and whether different variable and our target variable are roughly normal, skewed or exhibit other oddities.

for (col in colnames(train)){
  if(is.numeric(train[,col])){
    plot(density(train[,col]), main=col)
  }
}

# There are too many variables to discuss all the plots in detail but, a quick glance reveals that many of the numeric 
# variables show right skew. Also, many variables have significant density near zero, indicating certain features are only 
# present in subset of homes. It also appears that far more homes sell in the spring and summer months than winter. 
# Lastly, the target variable SalePrice appears roughly normal, but it has tail that goes off to the right, so a handful of 
# homes sell for significantly more than the average. Making accurate predictions for these pricey homes may be the most 
# difficult part of making a good predictive model.

#-----------------------------------------------Predictive Modeling-------------------------------------------------------#

# Before jumping into modeling, we should determine whether we have to alter our data structures get them to work with our 
# model and whether we want to add new features. We will use the XGBoost tree model for this problem. The XGBoost package in 
# R accepts data in a specific numeric matrix format, so if we were to use it directly, we'd have to one-hot encode all of the 
# categorical variables and put the data into a large numeric matrix. To make things easier, we will use R's caret package 
# interface to XGBoost, which will allow us to use our current data unaltered.

# This data set already has a large number of features, so adding more may not do much to improve the model, 
# but upon inspecting the variable text file, I noticed a couple key variables I was expecting to see were missing. 
# Namely, total square footage and total number of bathrooms are common features used to classify homes, 
# but these features are split up into different parts in the data set, such as above grade square footage, basement square 
# footage and so on. Let's add two new features for total square footage and total bathrooms:

# Add variable that combines above grade living area with basement sq footage
train$total_sq_footage = train$GrLivArea + train$TotalBsmtSF
test$total_sq_footage = test$GrLivArea + test$TotalBsmtSF

# Add variable that combines above ground and basement full and half baths
train$total_baths = train$BsmtFullBath + train$FullBath + (0.5 * (train$BsmtHalfBath + train$HalfBath))
test$total_baths = test$BsmtFullBath + test$FullBath + (0.5 * (test$BsmtHalfBath + test$HalfBath))

# Remove Id since it should have no value in prediction
train$Id = NULL    
test$Id = NULL

# Now we are ready to create a predictive model. Let's start by loading in some pacakges:
# install.packages("caret")
# install.packages("plyr")
# install.packages("xgboost")
# install.packages("Metrics")
library(caret)
library(plyr)
library(xgboost)
library(Metrics)

# Next let's create the control object and tuning variable grid we need to pass to our caret model. 
# The target metric used to judge this competition is root mean squared logarithmic error or RMSLE. 
# Caret optimizes root mean squared error for regression by default, so if we want to optimize for RMSLE we should pass 
# in a custom summary function via our caret control object. The R package "Metrics" has a function for computing RMSLE 
# so we can use that to compute the performance metric inside our custom summary function.

# Create custom summary function in proper format for caret
custom_summary = function(data, lev = NULL, model = NULL){
  out = rmsle(data[, "obs"], data[, "pred"])
  names(out) = c("rmsle")
  out
}

# Create control object
control = trainControl(method = "cv",  # Use cross validation
                       number = 5,     # 5-folds
                       summaryFunction = custom_summary                      
)

# Create grid of tuning parameters
grid = expand.grid(nrounds=c(100, 200, 400, 800), # Test 4 values for boosting rounds
                   max_depth= c(4, 6),           # Test 2 values for tree depth
                   eta=c(0.1, 0.05, 0.025),      # Test 3 values for learning rate
                   gamma= c(0.1), 
                   colsample_bytree = c(1), 
                   min_child_weight = c(1),
                   subsample = 0.5)

#Now we can train our model, using the custom metric rmsle:

set.seed(12)

xgb_tree_model =  train(SalePrice~.,      # Predict SalePrice using all features
                        data=train,
                        method="xgbTree",
                        trControl=control, 
                        tuneGrid=grid, 
                        metric="rmsle",     # Use custom performance metric
                        maximize = FALSE)   # Minimize the metric

# Next let's check the results of training and which tuning parameters were selected:
xgb_tree_model$results

xgb_tree_model$bestTune

# In this case, the model with a tree depth of 4, trained for 800 rounds with a learning rate of 0.025 was chosen. 
# According to the table, the cross-validated rmsle for this model was 0.12717267, so we'd expect a score close 0.127 
# if we were to use this model to make predictions on the test set. Before we make predictions let's check which variables 
# ended up being most important to the model:

varImp(xgb_tree_model)

# As expected, the variable with the highest correlation to SalePrice, OverallQual and total_sq_footage was very important to the 
# model. The extra feature for total square footage was also very important and our total bathrooms variable came in a distant third.
# Finally, let's make predictions on the test set using the trained model and submit to Kaggle to see if the actual performance 
# is near our cross validation estimate.

test_predictions = predict(xgb_tree_model, newdata=test)

submission = read.csv("/media/satya/backupdata/Hackathons/certace/housing_price/sample_submission.csv")
submission$SalePrice = test_predictions
write.csv(submission, "/media/satya/backupdata/Hackathons/certace/housing_price/home_prices_xgb_sub1.csv", row.names=FALSE)



