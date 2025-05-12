# UW ID: 20873217
# Name: Lakshmi Narayanan, Krish
# UW email: klnna@uwaterloo.ca
# Kaggle_public_score: 0.29589
# Kaggle_submission_count: 30
# Running_time: 0.36
# the running time is the time in seconds to run your FinalModel(), see evaluation.R for more details


# fill in the above ID, name, and time, but leave other existing comments untouched
# Only libraries and FinalModel function allowed in this file
# NO setwd, load, and write.csv commands allowed in this file
# add library here using the library function
# all libraries need to be used in your code in the library::function format
### Your libraries start here
# library
library(tidyverse)
library(xgboost)
### Your libraries end here

# dtrain: data.frame for the training set
# dtest: data.frame for the test set
# should return a data.frame for prediction
FinalModel <- function(dtrain, dtest){
# Write all your code below, including preprocessing and functions.
# Only the final model should be fitted, no model selection/tuning steps allowed here
# All plotting, diagnositic and model building/selection steps should be in your Rmd file.
### Your code starts here

    # replace the following lines with your prediction model
	# please do not specify the number of trees for the random forest, which will leave it as the default of 500
    
    new_dtrain <- dtrain
    # Convert the 'date' column to Date format
    new_dtrain$date <- as.Date(new_dtrain$date)
    
    # Extract date features
    new_dtrain$year <- year(new_dtrain$date)
    new_dtrain$month <- month(new_dtrain$date)
    new_dtrain$day <- day(new_dtrain$date)
    new_dtrain$day_of_week <- as.factor(wday(new_dtrain$date, label = FALSE))
    new_dtrain$is_weekend <- as.factor(ifelse(new_dtrain$day_of_week %in% 
                                                c(1, 7), 1, 0)) # Weekend flag
    
    # Cyclic encoding for month (to capture periodicity)
    new_dtrain$month_sin <- sin(2 * pi * new_dtrain$month / 12)
    new_dtrain$month_cos <- cos(2 * pi * new_dtrain$month / 12)
    
    # Convert all character columns to factors
    new_dtrain <- new_dtrain %>%
      mutate(across(where(is.character), as.factor))
    
    # Creating new features
    new_dtrain$npc <- new_dtrain$takeaway - new_dtrain$giveaway
    new_dtrain$pm_per_min <- new_dtrain$pm/new_dtrain$toi
    new_dtrain$ppg <- new_dtrain$pts/new_dtrain$gp
    new_dtrain$physicality <- new_dtrain$hits + 0.5*new_dtrain$pim
    new_dtrain$relCorsi_per_toi <- new_dtrain$relCorsi/new_dtrain$toi
    
    # Identify outliers with standardized residuals > ±3
    outliers <- c(26, 121, 170, 202, 228)
    # Removing outliers
    new_dtrain <- new_dtrain[-outliers,]
    
    # Dropping variates
    new_dtrain$date <- NULL
    new_dtrain$month <- NULL
    new_dtrain$takeaway <- NULL 
    new_dtrain$giveaway <- NULL
    new_dtrain$pm <- NULL
    new_dtrain$pts <- NULL
    new_dtrain$goal <- NULL
    new_dtrain$hits <- NULL
    new_dtrain$pim <- NULL
    new_dtrain$relCorsi <- NULL
    new_dtrain$day_of_week <- NULL
    
    # Creating test dataset
    new_dtest <- dtest
    
    new_dtest$salary <- 1
    
    # Convert the 'date' column to Date format
    new_dtest$date <- as.Date(new_dtest$date)
    
    # Extract date features
    new_dtest$year <- year(new_dtest$date)
    new_dtest$month <- month(new_dtest$date)
    new_dtest$day <- day(new_dtest$date)
    new_dtest$day_of_week <- as.factor(wday(new_dtest$date, label = FALSE)) # Sunday = 1, Saturday = 7
    new_dtest$is_weekend <- as.factor(ifelse(new_dtest$day_of_week %in% c(1, 7), 1, 0)) # Weekend flag
    
    # Cyclic encoding for month (to capture periodicity)
    new_dtest$month_sin <- sin(2 * pi * new_dtest$month / 12)
    new_dtest$month_cos <- cos(2 * pi * new_dtest$month / 12)
    
    # Convert all character columns to factors
    new_dtest <- new_dtest %>%
      mutate(across(where(is.character), as.factor))
    
    # Creating new features
    new_dtest$npc <- new_dtest$takeaway - new_dtest$giveaway
    new_dtest$pm_per_min <- new_dtest$pm/new_dtest$toi
    new_dtest$ppg <- new_dtest$pts/new_dtest$gp
    new_dtest$physicality <- new_dtest$hits + 0.5*new_dtest$pim
    new_dtest$relCorsi_per_toi <- new_dtest$relCorsi/new_dtest$toi
    
    # Dropping columns
    new_dtest$date <- NULL
    new_dtest$month <- NULL
    new_dtest$takeaway <- NULL 
    new_dtest$giveaway <- NULL
    new_dtest$pm <- NULL
    new_dtest$pts <- NULL
    new_dtest$goal <- NULL
    new_dtest$hits <- NULL
    new_dtest$pim <- NULL
    new_dtest$relCorsi <- NULL
    new_dtest$day_of_week <- NULL
    
    # Formula
    new_formula_full <- "log(salary)~team+toi+gp+pos+nat+age+injure+
                    day+is_weekend+month_sin+month_cos+npc+
                    pm_per_min+ppg+physicality+relCorsi_per_toi"
    
    set.seed(20873217)
    new_train_matrix <- model.matrix(as.formula(new_formula_full), 
                                     data = new_dtrain)[,-1]  # Remove intercept
    new_train_labels <- log(new_dtrain$salary)
    
    new_test_matrix <- model.matrix(as.formula(new_formula_full), 
                                    data = new_dtest)[,-1]
    
    # Handle any missing features (set to 0)
    missing_cols <- setdiff(colnames(new_train_matrix), colnames(new_test_matrix))
    new_test_matrix <- cbind(new_test_matrix, matrix(0, nrow = nrow(new_test_matrix), 
                                                     ncol = length(missing_cols),
                                                     dimnames = list(NULL, missing_cols)))
    
    # Ensure column order matches training data
    new_test_matrix <- new_test_matrix[, colnames(new_train_matrix)]
    
    new_chosen_features <- c("toi","ppg","gp","physicality","pm_per_min",
                             "age","injure","relCorsi_per_toi","month_sin",
                             "day","npc","month_cos","natRUS",
                             "teamVegas Golden Knights","teamNashville Predators",
                             "posRW/LW","teamCalgary Flames","posD","is_weekend1",
                             "natUSA","teamDetroit Red Wings","teamBuffalo Sabres",
                             "teamOttawa Senators","natDEU","teamPhiladelphia Flyers",
                             "teamNew York Islanders","teamVancouver Canucks",
                             "teamEdmonton Oilers","teamNew York Rangers",
                             "teamChicago Blackhawks","teamLos Angeles Kings",
                             "posC/LW","posRW","teamWashington Capitals",
                             "teamFlorida Panthers","posLW/RW","natSVK",
                             "natCAN","teamPittsburgh Penguins",
                             "teamMontreal Canadiens","natCHE","posLW",
                             "posRW/C","natCZE","teamColorado Avalanche",
                             "teamArizona Coyotes","teamMinnesota Wild",
                             "teamTampa Bay Lightning","teamSt. Louis Blues")
    
    # Prepare data for XGBoost
    new_xgb.dtrain <- xgb.DMatrix(data = new_train_matrix[, new_chosen_features] , 
                                  label = new_train_labels)
    new_xgb.dtest <- xgb.DMatrix(data = new_test_matrix[, new_chosen_features])
    
    params <- list(objective = "reg:squarederror",
                   eval_metric = "rmse",
                   eta = 0.1,
                   max_depth = 3,
                   gamma = 0,
                   subsample = 1,
                   colsample_bytree = 1,
                   min_child_weight = 4)
    
    new_final_model <- xgb.train(
      params = params,
      data = new_xgb.dtrain,
      nrounds = 200
    )
    
    pred <- exp(predict(new_final_model, newdata=new_xgb.dtest))
    res <- data.frame(Id=dtest$Id, salary=pred)
    return(res)
	
### Your code ends here
} # end FinalModel
