rm(list=ls())
# first set working directory to the data location, for example, setwd("/Users/DonaldDuck/Stat444/"), then read in the data by
load("final.Rdata")
# final.Rdata contains dtrain and dtest

# load user-defined FinalModel function, use UW_ID 20654321 as an example
source("20873217.R")
# call the function
tmp <- system.time(res <- FinalModel(dtrain, dtest))
tmp
# report the elapsed time as your running time in source file


# The grading team will compare the result to the solution to compute RMLSE
# for example, assume sol is the data.frame containing the true salary in the same order as in res
load("solution_fake.Rdata")
sqrt(mean((log(res$salary)-log(sol_fake$salary))^2))
# the line above shows how the RMLSE is calculated
# for obvious reasons, the solution provided here is not the real one and is only for the ilustration purpose

# you can save the result for submission to Kaggle, for example
# write.csv(res, file="mysolution.csv", row.names=FALSE)