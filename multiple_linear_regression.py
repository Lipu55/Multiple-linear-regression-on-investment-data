# MULTIPLE LINEAR REGRESSION
#yesterday we did simple linear regression now this time we add some dimension or we add more independent features
#simple lineare regression we have one independent varaibel but in multiple linear regression weh have more indipendent varaiable
#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#********************************************************************************************************************************

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\kdata\Desktop\KODI WORK\1. NARESH\1. MORNING BATCH\N_Batch -- 8.30AM\2.July\14th,15th\mlr\50_Startups.csv")
#This dataset contains information about 50-startups. R&D spend, amount of money spend on administration, amount of money spend on marketing,state which startup operates & profit of the organisation
#we want o build the model to see lineare dependency b/w all these independendt variables. which feature you have to invest more to get the companys more profit
#X- create matrix of independet variable exclude profit // y - creae a matrix of dependent variable that is ony profit
X = dataset.iloc[:, :-1]
#we created a X matrix belongs to all independent variable, -1 is to remove last columns
y = dataset.iloc[:, 4]

X=pd.get_dummies(X)

# If you execute this there is no more 1st column at all as i told you if you created 7 dummpy variable always take 7-1 consider 
# In our dataset we have only 3 dummy variable, so dummy variable trap we will always consider just 3-1 & we have now 2 dummy variable for this dataset 
# lets compare the dataset with dummy variable & we are done with dummy variable 

#******************************************************************************************************************************

# Splitting the dataset into the Training set and Test set

#our dataset contain 50 observation we will consider 40 observation in training test & 10 observation contain test set
# 20% is into test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling is not required for multiple linear regression, python library is take care of everything
#we are done with data preprocessing part

#************************************************************************************************************************
# Now we will fit the trainig part to build the linear regression model

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
#import the LinearRegression class from sklearn package
regressor = LinearRegression()
#creaet the regressor object for LineareRegression
regressor.fit(X_train, y_train)
#you have tot fit the regressor model in only training set 
#we fit machine using multiple lineare regression to the training set & now we will see how the testing will be perform in multiple linear regression
#next part we will go to the test set

#***********************************************************************************************************************

#if you check the dataset we have 4 independent variable & 1 is depenent variable 
#you cannot visualize or plot the 5 dimensions & it is very hard to represent & you cannot plot the graph for this model
#last step would be prediction of the test result

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#you have to create the vector of prediction is called as y_pred & these all are 10 predicted profits & also  let me open y_test which is actual observation or real profit
#we will compare the profit of real profit is y_test & predicted profit is y_pred which build by model 
#we almoset get accurate prediction & we can say that our model did quite good job & we build the multilinear regression model
#next part we will see that which variable is highest impact or very good corelation to the dependent variable

#**********************************************************************************************************************
#We build the multiple linear regression model and fit to the training set & do you think we made the actual model the dataset which we have hear
#when we build the model we used all independent variables & what if from all independent variable which is the one is highly statistic significant or corelate with dependent variable
#we will see which independent variable is highly impact on the profit part out of 4 independent variabel so that investor or company can invest money on that feature only
#we will find out which is more statistically significance & which are not & you have to keep removing the non-statistically significance feature from our dataset 
#let me start the backward elimination 

# BACKWARD ELIMINATION

#Building the optimal model using Backward elimination

import statsmodels.formula.api as sm
#for statistical model we will import the stats model.formula.api & keep a shortcut name as sm
#when you import the stats model you dont have to calculate anything & all statistical calculation done ny machine only
#if you check our multiple linerae equation we have B0 is constant but if you check our dataset we dont have any constant hear, thats why we will add 1 as constant for everyrow
#if  you check X then we have 2 variable along with R&d spent,marketing spent & admin spent
#we will add 1 as constant in our matrix of feature & we will consider as X0 as 1, x0 is constant hear which is 1
#if i add 1 then we will get our correct multiple linear equation & next step we will add 1 to all 50 observation
# Y = X0 + B1X1 + B2X2 + B3X3 + B4X4 ( Y = PROFIT, X0 IS CONSTANT where we will add 1s to all 50 observation, X1 - r &d, x2 - marketing , x3 - admin) now our equation is correct
# we will use append function to add all 1s in all 50 observation

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 
#we will take only independent variable & hear X is independent variable
#this is maths concept & you should know how to write this concept to python code
#first argument is append withe all numpy & if you check ctrl+i then it give you hint to go next option
#second argument you have to covert to integer datatype. you will add matrix of 50 lines with append 1, after add with append function you have to add the datatype or else you might encounter with erro
#3rd argument is to find where yo haave to add 1 in row or column, if in row then you have to right axis =o, column then you have to add axis = 1
#before you execute this line please check X once and after execute check so that new value 1 will add as constant
#we prepared backward elimination algorithm by adding constant in first column to stay in the equation called as X_opt which contains only independent variable

#**********************************************************************************************************************

# we will start now the backward elimination
#lets initialise all the independent variable only at first & remove one-by-one independent variable which is not statistically significance
#X_opt = X[:, [0,1,2,3,4,5]]
#lets create new matrix of feature or create an object of independent variable which is optimal matrix of feature 
#we will take all the index of the all independent variable , 0-constant,1-dummy variabel1, 2-dummy variable 2, 3-r&d,4-marketing, 5-admin, thats why we assigned 5
#first step is to select a significance level of 0.005, if p-value of independent value is above the significance level then you have to remove or else if p-value of independence varibale is below then that varaiable is stay in the model

#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#second step is to fit all the possisble predictor we will have library for that 
#we will create new object called OLS (ordinary least square) & we call this class from stats model for same we assigned short cut name as 'sm'
#endog - dependent variabe & exog which you created for X_opt for independent variable
#next step is to fit the ordinary least square algorithm in multi linear regression
#we did the 2nd step - fit the full model with all possbile predictor, now step-2 done
#next step we will check the p-value & Significance level to verify that either that INdep variable is stay in model or remove one-by-one
#lets look for predictor with highest p-value using stats model library called summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
#regressor_OLS is the object which we created earlier now if we add . summary() then we will get the p-value information
#we got the table can you check please first thing i explained you in model summary table in stats
#if rsquare>adj r-square then your model is perfect model,we will discuss again about r-square & adjusted r-square
#now lets check about the p-value  & if lower p-value then more SL, as per the table we have independent variabe - const,x1,x2,x3,x4,x5 - 0th index,1st index,2nd index,3rd index,4th index, 5th index
#constant (1), x1 - Dummy1, x2 - dummy2, x3- r&d, x4 - marketing, x5 - admin
#As per the table which have highest p-value  is 0.991 is x3 variable, as per condition P-value>SL then you have to remove the variable
#we have to remove the p-value which has highest p-value which is>0.005 , so we will remove X2
#we found that 2 nd index has highest pvalue, so we have to remove that by remove 2nd index
#0,1,2,3,4,5 ---> (remove 2nd index - 0.990) ---> New (X_opt would be -- 0,1,3,4,5)
# keep doing until p-vaue is less then SL 
X_opt = X[:, [0,1,2,3,5]]
#removed 2nd index which was X2 - 2nd index - dummy 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#again fit the regressor_OLS 
regressor_OLS.summary()
#after execute you got other table as which has higest p-value is x1- dummy1, which has 1st index , removed 1 st index

X_opt = X[:, [0,1,2,3]]
#removed 1st index which has X2 - 1st index - dummy1
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#as per the new table we found higest p-value x2 > 0.05, you have to remove x2 
#so far we removed 1st & 2nd index which is dummy variable , now we will remove 4th index cuz of highest p-value

X_opt = X[:, [0,1,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#after remove everything as per the condition p-value>SL & we get the final independent variable those are const,r&d , admin
#out of these 2 independnt variable lets look at the highest p-value whcih is marketing this is greate then 0.05 which index this has 5th indexing

X_opt = X[:, [0,1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#we generate final table where we have only 1 independent variable is called R&d spend 
#this R&D spend is more impact an profit part so finaly being datascientist you have to explain that you have to spend more on R&D spend
#this is end of the multiple linear regression
