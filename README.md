# Project proposal 

Predict results of a bank telemarketing campaign 

## Motivation

Predictive models can help a business invest their marketing time and effort to maximize their return on 
investment. This project will analyze the results of a bank telemarketing campaign to see 
if telephone sales resources could be applied in a more effective fashion.

## Data

The bank marketing dataset from the UCI dataset repository contains approximately 
41000 telemarketing records with customer information, telemarketing contacts, and economic context for a total of 20 predictors versus a categorical outcome variable (subscribed to a term deposit or not). The data is from a Portuguese bank and was collected between May, 2008 and November, 2010. 

https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014


## Target value 

Yes or no, did the telemarketer close the deal?

## Features 

The data set includes 20 features which the author grouped into
the following categories:

- Bank client data

   1 - Age: numeric
   
   2 - Type of job: categorical
   
   3 - Marital status: categorical
   
   4 - Education level: categorical
   
   5 - Has credit in default?: categorical
   
   6 - Has housing loan? categorical
   
   7 - Has personal loan? categorical
   
- Telemarketing contact fields

   8 - Contact type: categorical ('cellular','telephone')
   
   9 - Month: last contact month of year 
   
   10 - Day of week: last contact day of the week. The bank appears to only make calls on weekdays.
   
   11 - Length of call in seconds: numeric 
   
-  Telemarketing campaign fields:
   
   12 - Number of contacts performed during this campaign and for this client: numeric
   
   13 - Number of days since the client was last contacted from a previous campaign: numeric
   
   14 - Previous number of contacts performed before this 
   campaign and for this client: numeric
   
   15 - Outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

-  Social and economic context attributes

   16 - Employment variation rate - quarterly indicator (numeric)
   
   17 - Consumer price index - monthly indicator (numeric)
   
   18 - Consumer confidence index - monthly indicator (numeric)
   
   19 - Euribor 3 month rate - daily indicator (numeric)
   
   20 - Number of employees - quarterly indicator (numeric)

## Potential challenges

- I have not run queries to check for class imbalance, but I suspect that most 
telemarketing campaigns have imbalanced classes with more negatives than sales

- There are calls with length zero. That means the customer was
not reached. If a customer is not reached, then "made a sale" is 
always "no". The dataset writeup warns not to use call duration as a predictor, so I would
actually be predicting, "Which customers buy, assuming you can get them on the phone?" Once
I get the data loaded, I can investigate what proportion of calls have non-zero length. 
