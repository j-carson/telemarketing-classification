# Project Summary

Predict results of a bank telemarketing campaign 

## Design

### Need for study
By performing fewer and more effective telemarketing phone calls, businesses
save money, and people who are not interested in the product offering are less likely to receive an annoying phone call.

### Data
The bank marketing dataset from the UCI dataset repository contains approximately 
41000 telemarketing records with a total of 20 predictors versus a categorical outcome variable (subscribed to a term deposit or not). The data is from a Portuguese bank and was collected between May, 2008 and November, 2010. 

[Dataset link.](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Exploratory Data Analysis

### The target 
The target feature is whether or not the phone call was successful, and the customer subscribed to the term deposit.

The dataset is unbalanced. For the time period in question, the telemarketing department had an overall 11 percent success rate.

### About the features

The data set includes 20 features which the author grouped into
the following categories:

- Bank client data

Information about the clients has been anonymized, so this information is categorical.`For example, whether there is a loan, not how much the loan is for. 

There were no features that were obvious strong predictors for
the overall dataset, however there were some small subsets 
of 
customers that seemed to be likely prospects. For example, in the customer job category, students and retired customers were more likely to purchase than the general population. However, these were less than 5 percent of the population each. 

The most promising customer categories were: certain values for: job, education, and whether the bank had a cell phone number for the customer versus a land line.

- Telemarketing contact fields

Telemarketing contact fields included information about 
the month and day of week, whether the customer has converted
before, and how many phone calls the customer has received for the
current campaign. The data does not include the year or the time of the phone call, so no time series analysis is 
possible. In addition, there is no customer id field. If the same customer was 
contacted in 2008, 2009, and 2010 there is no way to match up the records.

The most promising field here was month. Demand for term
deposits in Portugal appears to be strongest in March and December. However, no rows are tagged January or February and the greatest number of calls are in April. The sample size
for each month are very inconsistent. September had the third highest conversion rate, but the smallest number of phone calls.
Survey irregularity may be affecting the quality of this
predictor.


-  Economic context attributes

Includes information about what was happening in the Portuguese economy
at the time of the call, including interest rates, unemployment data, and consumer confidence.

These seemed to be the most promising predictors: The box-plots of these fields by category seemed to show that the 
range of values and the median for the 'yes' and 'no' response subsets
were significantly different. 

However, three of the economic indicators (Euribor 3 month interest rate, employment level, and unemployment rate) were highly correlated. Only the interest rate was included in the logisitic
classifier because that model is sensitive to collinear terms. All three
predictors were used in the more robust ensemble learning models.


### Data cleaning

Only some minor data cleaning was done: 

The number of calls per campaign and the number
of previous campaigns were trimmed to 
prevent models from chasing outliers. 

- Changing the calls per campaign to a maximum
of 6 affected 2406 rows and included a long tail
of call counts from 7 to 56. (When you put a range of numbers, mostly zeroes and ones with a max of 56 into standard scaler, the 56 gets even larger
because it is so far from the mean!)

- Changing the previous campaign count to a maximum of 3 affected 94 rows and trimmed a long tail
that went out to 7.

In addition, the call duration was dropped from 
the feature set. If the operator was on the phone longer, that was more likely to be a successful
sale. However, this is not actionable information as there is no way to know the length of
the call before it is made.

The pdays count was also dropped from the feature
set. The pdays feature measures how many days elapsed since
the previous campaign in which the customer
was included. 39,700 of the 41188 rows were 999 (for no information or not included). 
Trying to impute a numerical field with so much
missing data was not reasonable.  The poutcome categorical column
gave information about whether the previous campaign
was successful and included a 'nonexistent' value for those customers who were not included
in a previous campaign. This is basically a categorical representation
of the information from the dropped column.

Categorical variables were all converted to 'one hot/ drop first' features with the ```OneHotEncoder``` in scikit learn. 
The positive count features were scaled with the ```PowerTransformer``` to make them more 'normalize' and the remaining numerical 
features were transformed with the ```StandardScaler.``` (Scaled 
features were not strictly necessary for the ensemble classifiers, but
I don't think there is harm in using the data consistently.)

##  Algorithms

### Dealing with unbalanced data

The ``imblearn``` package was used to deal with unbalanced data.
I used random over sampling and SMOTENC (for categorical data) in the 
final models. I also used random undersampling for some model development
so that prototype models completed quickly.

I also used the ```imblearn``` version of a pipeline, which works like
the ```sklearn``` pipeline except that it can handle resampling as a 
step in a pipeline. 


### Choosing features 

Features were charted with pandas, matplotlib, and seaborn to see 
what looked promising for model development. 

I tested three combinations of features: 

- The set of features that seemed most promising during exploratory data analysis.
- The subset of features dealing with seasonal and economic context. Because the EDA only showed three customer columns of any promise, I wanted to see what would happen if I tried to train on no customer data at all.
- The full set of usable, cleaned features (see above). 


### Choosing models

I started with ```LogisticRegression``` because the model was similar
to the linear regression models done in the last project which made it seem
a bit easier to get started with. In retrospect, I probably should have
dived right in with a more complex classifier. With my remaining time,
I worked with the ```sklearn GraidentBoostingClassifier``` and the ```RandomForestClassifier.``` There are a number of better boosting 
classifiers out there, but with my limited time I focused on the ```sklearn``` package where it was easy to swap different
model objects in and out. 

For this particular data set, I was able to get similar results from 
all three models. 

### Choosing metrics

I was able to get good accuracy and area under the curve figures from
a limited 'minimum viable project' attempt, so it was clear that a different
metric would have to be used to score the final models. I chose the 
F1 metric, which balances precision and recall. It was important for the 
model to find the customers that were good prospects, but because 
employees would be phoning all of the customers labeled as likely to
buy, it was also important
that the final call list have as high of a ratio of true positives to 
false positives as possible.



## Tools

In addition to the packages listed above, I used POSTGRES to store
the model data, and to keep a database of pkl files and the training 
stats for reference. 

Most of my SQL was written with the ```ipython-sql``` notebook extension. 


## Results


I was able to train and test a total of 24 different models with different
combinations of resampling strategy, model type, and column sets.

I was able to get similarly performing best models with any of the three
column sets. In other words, I could make a model with no customer data
at all perform as well as a model with the full feature set. Two of the three best performing models were random forest 
and the third was a gradient boosting model. Ensemble models are able 
to let the non-useful predictors just wash out. 

The features do not have a lot of predictive power,
regardless of the approach taken. I did not
try to create a best customer profile for my hypothetical bank. Instead,
my advice is for the bank to adapt its marketing plans to the 
current market conditions. 

## Conclusions

- The use of additional, lower quality features in a model can be a waste of time. I could throw in additional features to my random forest models without getting better model results. Since models with fewer columns are less complex and can run faster, exploratory data analysis can save time. 

- The use of SMOTENC versus RandomOverSampling
was a wash for this data set. I did notice some differences in that
the SMOTE models were less likely to overfit, but the same model with
different oversampling methods tended to do about the same with the 
holdout data even if the results on the training data indicated more
of a difference.

- The data was heavily edited to redact both personal customer
and bank competitive information, and to reflect the choices of 
the author of the original study, and my analysis probably
suffers from some garbage-in, garbage-out. A dataset that
was created by actual banking data science department may find more useful predictors in their customer data.

## Source code

[Github link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)




