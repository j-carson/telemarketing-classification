File name | Purpose
:---|:---
Data\_clean.ipynb | Data cleaning and modifications from the raw data set
Data_loader.ipynb | Load the CSV file to POSTGRES
EDA.ipynb | Exploratory Data Analysis
GB_models.ipynb | Gradient Boosting models
LR_models.ipynb | Logistic Regression models
Model.ipynb | Minimum viable project model
RF_models.ipynb | Random forest models
Test\_result\_database.ipynb | Notebook to test the test result database
data\_pipeline.py | Main data pipeline
load\_save.py | Loading and saving pickle files and creating a row in the test result database with training results. Because it uses the ipython-sql extension, this file is loaded to notebooks with %load rather than import.
test\_index.ipynb | The database part of the test results database. Also a little clean up function.
