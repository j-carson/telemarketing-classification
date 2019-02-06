import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.preprocessing import StandardScaler


def sort_columns_by_type(columns):
    """
    sort_columns_by_type(columns):
    inputs: the columns in the model
    returns two lists:  categorical_columns, numerical_columns
    so they can be scaled/dummied separately
    """

    categorical_cols = [
        'job',
        'marital',
        'education',
        'in_default',
        'housing',
        'loan',
        'contact',
        'month',
        'day_of_week',
        'poutcome'
    ]

    numerical_cols = [
        'age',
        'campaign',
        'pdays',
        'previous',
        'emp_var_rate',
        'cons_price_idx',
        'cons_conf_idx',
        'euribor3m',
        'nr_employed'
    ]
    
    ignore_cols = [
        'success',
        'bank_addl_id'
    ]

    selected_numerical = []
    selected_categorical = []
    
    for c in columns:
        if c in numerical_cols:
            selected_numerical += [c]
        elif c in categorical_cols:
            selected_categorical += [c]
        elif c not in ignore_cols:
            raise Exception('Unknown column', c)
        
    return selected_categorical, selected_numerical



def resample_data(X, y, strategy):
    """resample_data(X, y, strategy):
          X = training features for telemarketing project
          y = training target for telemarketing project
          strategy = one of 'ros' - RandomOverSampler
                      'smote' - SMOTENC             
                      ''      - No resampling 
        returns:
            X_resampled, y_resampled
    """
    if strategy == "ros":
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif strategy == "smote":
        c,n = sort_columns_by_type(X.columns)
        smote = SMOTENC(c, random_state=123)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    elif strategy == "":
        X_resampled = X
        y_resampled = y 
        
    else:
        raise Exception("resample_data: Unknown strategy")

    return X_resampled, y_resampled



def read_prepare_data(columns, strategy):
    """read_data(columns, strategy)
            columns = Columns to include in the model, must include 
                    index bank_addl_id and target feature success
            strategy = Resampling strategy ('ros', 'smote') or empty string
                    for no resampling 
                    
        returns:
            X_train, X_test, y_train, y_test
            
        data read in, resampled, scaled and one-hot encoded
    """
    database = "postgresql://localhost/bankcalls"
    
    sql_test  = "SELECT " + ", ".join(columns) + " FROM test_group;"
    df_test =  pd.read_sql(sql_test, database, index_col='bank_addl_id')
    
    sql_train = "SELECT " + ", ".join(columns) + " FROM train_group;"
    df_train = pd.read_sql(sql_train, database, index_col='bank_addl_id')

    categorical, numerical = sort_columns_by_type(columns)
    
    for c in categorical:
        df_test[c]  = df_test[c].astype('category')
        df_train[c] = df_train[c].astype('category')
        
    y_test = df_test['success'].replace({'yes':1, 'no':0})
    X_test = df_test.drop(columns='success')
    
    y_train = df_train['success'].replace({'yes':1, 'no':0})
    X_train = df_train.drop(columns='success')
    
    X_train, y_train = resample_data(X_train, y_train, strategy)
    
    standard = StandardScaler()
    X_train_scaled = standard.fit_transform(X_train[numerical])
    X_test_scaled = standard.transform(X_test[numerical])
    
    X_train_scaled = pd.DataFrame(data=X_train_scaled, 
                                 index=X_train.index,
                                 columns=numerical)
    X_test_scaled = pd.DataFrame(data=X_test_scaled,
                                index=X_test.index,
                                columns=numerical)
    
    X_train_dummies = pd.get_dummies(X_train[categorical], 
                                     drop_first=True)
    X_test_dummies = pd.get_dummies(X_test[categorical], 
                                     drop_first=True)
    
    X_train_prepped = pd.concat([X_train_scaled, X_train_dummies],
                               axis=1, ignore_index=False)
    X_test_prepped = pd.concat([X_test_scaled, X_test_dummies],
                               axis=1, ignore_index=False)
    
    return X_train_prepped, X_test_prepped, y_train, y_test
        
    
    
    
