import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# I've been all over this in the debugger and combining SMOTENC and 
# ColumnTransformer seems to lose some type information, but it really
# is the float columns that are being passed to StandardScaler
import warnings
from sklearn import exceptions
warnings.filterwarnings("ignore", category=exceptions.DataConversionWarning)

from pdb import set_trace as BREAKPOINT

def sort_columns_by_type(columns):
    """
    sort_columns_by_type(columns):
    inputs: the columns in the model
    indices: returns column indices rather than column names (for smotenc)
    returns three lists:  categorical_columns, numerical_columns, power_columns
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
    
    # --- columns I'm sending to PowerTransformer
    power_cols = [
        'campaign',
        'previous',
        'age'
    ]

    # -- columns that go to StandardScaler
    numerical_cols = [
        'pdays',  # -- We still need a data cleaning strategy for this guy! 
        'emp_var_rate',
        'cons_price_idx',
        'cons_conf_idx',
        'euribor3m',
        'nr_employed'
    ]
    
    # these show up in the sql query but are not predictors
    ignore_cols = [
        'success',
        'bank_addl_id'
    ]

    selected_numerical = []
    selected_categorical = []
    selected_power = []
    
    for co in columns:
        if co in numerical_cols:
            selected_numerical += [co]     
        elif co in power_cols:
            selected_power += [co]           
        elif co in categorical_cols:
            selected_categorical += [co]
        elif test:
            pass
        elif co not in ignore_cols: 
            raise Exception('Unknown column', co)
            
    # -- returns are alphabetical    
    return selected_categorical, selected_numerical, selected_power



def read_data(columns):   
    # -- Read the data -- # 
    database = "postgresql://localhost/bankcalls" 
    sql_test  = "SELECT " + ", ".join(columns) + " FROM test_view;"
    sql_train = "SELECT " + ", ".join(columns) + " FROM train_view;"
    
    df_test =  pd.read_sql(sql_test, database, index_col='bank_addl_id')
    df_train = pd.read_sql(sql_train, database, index_col='bank_addl_id')
    
    # -- Set success column to 0/1 and separate X's and y's -- #      
    y_test = df_test['success'].replace({'yes':1, 'no':0})
    y_train = df_train['success'].replace({'yes':1, 'no':0})

    X_test = df_test.drop(columns='success')
    X_train = df_train.drop(columns='success')
    
    # --
    # organize columns by type for ColumnTransformer convenience
    # categoricals last: making dummies adds columns and if they're all 
    # on the end the numerical columns don't move to new indices
    
    cats, nums, pows = sort_columns_by_type(X_train.columns)
    X_train = pd.concat([X_train[nums], X_train[pows], X_train[cats]], axis=1)
    X_test =  pd.concat([X_test[nums], X_test[pows], X_test[cats]], axis=1)

    X_train[cats] = X_train[cats].astype('category')
    X_test[cats] = X_test[cats].astype('category')
    
    return X_train, X_test, y_train, y_test
        
  

def run_a_model(model_name, model, columns, strategy, param_candidates):
    
    # -- get the data  -- #
    X_train, X_test, y_train, y_test = read_data(columns)
    
    X = pd.concat([X_train, X_test], axis=0)
    X_cols = list(X.columns)
    
    categorical, numerical, power = sort_columns_by_type(X_cols)
    cat_idx = [ X_cols.index(co) for co in categorical ]
    num_idx = [ X_cols.index(co) for co in numerical ]
    pwr_idx = [ X_cols.index(co) for co in power ]
    
    transformers = []
    if len(numerical) > 0: 
        std = StandardScaler()
        transformers += [ ('std', std, num_idx)]
    if len(power) > 0:
        pwt = PowerTransformer()
        transformers += [ ('pwr', pwt, pwr_idx)]
        
    if len(categorical) > 0: 
        # -- the unique()[1:] grabs the unique values then drops the first
        # -- the astype(str) is because I used pandas astype categorical in the 
        #    read_data step, this changes it back to string for sklearn
        drop_first = {}
        for i,co in enumerate( categorical ):
            drop_first[i] = X[co].unique()[1:].astype(str)
    
        one_hot = OneHotEncoder(categories=drop_first, 
                         handle_unknown='ignore',
                           sparse=False)
        transformers += [ ('oneh', one_hot, cat_idx)]
    
    col_xform = ColumnTransformer(transformers)
    
    # -- 
    # resampling needs to be added to the model with the 
    # imblearn version of make_pipeline
    # -- 
    if strategy == "ros":
        ros = RandomOverSampler(random_state=123)
        pipeline_model = imb_make_pipeline(ros, col_xform, model)
        
    elif strategy == "smote":
        # -- Use the NC for non-continuous data! -- #
        smote = SMOTENC(cat_idx, random_state=123)
        pipeline_model = imb_make_pipeline(smote, col_xform, model)
        
    elif strategy == "down":
        down = RandomUnderSampler(random_state=123)
        pipeline_model = imb_make_pipeline(down, col_xform, model)
    
    elif strategy == "":
        # -- Could use sklearn pipeline since there's no sampling here
        pipeline_model = imb_make_pipeline(col_xform, model) 
        
    else:
        raise Exception("run_a_model: unknown imbalance strategy")
            
    # -- run the model -- #
    if param_candidates == None:
        
        scores = cross_validate(pipeline_model, X_train, y_train, 
                               scoring='f1', 
                               cv=4,
                               n_jobs=-1,
                               return_estimator=True,
                               return_train_score=False)
    
        best = np.argmax(scores['test_score'])
        best_model  = scores['estimator'][best]
        best_params = None
        all_cv = scores
    else:
        grid = GridSearchCV(pipeline_model, param_candidates,
                            scoring='f1',
                            cv=4, 
                            n_jobs=-1,
                            iid=False, 
                            return_train_score=False)
        grid.fit(X_train, y_train)
        
        best_model  = grid.best_estimator_
        best_params = grid.best_params_
        all_cv      = grid.cv_results_
    
    
    # -- remove the resample from the steps for final testing
    best_model = imb_make_pipeline(best_model.steps[-2][1], best_model.steps[-1][1])
    
    y_predict = best_model.predict(X_train)
    
    # -- calculate all the stats -- #
    accuracy  = metrics.accuracy_score(y_train, y_predict)
    recall    = metrics.recall_score(y_train, y_predict)
    precision = metrics.precision_score(y_train, y_predict)
    f1        = metrics.f1_score(y_train, y_predict)
    cm        = metrics.confusion_matrix(y_train, y_predict)
    
    probs = pd.DataFrame(best_model.predict_proba(X_train))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, probs.loc[:,1])
                                           
    auc = metrics.auc(fpr, tpr)
    
    # -- and again for the holdout set -- #
    y_holdout = best_model.predict(X_test)
    
    test_accuracy  = metrics.accuracy_score(y_test, y_holdout)
    test_recall    = metrics.recall_score(y_test, y_holdout)
    test_precision = metrics.precision_score(y_test, y_holdout)
    test_f1        = metrics.f1_score(y_test, y_holdout)
    test_cm        = metrics.confusion_matrix(y_test, y_holdout)
    
    test_probs = pd.DataFrame(best_model.predict_proba(X_test))
    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(y_test, test_probs.loc[:,1])
                                           
    test_auc = metrics.auc(test_fpr, test_tpr)
    
    holdout = dict(accuracy=test_accuracy, 
                   recall=test_recall,
                   precision=test_precision,
                   f1=test_f1,
                   cm=test_cm,
                   probs=test_probs,
                   fpr=test_fpr,
                   tpr=test_tpr,
                   thresholds=test_thresholds,
                   auc=test_auc)
        
    
    # -- 
    # get final column names after one hot encoding for
    # understanding logistic regression results 
    # -- 
    columns = X_cols
    if len(categorical) > 0:
        fake_cols = []
        for k in range(len(drop_first.keys())):
            for f in drop_first[k]:
                name = categorical[k] + '_' + f
                fake_cols += [name]
            columns.remove(categorical[k])
        columns += fake_cols
                
    # -- store the results in dictionary -- #
    measures = dict(name=model_name, 
                    model=best_model, 
                    best_params=best_params,
                    columns=columns, 
                    strategy=strategy,
                    accuracy=accuracy, 
                    recall=recall, 
                    precision=precision, 
                    f1=f1,
                    auc=auc, 
                    cm=cm,
                    all_cv=all_cv,
                    holdout=holdout)                                         
                                       
    return measures
    
    
    
        
        
        
    
    
    
    