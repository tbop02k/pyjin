import pandas as pd
import numpy as np
   
def get_numerice_cols(
    df
):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']    
    return df.select_dtypes(include=numerics)

def make_lags(
    df,
    features,
    lags:list):
    
    for c in features:
        for i in lags:
            df = pd.concat(
                [df, df[c].shift(i).rename(c+'_lag_{}'.format(i))],
                axis=1)   
            
    return df             
            
def label_shift(
    df,
    y_col,
    shift_len=1    
):
    
    df = pd.concat([df, df[y_col].shift(-shift_len).rename('label')] ,axis=1)
    df = df.drop(label, axis=1)
    return df
    

def train_test_split(
    df,
    label,
    test_size = 0.2
):
    
    len_df=len(df)
    test_idx = int(len_df * test_size)
    
    df_train = df.iloc[:-test_idx] 
    df_test = df.iloc[-test_idx:]

    X_train, y_train = df_train.loc[:, df.columns.difference([label])], df_train[label]
    X_test, y_test = df_test.loc[:, df.columns.difference([label])], df_test[label]    

    return df_train, df_test, X_train, X_test, y_train, y_test