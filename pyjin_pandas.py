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
    lags:list,
    drop=False):
    
    for c in features:
        for i in lags:
            df = pd.concat(
                [df, df[c].shift(i).rename(c+'_lag_{}'.format(i))],
                axis=1)   
        
        if drop:
            df= df.drop(c, axis=1)
            
    return df             
            
def label_shift(
    df,
    y_col,
    shift_len=1    
):
    
    df = pd.concat([df, df[y_col].shift(-shift_len).rename('label')] ,axis=1)
    df = df.drop(y_col, axis=1)
    return df
    

def train_test_split(
    df,
    label,
    test_size = 0.2
):
    '''
    this is pyjin vesion train_test_split function, diffrerent from scikit-learn etc.
    
    '''
    
    len_df=len(df)
    test_idx = int(len_df * test_size)
    
    df_train = df.iloc[:-test_idx] 
    df_test = df.iloc[-test_idx:]

    X_train, y_train = df_train.loc[:, df.columns.difference([label])], df_train[label]
    X_test, y_test = df_test.loc[:, df.columns.difference([label])], df_test[label]    

    return df_train, df_test, X_train, X_test, y_train, y_test


def resample_weekly_mean(
    df, 
    datetimecol = 'ds',
    weekend=False
    ):
    
    df['dayofweek'] = pd.to_datetime(df[datetimecol]).dt.dayofweek
    
    if not weekend:
        temp =  df[
            (df['dayofweek'] != 5) &
            (df['dayofweek'] != 6)
        ]

    grouped = temp.groupby(pd.Grouper(key=datetimecol, freq='W'))
    temp = grouped['yhat'].mean().rename('yhat_mean').to_frame()
    temp['y_mean'] = grouped['y'].mean()
    # temp['dayofweek']= temp.index.dayofweek    
    
    return temp

def make_window_stats_column(
    df : pd.DataFrame, 
    col: str,
    window_size: int,
    lag=1
    ):
        
    temp = df[col].rolling(window=window_size)
    rolling_mean_temp = temp.mean().shift(lag).rename(f'{col}_rolling_mean_{window_size}')
    rolling_std_temp = temp.std().shift(lag).rename(f'{col}_rolling_std_{window_size}')

    return rolling_mean_temp, rolling_std_temp
