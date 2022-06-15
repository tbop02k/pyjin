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

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


def plot_timeseires(df):
    '''
    df should have datetime index    
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as md
    sns.set(font_scale = 1.6)
    
    fig, ax = plt.subplots(figsize=(20,6))
    g= sns.lineplot(data=df, marker='o')
    g.set(ylim=(0, None))
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(md.MonthLocator(interval = 1))
    ax.set_xlabel('datetime')
    ax.legend(fontsize=16)
    ax.legend(fontsize=16)