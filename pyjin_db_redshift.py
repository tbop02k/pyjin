import pandas as pd
import numpy as np
from pandas.io.sql import _wrap_result
from common.util.df import reduce_mem_usage
# import psycopg2
import redshift_connector      

'''
ver1.0

pandas ver 0.25 or later
'''

def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction = (start_mem - end_mem) / start_mem
    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)
    return df

            
def connectDB(host, 
              user, 
              password, 
              db):         
    
    redshift_connector.paramstyle = 'named'   
        
    return  redshift_connector.connect(
        host = host,
        user = user,
        database = db,
        password = password      
    )

# def connectDB(host, 
#               user, 
#               password, 
#               db):             
        
#     return  psycopg2.connect(
#         host = host,
#         user = user,
#         database = db,
#         password = password      
#     )


# def conn_exec_close(
#     connectInfo, 
#     query, 
#     output=None, 
#     is_return=True, 
#     **kwargs):  
    
#     chunksize = 2000
#     result = 0
    
#     with connectDB(**connectInfo) as conn:
#         with conn.cursor() as cs:
#             cs.itersize = chunksize
#             cs.execute(query)
#             columns = []
#             while rows := cs.fetchmany(chunksize):
#                 columns = [col_desc[0] for col_desc in cs.description] if len(columns) == 0 else columns
#                 df = _wrap_result(rows, columns)
#                 df = reduce_mem_usage(df)
#                 result.append(df)
                

#             if check_zero and len(result) == 0:
#                     raise Warning(f"[main read] row_count: 0, sql: {sql}")
#             elif not check_zero and len(result) == 0:
#                 print(f"[main read row_cont: 0, sql: {sql}")

#             result = pd.concat(result).reset_index(drop=1) if len(result) > 0 else pd.DataFrame()
#             return result


def conn_exec_close(connectInfo, query, output=None, is_return=True, **kwargs):  

    with connectDB(**connectInfo) as conn_DB:
        with conn_DB.cursor() as cursor:
            try:                
                cursor.execute(query, kwargs)
                columns = [
                    col_desc[0].decode('UTF-8') for col_desc in cursor.description]
                
                if is_return:
                    if output is None:
                        return cursor.fetchall(), columns
                    elif output=='df':
                        df_res = pd.DataFrame(cursor.fetchall(), columns=columns)
                        df_res = reduce_mem_usage(df_res)
                        return df_res
                else:
                    return True
                
            except ResourceWarning:
                return True
            except BaseException as e:
                raise e   