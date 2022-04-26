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
                        # df_res = reduce_mem_usage(df_res)
                        return df_res
                else:
                    return True
                
            except ResourceWarning:
                return True
            except BaseException as e:
                raise e   