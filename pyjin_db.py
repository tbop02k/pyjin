import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.pool import Pool, NullPool

'''
ver1.0

supported db : MariaDB, Mysql

pandas ver 0.25 or later
'''
            
def connectDB(host, port, user, password, db, engine_type='None', dbtype='mysql+pymysql'):    
    connect_args={'ssl':{'fake_flag_to_enable_tls': True}}    
    db = '{dbtype}://{user}:{password}@{host}:{port}/{db}?charset=utf8&ssl=true'.format(dbtype=dbtype, host=host,user=user,port=port,password=password,db=db)

    if engine_type=='NullPool':
        engine = create_engine(db, connect_args=connect_args, poolclass=NullPool, pool_recycle=3600)
    else:
        #default poolclass = QueuePool
        engine = create_engine(db, pool_recycle=3600, connect_args=connect_args)
    conn = engine.connect()
    return conn

### exception이 어느 단계에서 발생을 했는가.
def conn_exec_close(connectInfo, query, output=None, is_return=True,**kwargs):
    with connectDB(**connectInfo) as conn:
        try:
            res = conn.execute(text(query), kwargs)

            if is_return:
                if output is None:
                    return res.fetchall(), res.keys()
                elif output=='df':
                    res = pd.DataFrame(res.fetchall(), columns=res.keys())
                    return res
            else:
                return True
        except ResourceWarning:
            return True
        except BaseException as e:
            raise e

def execute_query(conn,query,output=None, is_return=True, **kwargs):
    try:
        res = conn.execute(text(query), kwargs)

        if is_return:
            if output== 'df':
                res = pd.DataFrame(res.fetchall(), columns=res.keys())
            return res
        else:
            return True
    except Exception as e:
        print('query error : ', e)
        raise e
