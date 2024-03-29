from this import d
from typing import Union
import numpy as np

try:
    import dtw
    dtw_flag = True
except:
    dtw_flag = False
    print('failed to import dtw')
    print('try : pip install dtw-python')

def MASE(true, pred, train_true):
    n = train_true.shape[0]
    d = np.abs(np.diff(train_true)).sum()/(n-1)
    
    errors = np.abs(true - pred)
    return errors.mean()/d

def Mdape(true, pred):
    nonzero_idx = true != 0      
    ape = (true[nonzero_idx] -pred[nonzero_idx])/ true[nonzero_idx]
    return np.median(np.abs(ape))

def RMSSE(true, pred, train_true):
    '''
    true: np.array 
    pred: np.array
    train_true: np.array
    '''    
    h = len(train_true)

    numerator = np.mean(np.sum(np.square(true - pred)))    
    denominator = 1/(h-1) * np.sum(np.square((train_true[1:] - train_true[:-1])))  
    msse = 1/h * numerator/denominator    
    return msse ** 0.5

def Jin_RMSSE(true, pred):
    '''
    This is modified RMSSE version by Jin (author)
    The difference is normalization part which use test period true value (not traning period)
    '''
    
    return RMSSE(true, pred, true)

def MAPE(true, pred): 
    nonzero_idx = true != 0  
    
    ape = (true[nonzero_idx] -pred[nonzero_idx])/ true[nonzero_idx]
    
    return np.mean(np.abs(ape))


def SMAPE(true, pred):
    return 1/len(true) * np.sum(2 * np.abs(pred-true) / (np.abs(true) + np.abs(pred))*100)


def action_loss_RMSSE(
    true: np.array, 
    pred: np.array,
    auto_weighting = False,
):               
    denominator = np.mean((true[1:] - true[:-1])**2)        
    
    list_norm_action_loss =[0]
    for i in range(len(true)-1):               
        
        true_diff = true[i+1] - true[i]
        true_pred_diff = pred[i+1] - true[i]     
                        
        norm_diff_loss = np.abs(true_diff**2/ denominator)                                
        if auto_weighting:
            # prediction 증가량을 더해준다
            norm_diff_loss = norm_diff_loss * true_pred_diff        
        
        prediction_truth = true_diff * true_pred_diff        
        
        # same direction between pred and true increasement
        if prediction_truth > 0:
            list_norm_action_loss.append(-norm_diff_loss)
        
        # different direction between pred and true increasement
        elif prediction_truth < 0:
            list_norm_action_loss.append(norm_diff_loss)    
        
        else:            
            pass                       
    
    mean_norm_action_loss = np.mean(list_norm_action_loss)    
    return np.sqrt(np.abs(mean_norm_action_loss))* np.sign(mean_norm_action_loss)


def action_loss_MAPE(
    true: np.array, 
    pred: np.array,
    n_future=1,
    auto_weighting=False
):    
    
    list_action_loss = [0]
    list_true_pred_inc = [0]
    
    for i in range(len(true)-n_future):               
        
        if true[i] == 0:
            continue
        
        true_inc = ((true[i+n_future] - true[i]) / true[i])
        true_pred_inc = ((pred[i+n_future] - true[i]) / true[i])
        
        prediction_truth = true_inc * true_pred_inc        
        
        if not auto_weighting:
            effective_true_inc= np.abs(true_inc)            
        else:            
            effective_true_inc= np.abs(true_inc) * np.abs(true_pred_inc)         
        
        # same direction between pred and true increasement
        if prediction_truth > 0:
            list_action_loss.append(-effective_true_inc)
        
        # different direction between pred and true increasement
        elif prediction_truth < 0:
            list_action_loss.append(effective_true_inc)
        
        else:            
            pass              
        
        list_true_pred_inc.append(np.abs(true_pred_inc))           
    
    if not auto_weighting:    
        action_loss_mean = np.mean(list_action_loss)
    else:
        action_loss_mean = np.sum(list_action_loss) / np.sum(list_true_pred_inc)    
        if np.sum(list_true_pred_inc) == 0:
            action_loss_mean = 0 
    
    return action_loss_mean

def action_loss_MAPE2(
    true: np.array, 
    pred: np.array,
    n_future=1,
):    
    '''
    This metric consider inverstment based MAPE.
    Absolute percentage difference of true and predicted value between t and t+1 is considered.s
    
    It has still zero value problem
    '''    
    
    list_action_loss = [0]    
    
    for i in range(len(true)-n_future):                       
        if true[i] == 0:
            continue
        
        true_inc = ((true[i+n_future] - true[i]) / true[i])
        true_pred_inc = ((pred[i+n_future] - true[i]) / true[i])
        
        percentage_diff = np.abs(true_pred_inc - true_inc)        
        list_action_loss.append(percentage_diff)
    
    return np.mean(list_action_loss)


def DTW(true, pred):
    target_idx = np.where(true!=0)    
    
    true = true[target_idx]
    pred = pred[target_idx]
    
    return dtw.dtw(true, pred).distance

## This is generation part 
def persist_generation(true,                                                     
                       n_future=1,
                       train_true = None):
    '''
    makes prediction values using true value
    pred(t) = true(t-n)
    '''
        
    len_true = len(true)
    all_true = np.concatenate((train_true ,true))
    pred = all_true[-len_true-n_future: -n_future]
        
    return true, pred
    
    
'''
you can get all metric values using this function
'''

def all_metric(
    true : np.array, 
    pred : np.array, 
    train_true : Union[None, np.array] = None, 
    n_future=1,
    warning=False):
    
    res = {}
    
    nonzero_idx = true != 0  
    if np.count_nonzero(nonzero_idx) >0:        
        if warning:
            print('MAPE, Mdape caulcated omitting zero') 

    

    res['MAPE'] = MAPE(true, pred)
    res['Mdape'] = Mdape(true, pred)
    res['SMAPE'] = SMAPE(true, pred)
    res['action_loss_RMSSE'] = action_loss_RMSSE(true, pred)        
    res['action_loss_MAPE'] = action_loss_MAPE(true, pred, n_future=n_future)
    res['action_loss_RMSSE_autoWeighted'] = action_loss_RMSSE(true, pred, auto_weighting=True)        
    
    res['action_loss_MAPE2'] = action_loss_MAPE2(true, pred, n_future=n_future)
    res['action_loss_MAPE_autoweighted'] = action_loss_MAPE(true, pred, n_future=n_future, auto_weighting=True)
    
    
    if train_true is not None:
        true , persist_pred = persist_generation(true,  n_future=n_future, train_true=train_true)


        res['RMSSE'] = RMSSE(true, pred, train_true)
        res['persist_RMSSE'] = RMSSE(true, persist_pred, train_true)
        res['persist_norm_RMSSE'] = res['RMSSE'] / res['persist_RMSSE']        
        
        res['MASE'] = MASE(true, pred, train_true)
        res['persist_MASE'] = MASE(true, persist_pred, train_true)
        res['persist_norm_MASE'] = res['MASE'] / res['persist_MASE']

        res['Jin_RMSSE'] = Jin_RMSSE(true, pred)        

        res['persist_MAPE'] = MAPE(true , persist_pred)
        res['persist_MAPE_norm'] = res['MAPE'] / res['persist_MAPE']
        
        if dtw_flag is True:
            res['dtw'] = DTW(true, pred)
            if train_true is not None:
                res['persist_dtw'] = DTW( true , persist_pred)
                res['persist_norm'] = res['dtw'] / res['persist_dtw']
        
        res['persist_Mdape'] = Mdape(true , persist_pred)
        res['persist_Mdape_norm'] = res['Mdape'] / res['persist_Mdape']
        
        
        res['persist_SMAPE'] = SMAPE( true , persist_pred)
        res['persist_norm_SMAPE'] = res['SMAPE']/res['persist_SMAPE']
            
        res['persist_action_loss_MAPE'] = action_loss_MAPE( true , persist_pred, n_future=n_future)    
        
        res['persist_action_loss_MAPE_autoweighted'] = action_loss_MAPE(true , persist_pred, n_future=n_future, auto_weighting=True)    
        
        res['persist_action_loss_MAPE2'] = action_loss_MAPE2(true , persist_pred, n_future=n_future)

    else:
        if warning:
            print('For persistig metric, Front part of prediction can be missing as train_true data was not inserted')
            print('If you want precise persistance time series metric, please insert train_true data')


    
    
    
    
    return res

if __name__ =='__main__':
    print(persist_generation(np.array([1,2,3,4,8,10]), n_future=2))

