import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import math
# from .models.DeepCOVID import DeepCOVID
from models.DeepCOVID import DeepCOVID
# params
#N_SAMPLES = 20
#N=3 # stochastic repetitions for each combination of prev predictions
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import os
device = torch.device("cpu")
dtype = torch.float

train_since_23_death = ['NJ','NY']
train_since_23 = ['NJ','AK','IN','MA', 'WA', 'WV']
train_since_29 = ['AL']
train_since_35 = ['AZ','KY','NE']

def get_region_data(datapath,next,region,target_name, include_col, exclude_col, n_samples,bootstrap=False,preds=None):
    '''
        Get regional data, will open only the file for that week

        @param target_name: 'death' or 'hosp'
        @param next: 1 or 2
        @param datapath: e.g. "./data/merged-data3.csv"
        @param check_train_data: to visualize data X,y,X_test
        @param preds: list of 1 or more past predictions
    '''

    start_col =4
    # NOTE: change this later
    df = pd.read_csv(datapath, header=0)
    # df = pd.read_csv("./data/merged-data3.csv", header=0)
    # df = df.drop(['deathIncrease','death_jhu_cumulative'],axis=1)
    if target_name=='hosp':
        if region in train_since_23:
            df = df[(df.loc[:,'epiweek'] >= 23)]
        elif region in train_since_29:
            df = df[(df.loc[:,'epiweek'] >= 29)]
        elif region in train_since_35:
            df = df[(df.loc[:,'epiweek'] >= 35)]
    elif target_name=='death':
        if region in train_since_23:
            df = df[(df.loc[:,'epiweek'] >= 23)]

    ###Select features columns based on include_col
    # print('exclude_col',exclude_col)
    for col in exclude_col:
        # include_col.remove(col)
        if col in include_col:
            include_col.remove(col)


    main = df.iloc[:,:start_col].copy()
    rest  = df[include_col].copy()
    ##Combine the first 4 columns of df and the include_col as a new dataframe
    df = pd.concat([main,rest], axis =1)

    df = df[(df['region']==region)].dropna(1,'all')  # drop empty columns
    ###exclude columns not used as features
    #df = df.drop(exclude_col, axis =1)


    # df = df.loc[:, (df != df.iloc[0]).any()]   # remove columns with constant value, i.e. zeros

    # drop some columns NOTE: fix later

    end_col = df.shape[1]  # NOTE: use only when removing next columns
    # include column

    # remove columns
    # try:
    #     df = df.drop(['kinsa_cases'],axis=1)
    # except:
    #     pass
    # try:
    #     df = df.drop(['Observed Number','Excess Higher Estimate'],axis=1)
    # except:
    #     pass
    # remove last obs iqvia -- NOTE: ONLY WORKS FOR WEEK20
    df.reset_index(drop=True,inplace=True)  # do it

    if target_name == 'hosp':  # next starts at 1
        y = df.loc[:,'target_hosp'].to_numpy().copy()
        df.drop(['target_hosp'],axis=1,inplace=True)
    elif target_name == 'death':
        y = df.loc[:,'target_death'].to_numpy().copy()
        df.drop(['target_death'],axis=1,inplace=True)
    # elif target_name == 'cum_death':
    #     y = df.loc[:,'deathCumulative'].to_numpy().copy()

    if target_name=='hosp':
        y = y[next+1:]  # fixing that for hosp it starts at 0
    else:
        y = y[next:]
    gt=y[-1]
    # add previous predictions
    if len(preds)>0:
        y = np.concatenate((y,preds))
        # for i in range(len(preds)):
            # y[-i-1] = preds[i]
    print('y',y)
    # print('preds',preds)
    # remove zeros from the beginning
    w = 0
    start = 0
    # print(y)
    # for yy in y:  # NOTE: commented on Sep 15, shouldn't change anything
    #     if yy != 0.:
    #         start = w
    #         break
    #     w += 1
    # change y
    y = y[start:]
    X = df.iloc[start:-1,4:end_col].to_numpy()
    X_test = df.iloc[-1,4:end_col].to_numpy().reshape(1, -1)
    # print(df.columns)
    # quit()
    if bootstrap:
        boot_idx,X_boot,y_boot = resample(range(len(y)),X,y,n_samples=n_samples)
    else:
        boot_idx,X_boot,y_boot = list(range(len(y))), X, y

    # out of bag indexes
    oob_idx = [x for x in range(len(y)) if x not in boot_idx]
    # print(oob_idx)
    X_oob = X[oob_idx,:]
    y_oob = y[oob_idx]
    X = X_boot
    y = y_boot
    # print(X,y,X_test,gt)
    # quit()
    return X,y,X_test,gt # ,X_oob,y_oob




def train(datapath,region,ew,target,n_models,runs,suffix,norm_layer, include_col, exclude_col, n_samples, stochastic_reps, jobs, daily, boot=True):
    """
        @param daily: boolean, true if daily model should be trained
        @param boot: use bootstrap sampling?
        @param n_samples: number of bootstrap samples 
        @param jobs: parallel threads
        @param stochastic_reps: do not use, deprecated
    """
    layers = [10,5,2]
    if daily:
        k_ahead= 31 # changed from 28 to 30+1 on nov14 as requested by CDC (only needed for training)
        # +1 because of shift, see below
        init_k=0  # shift one day b/c Sunday is not being submitted
    else:
        k_ahead = 4 # weekly
        init_k=1
    # layers = [15,10,5,2]
    # layers = [25,35,15,5]
    # layers = [5]
    def run_boot(datapath,path,next,region,ew,target,boot,predictions_dict,norm_layer,stochastic_reps):
        # sample only for next > 1
        pred_next = []
        if next > init_k:
            stoch_reps = stochastic_reps
            for n in range(init_k,next):
                pred_next.append(np.random.choice(predictions_dict[n], 1).item())
        else:
            stoch_reps=1
        # load data, which is different for each model
        X,y,X_test,gt = get_region_data(datapath,next,region,target, include_col, exclude_col, n_samples,boot,pred_next)
        #(datapath,next,region,ew,target_name, include_col, exclude_col, n_samples,bootstrap=False,preds=None)
        f = open(path,'a+') # append and + to create if doesn't exist
        predicted = []
        for _ in range(stoch_reps):
            counter=0
            m_best = None
            best_loss = np.inf
            error_train = []
            error_test = []
            while counter<runs:
                mlp = DeepCOVID(X.shape[1],layers,norm_layer)
                counter+=1
                m_loss = mlp.fit(X,y.reshape(-1,1))  # return loss in train
                pred = mlp.predict(X_test.tolist()).item()
                t_loss = np.sqrt(((pred - gt) ** 2))
                error_train.append(m_loss)
                error_test.append(t_loss)
                # print(m_loss, t_loss)
                if m_loss < best_loss: # and y_pred.mean() > 0.:  # to avoid zero predictions
                # if m_loss < best_loss:
                    best_loss = m_loss
                    m_best = mlp
            if m_best is None:
                return np.nan
                # continue
            pred = m_best.predict(X_test.tolist()).item() # it is only one
            f.write(str(pred)+'\n')
            f.flush()
            print('predicted',pred)
            predicted.append(pred)
        return predicted
        # predictions.append(pred)

    predictions_dict = {}
    if not os.path.exists('./results/'+'EW'+str(ew)):
        os.makedirs('./results/'+'EW'+str(ew))
    # for next in range(1,3):
    for next in range(init_k,k_ahead+1):  # 5:next4
        path='./results/'+'EW'+str(ew)+'/'+target+'_'+region+'_next'+str(next)+suffix+'.csv'
        predictions = Parallel(n_jobs=jobs)(
                delayed(run_boot)(datapath,path,next,region,ew,target,boot,predictions_dict,norm_layer,stochastic_reps) for _ in range(n_models))
        # TODO: to make below work, it needs ground truth
        predictions = np.array(predictions).reshape(-1).tolist()
        print(predictions)
        predictions_dict[next] = predictions




# hosp_state_error=['CA','DC','TX','IL','LA','NJ','PA','WA','MI','MO','NC','NE','NV','VI','DE']
hosp_state_error=[]
def check_region_data(datapath,region,target_name,ew):
    df = pd.read_csv(datapath, header=0)
    df = df[(df['region']==region)]
    if df.size == 0:
        print('region ', region, ' is missing!')
        return False
    if target_name == 'hosp':  # next starts at 1
        y = df.loc[:,'hospitalizedIncrease'].to_numpy()
        if region in hosp_state_error:
            print('region ', region, ' is in list of hosp error!')
            return False
    elif target_name == 'death':
        # y = df.loc[:,'deathIncrease'].to_numpy()
        y = df.loc[:,'death_jhu_incidence'].to_numpy()
    # elif target_name == 'cum_death':
    #     y = df.loc[:,'deathCumulative'].to_numpy()
    if y.sum()==0.:
        print('region ', region, ' is all zeros part!')
        return False
    return True


from epiweeks import Week, Year
# NOTE: if you change it, change also the one in ./data/data_utils.py
def get_epiweeks_list(start_ew,end_ew):
    """
        returns list of epiweeks objects between start_ew and end_ew (inclusive)
        this is useful for iterating through these weeks
    """
    iter_weeks = list(Year(2020).iterweeks())
    idx_start = iter_weeks.index(start_ew)
    idx_end = iter_weeks.index(end_ew)
    return iter_weeks[idx_start:idx_end+1]


def get_res_point_preds(region,ew,target_name,suffix,daily):
    """
        alex: returns a vector of 4 (weekly forecast) or 28 (daily forecast) that contains point estimates of our predictions

        @param ew: week for which we are making predictions
        @param daily: boolean
    """
    if daily:
        k_ahead=28
    else:
        k_ahead=4

    point_preds = []
    for next in range(1,k_ahead+1):
        import os
        path='./results/'+'EW'+str(ew)+'/'+target_name+'_'+region+'_next'+str(next)+suffix+'.csv'
        if not os.path.exists(path):
            print(path)
            continue
        predictions = []
        with open(path, 'r') as f:
            for line in f:
                pred = float(line)
                predictions.append(pred)
        point_preds.append(np.mean(predictions))

    return point_preds

def get_distribution(region, ew, target_name, suffix, daily):
    """
        returns a matrix of 4(weekly forecast) or 28(daily forecast) containing the distribution of predictions

        @param ew: week for which we are making predictions
        @param daily: boolean
    """
    if daily:
        k_ahead=28
    else:
        k_ahead=4

    distribution = []
    for next in range(1,k_ahead+1):
        import os
        path='./results/'+'EW'+str(ew)+'/'+target_name+'_'+region+'_next'+str(next)+suffix+'.csv'
        if not os.path.exists(path):
            print(path)
            continue
        predictions = []
        with open(path, 'r') as f:
            for line in f:
                pred = float(line)
                predictions.append(pred)
        distribution.append(predictions)
    return distribution


def get_ground_truth(region, ew, target_name, ground_truth_file, daily):
    """
        Returns the ground truth values to compare with predictions for particular week, region
        @param ground_truth_file: ground truth file for different targets from the most recent epiweek
    """

    overlap = False
    df = pd.read_csv(ground_truth_file, header=0)
    df = df[df['region']==region]
    def convert(x):
        return int(str(x)[-2:])
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert)
    ew = convert(ew)
    end_week = df['epiweek'].max()
    if end_week - ew != 0:
        overlap = True
        df_overlap = df[(df.loc[:,'epiweek'] <= end_week) & (df.loc[:, 'epiweek'] > ew)].copy()
        if target_name =='hosp':
            ground_truth = df_overlap.loc[:,'hospitalizedIncrease'].to_numpy()
        elif target_name == 'death' or target_name=='cum_death':
            ground_truth = df_overlap.loc[:,'death_jhu_incidence'].to_numpy()
    if daily:
        ground_truth = ground_truth[:min(28, (end_week-ew)*7)]
    else:
        ground_truth = ground_truth[:min(4, end_week-ew)]

    return ground_truth




def get_point_error(region,ew, target_name,  ground_truth_file, point_predictions, daily):
    """
        returns the error for point predictions for a model
        @param ground_truth_file: ground truth file for different targets from the most recent epiweek
        @param point_predictions: point predictions of the trained model without some signals
    """

    ground_truth = get_ground_truth(region, ew, target_name, ground_truth_file, daily)
    point_predictions = point_predictions[:len(ground_truth)]
    error = (np.array(ground_truth) - np.array(point_predictions))
    return error


def evaluate_error_metrics(region, ew, target_name, error_file, daily, error_metric):
    # if error_metric == "Absolute":
    #     error = abs(np.array(ground_truth) - np.array(point_predictions))/np.array(ground_truth)
    # elif error_metric == "RMSE":
    #     error = mean_squared_error(ground_truth, point_predictions, squared=False)
    # elif error_metric == "MAPE":
    #     try:
    #         error = np.mean(np.abs((np.array(ground_truth) - np.array(point_predictions)) / np.array(ground_truth))) * 100
    #     except ZeroDivisionError:
    #         print("Divided by zero, error in MAPE calculations.")
    # return error
    return 0



def get_distribution_error(dist_base, dist_ablation, error_metric):
    return 0
    """
        currently stubbed - returns the error for point predictions for a model trained with some signals dropped
        @param arr_base: point predictions of the baseline model with all signals
        @param arr_ablation: point predictions of the trained model without some signals
        @param error_metric: type of point error to evaluate, options so far are 'Absolute', 'RMSE', 'MAPE'
    """
