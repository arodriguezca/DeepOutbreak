
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import math
import numpy as np

truncate_vis_500 = ['AK', 'AL', 'AZ', 'MA', 'NE', 'WA', 'WV', 'CT'] # sept 15

def visualize_region(target_name,region,predictions,datafile,opt,_min=None,_max=None,ew=19,suffix='',daily=False):
    """
        @param target_name: 'death' or 'hosp'
        @param predictions: [next1, next2, ...]
        @param datafile: e.g. "./data/merged-data3.csv"
        @param opt: 'inc' or 'cum'   NOTE: it converts inc to cumulatives
        @param _min: percentile for 5% - e.g. [min1,min2] for next2 pred  
        @param _max: percentile for 95% - e.g. [max1,max2] for next2 pred 
        @param daily: boolean

            
        ##### IMPORTANT !!!!!!!!!!!!!!!!!!!!
        @param ew: the week before the start of prediction, like ew =19, so the first week of prediction is 20, but the ground truth maybe more than 20
    """
    
    overlap= False
    df = pd.read_csv(datafile, header=0)
    df = df[df['region']==region]
    def convert(x):
        return int(str(x)[-2:])
    df['epiweek'] = df.loc[:,'epiweek'].apply(convert)
    ## The end week of the groud turth
    end_week = df['epiweek'].max()
    ## The length of predictions
    len_pred = len(predictions)

    ## determine whether rmse is needed
    ## then establish the variables in need
    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    ## overlap_inc is the instance ground truth in the same time inveral with overlap_pred
    if end_week - ew !=0:
        overlap= True
        #overlap_pred = predictions[:(end_week-ew)]
        df_overlap = df[(df.loc[:,'epiweek'] <= (end_week)) & (df.loc[:,'epiweek'] > ew)].copy() 
        if target_name =='hosp':
            overlap_inc = df_overlap.loc[:,'hospitalizedIncrease'].to_numpy()
        elif target_name == 'death' or target_name=='cum_death':
            overlap_inc = df_overlap.loc[:,'death_jhu_incidence'].to_numpy()

    df = df[(df.loc[:,'epiweek'] <= ew) & (df.loc[:,'epiweek'] >= 10)]   ## data only from 10 to ew
    epiweeks = list(df.loc[:,'epiweek'].to_numpy())
    days=list(df.loc[:,'date'].to_numpy())
    days=list(range(1,len(days)+1))
    if target_name == 'hosp':  # next starts at 1
        inc = df.loc[:,'hospitalizedIncrease'].to_numpy()
        title_txt = 'Hospitalizations'
    elif target_name == 'death' or target_name=='cum_death':
        # inc = df.loc[:,'deathIncrease'].to_numpy()
        inc = df.loc[:,'death_jhu_incidence'].to_numpy()
        title_txt = 'Mortality'

    # print(df)
    # print(epiweeks)
    # inc.insert(0,0) # add at the beginning

    if opt=='inc':
        y=inc
        if overlap ==True:
            y_overlap = overlap_inc
        label='Incidence'
    elif opt=='cum':
        if region=='X':
            cum = [1]
        else:
            cum = [0]
        for i in range(len(inc)-1):
            cum.append(inc[i+1]+cum[-1])
        #print(cum)
        y=cum
        #print(y)
        if overlap==True:
            overlap_inc[0] += y[-1]
            y_overlap  = np.cumsum(overlap_inc, dtype=np.float64)
        label='Cumulative'
        # print(_min,_max)
        # quit()
        _min[0] = _min[0]+cum[-1]
        _min[1:] = [_min[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_min))]
        _max[0] = _max[0]+y[-1]
        _max[1:] = [_max[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_max))]

        predictions[0] = predictions[0] + cum[-1]
        predictions[1:] = [sum(predictions[:i+1]) for i in range(1,len_pred)]
        # print(_min,_max)
    ## weeks of predictions: like from 10 to 18 is the range of ground truth, 19 to 21 is the range of predictions

    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    if overlap==True:
        overlap_pred = predictions[:(end_week-ew)]
        #print(end_week)
        overlap_pred_weeks = list(range(ew+1, end_week+1))
    pred_weeks = [w+epiweeks[-1] for w in range(1,1+len_pred)]
    pred_days = [w+days[-1] for w in range(1,1+len_pred)]
    # print([epiweeks[-1]]+pred_weeks)
    # print([y[-1]]+predictions)
    # f, ax = plt.subplots(figsize=fig_size)


    ## Calculate the RMSE
    if overlap ==True:  
        RMSE = []
        #print(overlap_pred)
        #print(y_overlap)
        for index in range(1,len(overlap_pred)+1):
            #print(overlap_pred[:index+1])
            #print(y_overlap[:index+1])
            # RMSE.append(mean_squared_error(overlap_pred[:index], y_overlap[:index], squared =False))
            # RMSE.append(np.sqrt(mean_squared_error(overlap_pred[:index], y_overlap[:index])))  # previous, which acumulates RMSE
            RMSE.append(np.sqrt(mean_squared_error([overlap_pred[index-1]], [y_overlap[index-1]])))
        y_overlap = y_overlap.tolist()
        red_x= [epiweeks[-1]] + overlap_pred_weeks
        red_y = [y[-1]] + y_overlap

    #print(RMSE)


    #print(overlap_pred_weeks)

    #print(red_x)
    #print(red_y)
    #print([epiweeks[-1]]+overlap_pred_weeks)
    #print([y[-1]]+ y_overlap)

    if daily:
        plt.plot(days,y,'b',label='Ground truth data from JHU',linestyle='-')
        plt.plot([days[-1]]+pred_days,[y[-1]]+predictions,linestyle='-', marker='o', markersize=1, linewidth=1, color='b',label='Associated predictions')
    else:
        ## Plot ground truth data first
        plt.plot(epiweeks,y,'b',label='Ground truth data from JHU',linestyle='-')
        ## The predictions starts from the last week of ground truth
        plt.plot([epiweeks[-1]]+pred_weeks,[y[-1]]+predictions,linestyle='--', marker='o', color='b',label='Associated predictions')
    ## Plot the overlap data
    
    if overlap==True:
        plt.plot(red_x,red_y,linestyle='-', color='r', marker= "^",label=' Associated Ground Truth to Compare')


        ##Plot RMSE 
        y_max = np.max(predictions)
        tick = y_max/10
        for  index,value in enumerate(RMSE): 
            plt.text( 0.85* end_week, 0.5*y_max - tick*index, 'rmse'+ str(index+1)+ ': '+str(round(value,2)), size=12)

    if daily:
        plt.xlabel('day')
    else:
        plt.xlabel('epidemic week')
    plt.ylabel(label+' '+target_name+' counts')

    #print(y)
    if _min is not None and _max is not None:  

        _min.insert(0,y[-1])
        _max.insert(0,y[-1])
        if daily:
            plt.fill_between([days[-1]]+pred_days, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        else:
            plt.fill_between([epiweeks[-1]]+pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        # plt.fill_between(pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
    # plt.xscale([0,])
    plt.legend(loc='upper left')
    plt.gca().set_ylim(bottom=0)
    if region in truncate_vis_500 and target_name=='hosp':
        plt.gca().set_ylim(top=500)
    # plt.show()
    if opt=='inc':
        if region=='X':
            plt.title('US Incidence '+title_txt)  
        else:
            plt.title(region+' '+label)  
        plt.savefig('./figures/'+region+'_'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('inc predictions >>>>>',predictions)
    else:
        if region=='X':
            plt.title('US Cumulative '+title_txt)  
        else:
            plt.title(region+' '+label)   
        plt.savefig('./figures/'+region+'_cum'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('cum predictions >>>>>',predictions)
    
    plt.close()
