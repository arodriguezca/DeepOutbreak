from utils import check_region_data
import numpy as np
import pandas as pd
from datetime import date, timedelta
from visualization.vis import visualize_region
import time
from epiweeks import Week

death_remove = [] 
hosp_remove = [] 
death_fix_confidence = [] 
hosp_fix_confidence = []
death_bias={}
hosp_bias={}

# get cumulative
def get_cumsum_region(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    def convert(x):
        return int(str(x)[-2:])
    df['epiweek'] = df.loc[:,'epiweek'].apply(convert)
    # df = df[(df.loc[:,'epiweek'] <= ew) & (df.loc[:,'epiweek'] >= 5)]
    df = df[(df.loc[:,'epiweek'] >= 5)]
    if target_name=='death' or 'cum_death':
        # cum = df.loc[:,'deathIncrease'].sum()
        cum = df.loc[:,'death_jhu_incidence'].sum()
    elif target_name=='hosp':
        cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

def parse(region,ew,target_name,runs,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results/',sub_path='./submissions/'):
    """
        @param write_submission: bool
        @param visualize: bool
        @param data_ew: int, this is needed to use the most recent data file
                    if None, it takes the values of ew
    """
    if data_ew is None:
        data_ew=ew  
    if daily:
        k_ahead=30 # changed from 28 to 30 on nov14 as requested by CDC (only needed for training)
        datafile='./data/covid-hospitalization-data/covid-hospitalization-daily-all-state-merged_vEW202046.csv'
    else:
        k_ahead=4
        datafile='./data/covid-hospitalization-data/covid-hospitalization-all-state-merged_vEW202046.csv'

    if not check_region_data(datafile,region,target_name,ew): # checks if we should train
        return 0    

    prev_cum = get_cumsum_region(datafile,region,target_name,ew)
    print(region,prev_cum)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []
    for next in range(1,k_ahead+1):
        import os
        path=res_path+'EW2020'+str(ew)+'/'+target_name+'_'+region+'_next'+str(next)+suffix+'.csv'
        if not os.path.exists(path):
            print(path)
            continue
        predictions = []
        with open(path, 'r') as f:
            for line in f:
                pred = float(line)
                predictions.append(pred)
            
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        quantiles = np.quantile(predictions, quantile_cuts)
        # print(quantiles)
        df = pd.read_csv(datafile, header=0)
        df = df[(df['region']==region)]
        # add to list
        lower_bounds_preds.append(quantiles[1])
        upper_bounds_preds.append(quantiles[-2])
        point_preds.append(np.mean(predictions))
        
        suffix_=suffix

    if visualize:
        if target_name=='death':
            print('==='+target_name+' '+region+'===')
            visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)
            if region=='X':
                visualize_region(target_name,region,point_preds,datafile,'cum',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)
                import time
        if target_name=='hosp':

            print('==='+target_name+' '+region+'===')
            visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)


    
if __name__ == "__main__":
    
    PLOT=True
    ew=46  # change prediction week

    states = pd.read_csv("./data/states.csv", header=0, squeeze=True).iloc[:,1].unique()
    regions = np.concatenate((['X'], states),axis=0)
    regions = list(regions)
    
    target_name='death'
    daily=False
    temp_regions = regions

    suffix='M1_20_vEW2020'+str(ew)
    print(suffix)

    for region in temp_regions:
        parse(region,ew,target_name,None,suffix,daily,True,PLOT)
    target_name='hosp'
    suffix='M1_daily_10_vEW2020'+str(ew)
    temp_regions = regions
    daily=True
    for region in temp_regions:
        parse(region,ew,target_name,None,suffix,daily,True,PLOT)
    quit()
