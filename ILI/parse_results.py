import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    Parses rmse results file by 
    Import file as dataframe
"""


#%%
def parse(regionName,date,suffix_results):
    path_rmse="./rmse_results/"+str(regionName)+"_Future_"+date+suffix_results+".txt"
    df = pd.read_csv(path_rmse, header=0)
    df = df.groupby('epiweek', as_index=False).mean()[['epiweek','rmse1','rmse2','rmse3','rmse4']]
    df['region'] = regionName.split()[-1]
    print(df)
    return df

date='2020-05-20'
feat_data='_none'
suffix=''
suffix_results=feat_data+suffix

df_l = []
for region in range(11):
    if region == 0:                         #National
        regionName = "X"
    else:
        regionName = "Region "+str(region)  #HHS Region 1 to 10
    df_l.append(parse(regionName,date,suffix_results))
df = pd.concat(df_l)
# make region column be first
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
df.to_csv('./rmse_results/rmse'+suffix_results+'.csv', index=False, na_rep='')


#%%
import pandas as pd
path_inter='./rmse_results/rmse_global_intermediate.csv'
df = pd.read_csv(path_inter, header=0)
df.iloc[:,3:] = df.iloc[:,3:].where(df.iloc[:,3:] < 15, np.nan)  # remove outliers
df = df.groupby(['region','epiweek','date'], as_index=False).mean()[['region','epiweek','date','rmse1','rmse2','rmse3','rmse4']]
df.sort_values(by=['epiweek','region'])
df.groupby(['epiweek','date']).mean()
(df.drop('epiweek',1)).groupby(['region']).mean().mean(1) # per region

#%%
import pandas as pd
path_inter='./rmse_results/rmse_epiglobal_intermediate.csv'
# path_inter='./rmse_results/results2.csv'
df = pd.read_csv(path_inter, header=0)
# df.iloc[:,3:] = df.iloc[:,3:].where(df.iloc[:,3:] < 15, np.nan)  # remove outliers
df = df.groupby(['region','epiweek','date'], as_index=False).mean()[['region','epiweek','date','rmse1','rmse2','rmse3','rmse4']]
df.sort_values(by=['epiweek','region'])
df.groupby(['epiweek','date']).mean() # per week
(df.drop('epiweek',1)).groupby(['region']).mean().mean(1) # per region

#%%
import pandas as pd
path_inter='./rmse_results/rmse_globalepideep_intermediate.csv'
# path_inter='./rmse_results/results2.csv'
df = pd.read_csv(path_inter, header=0)
df.iloc[:,3:] = df.iloc[:,3:].where(df.iloc[:,3:] < 15, np.nan)  # remove outliers
df = df.groupby(['region','epiweek','date'], as_index=False).mean()[['region','epiweek','date','rmse1','rmse2','rmse3','rmse4']]
df.sort_values(by=['epiweek','region'])
df.groupby(['epiweek','date']).mean() # per week
(df.drop('epiweek',1)).groupby(['region']).mean().mean(1)  # per region

#%%
import pandas as pd
path_inter='./rmse_results/rmse_epiglobalKD_intermediate_newfeatmod.csv'
# path_inter='./rmse_results/results2.csv'
df = pd.read_csv(path_inter, header=0)
df.iloc[:,3:] = df.iloc[:,3:].where(df.iloc[:,3:] < 15, np.nan)  # remove outliers
df = df.groupby(['region','epiweek','date'], as_index=False).mean()[['region','epiweek','date','rmse1','rmse2','rmse3','rmse4']]
df.sort_values(by=['epiweek','region'])
# df.groupby(['epiweek','date']).mean().to_csv('b.csv') # per week
(df.drop('epiweek',1)).groupby(['region']).mean().mean(1)  # per region


path_inter='./rmse_results/rmse_epiglobalKD_intermediate_attention.csv'
path_inter='./rmse_results/rmse_epiglobalKD_intermediate_attention_hint_test.csv'

# path_inter='./rmse_results/results2.csv'
import pandas as pd
import numpy as np
def parse(path_inter):
    df = pd.read_csv(path_inter, header=0)
    # df.iloc[:,3:] = df.iloc[:,3:].where(df.iloc[:,3:] < 15, np.nan)  # remove outliers
    df = df.groupby(['region','epiweek','date'], as_index=False).mean()[['region','epiweek','date','rmse1','rmse2','rmse3','rmse4']]
    df.sort_values(by=['epiweek','region'])
    df.groupby(['epiweek','date']).mean().to_csv('week.csv') # per week
    (df.drop('epiweek',1)).groupby(['region']).mean().mean(1).to_csv('region.csv')  # per region

path_inter='./rmse_results/rmse_global_f1f2_intermediate_Gf1f2rec.csv'
parse(path_inter)
# path_inter='./rmse_results/rmse_epiglobal_intermediate_fixeddata.csv'
# parse(path_inter)
path_inter='./rmse_results/rmse_epiglobal_intermediate_HTLrec.csv'
parse(path_inter)
path_inter='./rmse_results/rmse_epiglobalKD_intermediate_KDexp1.csv'
parse(path_inter)
path_inter='./rmse_results/rmse_epiglobalKD_intermediate_KDrec.csv'
parse(path_inter)
path_inter='./rmse_results/rmse_epiglobalKD_intermediate_Alx2.csv'
parse(path_inter)

path_inter='./rmse_results/rmse_epiglobalKD_intermediate_KD_all.csv'
parse(path_inter)





















#%%

## KDD

import pandas as pd
import numpy as np

# def parse_kdd(file):
#     '''
#         set NSF_COL=True when there is one column for NSF in results (last column)
#     '''
#     # rmse_list = []
#     df = pd.read_csv(file,header=None,skipfooter=3)
#     print(df)
#     next1 = np.mean(np.sqrt((df[1] - df[5])**2))
#     next2 = np.mean(np.sqrt((df[2] - df[6])**2))
#     next3 = np.mean(np.sqrt((df[3] - df[7])**2))
#     next4 = np.mean(np.sqrt((df[4] - df[8])**2))
#     print(next1)
#     print(next2)
#     print(next3)
#     print(next4)
#     rmse_next = [next1,next2,next3,next4]
#     rmse_total = np.mean(rmse_next)
#     print(rmse_total)
#     # quit()
#     return rmse_next, rmse_total

# end=3
# regions = list(range(11))
# # regions.remove(2)
# RMSE_next_l = {}
# RMSE_region_l = {}
# for region in regions:
#     rmse_next_l = []
#     rmse_total_l = []
#     for epi_it in range(0,end):
#         if region == 0:                         #National
#             regionName = "X"
#         else:
#             regionName = "Region"+str(region)  #HHS Region 1 to 10
#         file='./results_kdd/'+regionName+'_Future_2019_it'+str(epi_it)+'.txt'
#         rmse_next, rmse_total = parse_kdd(file)  
#         rmse_next_l.append(rmse_next)
#         rmse_total_l.append(rmse_total)
#     RMSE_next_l[regionName] = np.mean(rmse_next_l,axis=0)
#     RMSE_region_l[regionName] = np.mean(rmse_total_l)

# rmse_ = []
# for key,val in RMSE_next_l.items():
#     rmse_.append(val)

# print(np.mean(rmse_,axis=0))  # 
# print(RMSE_region_l)

#%%
import pandas as pd
import numpy as np

## new!! AAAI
def parse_kdd(file,k_wk_ahead):
    '''
        set NSF_COL=True when there is one column for NSF in results (last column)
    '''
    # rmse_list = []
    df = pd.read_csv(file,header=None,skipfooter=3)
    df = df[df.iloc[:,0].astype(int)>=42]
    print(df)
    next1 = (df[1] - df[5])**2
    next2 = (df[2] - df[6])**2
    next3 = (df[3] - df[7])**2
    next4 = (df[4] - df[8])**2
    # print(next1)
    # print(next2)
    # print(next3)
    # print(next4)
    
    sq_err = [next1,next2,next3,next4]
    sq_err = sq_err[:k_wk_ahead+1]
    mse = np.mean(sq_err, axis=0)
    rmse = np.sqrt(mse)
    # print(rmse)
    # quit()
    return rmse
    # return rmse_next, rmse_total


end=5
regions = list(range(11))
# regions.remove(2)
RMSE_next_l = {}
RMSE_region_l = {}
weekly_RMSE=[]
for region in regions:
    rmse_next_l = []
    rmse_total_l = []
    for epi_it in range(0,end):
        if region == 0:                         #National
            regionName = "X"
        else:
            regionName = "Region"+str(region)  #HHS Region 1 to 10
        file='./results_kdd/'+regionName+'_Future_2019_it'+str(epi_it)+'.txt'
        rmse_week = parse_kdd(file,1)  
        rmse_next_l.append(rmse_week.tolist())
        # rmse_total_l.append(rmse_total)
    RMSE_next_l[regionName] = np.mean(rmse_next_l,axis=0)
    weekly_RMSE.append(rmse_next_l)
    # RMSE_region_l[regionName] = np.mean(rmse_total_l)

print('weekly')
np.mean(weekly_RMSE,axis=0).mean(axis=0)

# rmse_ = []
print('phase1-11')
for key,val in RMSE_next_l.items():
    # print(key)
    print(np.mean(val[0:3]))  # phase 1  use 3 for 11, 4 for 12
    # print(np.mean(val[4:]))  # phase 2
print('phase1-12')
for key,val in RMSE_next_l.items():
    # print(key)
    print(np.mean(val[0:4])) 

print('phase2-11')
for key,val in RMSE_next_l.items():
#     # print(key)
#     # print(np.mean(val[0:3]))  # phase 1  use 3 for 11, 4 for 12
    print(np.mean(val[4:]))  # phase 2
print('phase2-12')
for key,val in RMSE_next_l.items():
#     # print(key)
    print(np.mean(val[5:]))  # phase 2

print('all')
for key,val in RMSE_next_l.items():
#     # print(key)
    print(np.mean(val))  # phase 2

# print(np.mean(rmse_,axis=0))  # 
# print(RMSE_region_l)







# %%

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas as pd

def get_RMSE(path,k_wk_ahead=4):
    df_new=pd.read_csv(path)
    df_new.head()

    df_new['date'] = pd.to_datetime(df_new['date'])
    # df_new = df_new[df_new['date']>=dt.strptime('2020-02-01',"%Y-%m-%d")]
    # df_new = df_new[df_new['date']>=dt.strptime('2020-02-29',"%Y-%m-%d")]
    df_new = df_new[(df_new['date']>=dt.strptime('2020-02-29',"%Y-%m-%d")) & (df_new['date']<=dt.strptime('2020-04-11',"%Y-%m-%d"))]  # 2020-04-11 is 15 2020-04-25 is 17
    df_new.reset_index(inplace=True)
    df_new.drop('index',axis=1,inplace=True)
    # df_new.head()

    squared_error=dict()
    for key,val in df_new.groupby(['region','date','iter_number']):
        for i in range(1,5):
            sq_err=np.square(val['val{}'.format(i)] - val['pred{}'.format(i)])
            df_new.loc[val.index.values,'squared_error_{}'.format(i)] = sq_err

    
    cols = ['squared_error_{}'.format(i) for i in range(1,k_wk_ahead+1)]
    # if k_wk_ahead==4:
    #     cols=['squared_error_1','squared_error_2','squared_error_3','squared_error_4']
    # elif k_wk_ahead==2:
    #     cols=['squared_error_1','squared_error_2']
    sq_errs=df_new[cols]
    mean_squared_errors_across_kweek_forecasts = np.mean(sq_errs,axis=1)
    df_new['mean_sq_err'] = mean_squared_errors_across_kweek_forecasts

    rmses_per_region_per_iter_number={'region':[],'rmse':[],'iter_number':[]}

    #Mean across all Squared Errors for all dates, followed by square root of the mean.
    for key,val in df_new.groupby(['region','iter_number']):
    #     print(val.shape)
        rmse = np.sqrt(val['mean_sq_err']).mean()
        rmses_per_region_per_iter_number['region'].append(key[0])
        rmses_per_region_per_iter_number['iter_number'].append(key[1])
        rmses_per_region_per_iter_number['rmse'].append(rmse)

    rmses_per_region_per_iter_number=pd.DataFrame(rmses_per_region_per_iter_number)
    for key,val in rmses_per_region_per_iter_number.groupby('region'):
        print("Region = {}, RMSE = {}".format(key,val['rmse'].mean()))
    #%%
    print('by region')
    for key,val in rmses_per_region_per_iter_number.groupby('region'):
        # print("Region = {}, RMSE = {}".format(key,val['rmse'].mean()))
        # valx = val[val['rmse'] < 5]
        print(val['rmse'].mean())

    #%%
    rmses_per_date_per_iter_number={'date':[],'rmse':[],'iter_number':[]}

    #Mean across all Squared Errors for all regions, followed by square root of the mean.
    for key,val in df_new.groupby(['date','iter_number']):
        #print(val.shape)
        rmse = np.sqrt(val['mean_sq_err']).mean()
        rmses_per_date_per_iter_number['date'].append(key[0])
        rmses_per_date_per_iter_number['iter_number'].append(key[1])
        rmses_per_date_per_iter_number['rmse'].append(rmse)

    rmses_per_date_per_iter_number=pd.DataFrame(rmses_per_date_per_iter_number)
    for key,val in rmses_per_date_per_iter_number.groupby('date'):
        print("Date = {}, RMSE = {}".format(key,val['rmse'].mean()))
    #%%
    print('by date')
    for key,val in rmses_per_date_per_iter_number.groupby('date'):
        # print("Date = {}, RMSE = {}".format(key,val['rmse'].mean()))
        # valx = val[val['rmse'] < 5]
        print(val['rmse'].mean())
    print('total:',rmses_per_date_per_iter_number['rmse'].mean())

    print('region_date')
    
    df = df_new.drop(['date'],axis=1).groupby(['region','epiweek']).apply(lambda x: np.sqrt(x).mean())
    # print(df)
    return df['mean_sq_err']

    
# %%

for param in [0.001,0.01,0.1,1.]:
    print('param:',param)
    path="/home/alex/Insync/acastillo41@gatech.edu/OneDrive/Research/Projects/COVID-19/COVID-ILI-Forecasting/rmse_results/"+\
    "results_epiglobalKD_intermediate_KD_all_a0.01_rw{}_b0.001_g0.0".format(param) + ".csv"
    get_RMSE(path,4)

# %%
# beta doesn't affect much
for param in [0.01,0.1,1.,10.]:
    print('param:',param)
    path="./rmse_results/results__KD_all_old_a0.01_rw0.005_b{}_g0.0_lamb0.01".format(param) + ".csv"
    get_RMSE(path,4)

# %%
# recon_weight doesn't affect much
for param in [0.005, 0.05, 0.5, 5., 50.]:
    print('param:',param)
    path="./rmse_results/results__KD_all_old_a0.01_rw{}_b0.01_g0.0_lamb0.01".format(param) + ".csv"
    get_RMSE(path,4)

# %%
# alpha : large values degrade performance
for param in [0.01,0.1,1.,10.,100.]:
    print('param:',param)
    path="./rmse_results/results__KD_all_old_a{}_rw0.005_b0.01_g0.0_lamb0.01".format(param) + ".csv"
    get_RMSE(path,4)

# %%
path="/home/alex/Insync/acastillo41@gatech.edu/OneDrive/Research/Projects/COVID-19/COVID-ILI-Forecasting/rmse_results/"+\
"results_epiglobalKD_intermediate_KD_all_a0.001_rw0.1_b0.001_g0.0" + ".csv"
path='./rmse_results/var_model_results_new_with_2019_data.csv'
path='./rmse_results/results_epiglobalKD_intermediate_FFN_KD_all_a0.9_rw0.005_b0.01_g0.0.csv'
path='./rmse_results/EB_results.csv'
path='./rmse_results/best.csv'

# get_RMSE(path,4)
path='./rmse_results/results__KD_all_a0.1_rw0.005_b0.001_g0.0_lamb0.01.csv'
path='./rmse_results/results__KD_onlyMob_a0.1_rw0.005_b0.001_g0.0_lamb0.01.csv'
path='./rmse_results/results__KD_all_old_a0.1_rw0.005_b0.001_g0.0_lamb0.01.csv'
path='./rmse_results/results__KD_line_mob_a0.1_rw0.005_b0.01_g0.0_lamb0.01.csv'
path='./rmse_results/results__KD_all_new_a0.1_rw0.005_b0.01_g0.0_lamb0.01.csv'
path='./rmse_results/markovian_kde_results.csv' 
path='./rmse_results/results__KD_all_old_a0.9_rw0.005_b0.01_g0.0_lamb0.01.csv' 
path='./rmse_results/results__KD_all_new_a0.9_rw0.005_b0.01_g0.0_lamb0.01.csv' 
path='./rmse_results/results__KD_linelist_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' # degrades
path='./rmse_results/results__KD_kinsa_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' # degrades
path='./rmse_results/results__KD_testing_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' # improves
df=get_RMSE(path,4)

# %% for new AAAI plot
path='./rmse_results/results__KD_all_old_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' 
path='./rmse_results/results__KD_all_old_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' 
path='./rmse_results/results__KD_all_new_a0.01_rw0.005_b0.01_g0.0_lamb0.01.csv' 
path='./rmse_results/var_model_results_new_with_2019_data.csv' 
path='./rmse_results/best.csv'
path='./rmse_results/KDE_deltadensity_results.csv'
path='./rmse_results/EB_results.csv' 
path='./rmse_results/markovian_kde_results.csv' 
path='./rmse_results/var_model_results_new_with_2019_data_lag_3.csv' 
path='./rmse_results/var_model_results_new_with_2019_data_lag_10.csv' 
path='./rmse_results/SARIMA_results.csv' 
df=get_RMSE(path,1)
def phase(x):
    if x<=11:
        return 1
    elif x<=15:
        return 2
    else:
        return 3
df = df.reset_index()
df['phase'] = df['epiweek'].apply(phase)
df=df.groupby(['region','phase'],as_index=False).mean().sort_values(['phase','region'])#.to_csv('temp.csv')
df1=df[df['phase']==1]['mean_sq_err']
df2=df[df['phase']==2]['mean_sq_err']
print('1')
print(df1.to_string(index=False))
print('2')
print(df2.to_string(index=False))

# %%
ours = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.01_rw1.0_b0.001_g0.0".format(param) + ".csv"
get_RMSE(path,2)

# %%
# for model ablation
path='./rmse_results/best.csv'
get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_r-recurrent"+ ".csv"
# get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_r-reconstruction" + ".csv"
# get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_r-laplacian"+ ".csv"
# get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_r-region_reconstruction"+ ".csv"
get_RMSE(path,4)

# %%
# for data ablation
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_kinsa_a0.9_rw0.005_b0.01_g0.0"+ ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_linelist_a0.9_rw0.005_b0.01_g0.0" + ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_social_a0.9_rw0.005_b0.01_g0.0"+ ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_testing_a0.9_rw0.005_b0.01_g0.0"+ ".csv"
get_RMSE(path,4)


# %%
# param sensitivity
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.05_b0.01_g0.0"+ ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.0005_b0.01_g0.0" + ".csv"
get_RMSE(path,4)
# %%
# param sensitivity: beta (laplacian)
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.1_g0.0"+ ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.001_g0.0" + ".csv"
get_RMSE(path,4)
# %%
# param sensitivity: alpha (KD)
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.001_rw0.005_b0.01_g0.0"+ ".csv"
get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.05_rw0.005_b0.01_g0.0"+ ".csv"
# get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.1_rw0.005_b0.01_g0.0"+ ".csv"
# get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.3_rw0.005_b0.01_g0.0" + ".csv"
# get_RMSE(path,4)
# path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.5_rw0.005_b0.01_g0.0" + ".csv"
# get_RMSE(path,4)
# %%
# param sensitivity: _lambda (region reconstruction)
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_lamb0.1"+ ".csv"
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_lamb0.001"+ ".csv"
get_RMSE(path,4)
# %%
# param sensitivity: _lambda (region reconstruction)
path='./rmse_results/best.csv'
get_RMSE(path,4)
path = "./rmse_results/results_epiglobalKD_intermediate_KD_all_a0.9_rw0.005_b0.01_g0.0_lamb0.01"+ ".csv"
get_RMSE(path,4)
