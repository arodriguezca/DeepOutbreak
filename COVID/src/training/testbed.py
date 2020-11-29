"""
    Use to get a thread per region, and get several sequential runs per each region
"""
import time
# from Visualize import genfigure, genfigureCOVID
import os
import numpy as np
from collections import Counter
import argparse
import pandas as pd
from scipy import stats
from utils import *
# from sklearn.preprocessing import MinMaxScaler
import random
import json
from epiweeks import Week, Year

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Read command line inputs from user
"""
parser = argparse.ArgumentParser(description="DeepCOVID")
parser.add_argument('--infile1', nargs=1,
                    help="JSON file to be processed",
                    type=argparse.FileType('r'))

parser.add_argument('--infile2', nargs=1,
                    help="JSON file to be processed. Will overwrite infile1.",
                    type=argparse.FileType('r'))
parser.add_argument('--target',type=str, default='hosp',help='Prediction target: death or hosp')
parser.add_argument('--suffix',type=str, default='default',help='Suffix to be used in results file')
parser.add_argument('--runs',type=int, default='999',help='Number of runs')
parser.add_argument('--data_ew',type=str, default='default',help='The epidemic week in which the data was updated. Default is automatically the current epiweek. Format: YYYYWW')
parser.add_argument('--pred_ew',type=str, default='default',help='The epidemic week in which we want to update (only one single week). Default uses user start_ew and end_ew from json file. Format: YYYYWW')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--daily', dest='daily', action='store_true',help='Predict on daily basis')
group.set_defaults(daily=False)
args = parser.parse_args()

"""
    Loading a JSON object returns a dict.
"""
params = json.load(args.infile1[0])
# infile2 can overwrite params in infile1
if args.infile2 is not None:
    params2 = json.load(args.infile2[0])
    ##to overwrite the params from params2
    for label, data in params2.items():
        if not data: # empty string
            continue
        if label == 'include_col' or label == 'exclude_col':
            if label in params:
                params[label] = data + ',' + params[label]
            else:  # init
                params[label] = data
        else:
            params[label] = data

"""
    Set variables with some of command line inputs
"""
target = args.target  # string
daily = args.daily  # bool
# data epiweek
if args.data_ew=='default': # this is default, set to current epiweek
    data_ew = Week.thisweek(system="CDC") - 1  # -1 because we have data for the previous (ending) week
else:
    data_ew = args.data_ew
    # try to convert to epiweek
    try:
        data_ew = Week.fromstring(data_ew, system="CDC")
    except:
        raise "Wrong epiweek format. Use YYYYWW. E.g. 202015."

"""
    Set JSON file params to variables
"""
##include_col will be splited by a comma, it will form a list of columns'names eventually
jobs = params["jobs"]
suffix = params["suffix"]
start_ew = params["start_ew"]
end_ew = params["end_ew"]
if 'runs' in params:
    runs =params["runs"]
epochs = params['epochs']
n_models = params['n_models'] # number of bootstrapped datasets
n_samples = params['n_samples'] # No of samples on each boostrapped dataset
stochastic_reps = params['stochastic_reps']
states= params['states']
##Convert string true to boolean true
if params['norm_layer'] =="True":
    norm_layer = True
else:
    norm_layer = False
lr =params['lr']
model_type = params['model_type']
# target = params['target']

include_col = [item for item in params['include_col'].split(",")]
#include also target
if target=='death':
    include_col.append('target_death')
elif target=='hosp':
    include_col.append('target_hosp')
exclude_col = []
if 'exclude_col' in params:
    # # For sequences, (strings, lists, tuples), use the fact that empty sequences are false.
    # # https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty
    if params['exclude_col']:
        exclude_col = [item for item in params['exclude_col'].split(",")]
exclude_col = [i for i in exclude_col if i] # remove empty


"""
    Overwrite some params from JSON file with command line inputs
"""
if args.runs != 999:  # use user input if different from default
    runs = args.runs
if args.suffix != 'default':  # use user input if different from default
    suffix = args.suffix
# always add runs and version of data used for training
suffix += '_'+str(runs)+'_vEW'+data_ew.cdcformat()
if args.pred_ew != 'default':  # if diff from default, only predict a single week
    start_ew = args.pred_ew
    end_ew = args.pred_ew



if __name__ == "__main__":

    #jobs=args.jobs
    ##Please change the path of states.csv as needed
    if states =='All':
        states = pd.read_csv("./data/states.csv", header=0, squeeze=True).iloc[:,1].unique()
        regions = np.concatenate((['X'], states),axis=0)
        regions = list(regions)

    elif states=='OnlyUS':
        regions = ['X']

    elif states == 'All_noUS':
        states = pd.read_csv("./data/states.csv", header=0, squeeze=True).iloc[:,1].unique()
        regions = np.array(states)

    else:
        regions = [item for item in states.split(",")]
        # TODO: verify that all states are valid, i.e. there are no misspellings

    check_train_data=False

    # get list of epiweeks for iteration
    start_ew = Week.fromstring(start_ew)
    end_ew = Week.fromstring(end_ew)
    iter_weeks = get_epiweeks_list(start_ew,end_ew)

    for ew in iter_weeks:
        # Name covention: vEW stands for version of dataset expressed in EpiWeeks. The version depends on the epiweek of the dataset being used.
        path='./data/EW'+ew.cdcformat()
        if daily:
            datapath = path+'/train_data_daily_vEW'+data_ew.cdcformat()+'.csv'
        else:
            datapath = path+'/train_data_weekly_vEW'+data_ew.cdcformat()+'.csv'

        for region in regions:
            print('===== ', region, ' ew', ew,' =====')
            if not check_region_data(datapath,region,target,ew): # checks if we should train for that region
                continue
            start=time.time()
            #convert ew to numeric

            train(datapath,region,ew,target,n_models,runs,suffix,norm_layer, include_col, exclude_col, n_samples, stochastic_reps, jobs, daily)
            end=time.time()
            print('>>>>>>> total time',region,':',end-start)
