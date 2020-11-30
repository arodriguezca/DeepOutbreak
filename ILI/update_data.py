# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:16:34 2018

@author: Bijaya
"""

# import requests
import sys
from clustering_datasets import is_number


rawDatafileName = 'data/ILINet.csv'

processedDatafileName = 'data/ILINetProcessed.csv'
raw1 = 'data/NationalILINet.csv'
raw2 = 'data/RegionalILINet.csv'


def download_data():
    pass


def mergeNationalRegional():
    f = open(rawDatafileName, "w")
    firstline = ''
    for tempfile in [open(raw1), open(raw2)]:
        firstline = tempfile.readline()
        secondline = tempfile.readline()
        f.write(tempfile.read())
 
def removeWeek53():
    """
        Moves data from rawDatafile to proccessedDatafile
    """
    
    outfile = open(processedDatafileName, 'w')
    infile = open(rawDatafileName,'r')
    line = infile.readline() ## don't go over the header
    outfile.write(line)
    for line in infile:
        
        arr = line.strip().split(',')
        if arr[3] != '53':
            outfile.write(line)
    
    outfile.close()
    

def validate():
    """
        Goes over all season, make sure they are all there, and they should have length 52, except last one
    """
    input_file =  processedDatafileName
    clusters_file = 'data/SeasonClustersFinal'
    
    
    seasonDic = {}
    allSeasons = {}
    
    for line in open(clusters_file):
        arr = line.strip().split()
        year = int(arr[0])
        season = int(arr[1])
        seasonDic[year] = season
        allSeasons[season] = True
    
    
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
    
    isValid = True
    
    region_order = []
    for region, raw in all_data.items():
        region_order.append(region)
        keylist = list(raw.keys())
        keylist.sort()
        for year in keylist:
            if year>=1998 and year<=2018 and len(raw[year]) != 52: 
                print(region, year)
                isValid = False
                
                
            


    return isValid 


def create_histILI_dataset():
    import pandas as pd
    from epiweeks import Week
    df = pd.read_csv('./data/ILINetProcessed.csv', header=0, squeeze=True,\
        usecols=[1,2,3,4], names=['region','year','week','wILI_historical'])
    # df.loc[:,'week'] = df['year'].astype(str) + df['week'].astype(str)

    df = df[(df['year']>=2004) & (df['year']<=2020)]
    # week = Week(2020,1)
    # change epiweek to date
    def convert(x,y):
        return Week(x,y).enddate()
    # dates = df.loc[:'week'].apply(convert)
    dates = df[['year','week']].apply(lambda x: convert(x['year'],x['week']),axis=1)
    df['week_end_date'] = dates
    df['week_end_date'] = pd.to_datetime(df['week_end_date'])
    # only use after 2004
    # add column
    df['wILI'] = df['wILI_historical']
    # ['wILI_historical'] = ['historical_endogeneous']  # NOTE: here you should put change col name
    df = df[['region','week_end_date','wILI_historical','wILI']].sort_values(by=['region','week_end_date'])
    df.to_csv('./data/hist_ILI_sorted.csv',index=False)
    # print(df)

print("here")

if __name__ == '__main__':
    print("here")
    mergeNationalRegional()
    download_data()
    removeWeek53()
    
    if validate():
        print('Data is Valid')

    create_histILI_dataset()
    


#%%%

# import pandas as pd
# from epiweeks import Week
# df = pd.read_csv('./data/ILINetProcessed.csv', header=0, squeeze=True,\
#     usecols=[1,2,3,4,13], names=['region','year','week','wILI_historical','no_providers'])
# # df.loc[:,'week'] = df['year'].astype(str) + df['week'].astype(str)

# df = df[(df['year']>=2019) & (df['year']<=2020)]
# # week = Week(2020,1)
# # change epiweek to date
# def convert(x,y):
#     return Week(x,y).enddate()
# # dates = df.loc[:'week'].apply(convert)
# dates = df[['year','week']].apply(lambda x: convert(x['year'],x['week']),axis=1)
# df['week_end_date'] = dates
# df['week_end_date'] = pd.to_datetime(df['week_end_date'])
# # only use after 2004
# # add column
# df['wILI'] = df['wILI_historical']
# # ['wILI_historical'] = ['historical_endogeneous']  # NOTE: here you should put change col name
# df = df[['region','week_end_date','wILI_historical','wILI','no_providers']]
# df=df.sort_values(by=['region','week_end_date'])
# df.to_csv('./data/hist_ILI_sorted.csv',index=False)
# print(df)
