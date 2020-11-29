import matplotlib.pyplot as plt
import pandas as pd
from vis import visualize_region





datapath = "../data/train_data_noscale20.csv"
target_name = "death"
region= 'X'
predictions = [13000.0, 9000.0]
_min = [8000.0,7000.0 ]
_max = [20000.0,12500.0]
ew =18
suffix=''
opt = 'inc'
visualize_region(target_name,region,predictions,datapath,opt,_min,_max,ew,suffix)
