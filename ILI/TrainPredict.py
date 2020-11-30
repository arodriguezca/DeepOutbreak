import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import sys
import math
import numpy as np
import time
import os
import pandas as pd
import datetime
from EpiDeep import EpiDeepCOVID
from epiweeks import Week, Year
from EpiDeepHTL import test, trainKD
from collections import OrderedDict, defaultdict
from datetime import datetime
from utils import buildNetwork, prepare_train_data, save_results
from model_scripts.exog_model_utils import Model
import argparse
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float
EPIDEEP_SEQ_LEN=5
DROPOUT_PROB = .0  # only used in f1
max_hist=3
POST_TRAIN_EPOCHS = 100
parser = argparse.ArgumentParser(description="KD")
parser.add_argument('--data',type=str, default='all_old',help='The data to remove: testing, kinsa, social, linelist. If you do not want to remove, use all')
parser.add_argument('--remove',type=str, default='none',help='Module to remove for ablation: KD, recurrent, laplacian, reconstruction')
parser.add_argument('--alpha',type=float, default='0.1',help='KD loss weight')
parser.add_argument('--recon_weight',type=float, default='0.005',help='')
parser.add_argument('--_beta',type=float, default='0.001',help='laplacian weight')
parser.add_argument('--_lambda',type=float, default='0.01',help='region reconstruction weight')
parser.add_argument('--_gamma',type=float, default='0.0',help='equity (not used)')
parser.add_argument('--recon_emb',type=str, default='False',help='reconstruction by AE')
parser.add_argument('--plot',type=str, default='False',help='')
args = parser.parse_args()


def TrainPredict(regionName,iYear,currentWeek,method,path,epiweek,epochs,feat_data,it,suffix='',data='all',remove='none',alpha=0.6,recon_weight=0.005,_beta=0.001,_gamma=0.001,_lambda=0.01,recon_emb=False,plot=False,feat_hidden_size=20):  
    """
        #Here are the inputs example
        #regionName="X" or "Region Z"
        #iYear=2016 - current year
        #currentWeek=27
        #method=0   #0-3 for predicting wILI, 4 for peak value, 5 for peak-time, 6 for onset time
    """
    print('region ',regionName, ' week', epiweek)
    # suffix = 'hs'+str(feat_hidden_size)+'_'+suffix
    suffix = suffix + '_a'+str(alpha)+'_rw'+str(recon_weight) + '_b'+str(_beta) + '_g'+str(_gamma) + '_lamb' + str(_lambda)
    if remove!='none':
        suffix += '_r-'+remove
    if recon_emb:  # reconstruct embedding
        suffix += '_embrec'
    model_hyperparameters = MODEL_HYPERPARAMS + '_' + data + '.json'
    print(suffix)
    K=4
    k_week_ahead = 1
    predictions={}
    targets={}
    rmse={}
    while k_week_ahead <=K: 
        print("====== next", k_week_ahead)

        histILI_dataset, data_loader_hist_train,\
            data_loader_hist_test, dataset, region_graph,\
                data_loader_train, data_loader_test, size_feat_input_data,\
                    data_loader_train_overlap = \
                    prepare_train_data(regionName,epiweek,k_week_ahead,EPIDEEP_SEQ_LEN,device,RECURRENT,data)

        epideep = EpiDeepCOVID(EPIDEEP_SEQ_LEN, 20, EPIDEEP_SEQ_LEN+1, 20, 4, device=device)
        # pre-train epideep
        # load previous model if exists 
        path_model='./models/epideep_global_ew'+str(epiweek)+'_k'+str(k_week_ahead)+'_it'+str(it)+'.th'
        # path_model='./models/epideep_global_ew'+str(epiweek)+'_k'+str(k_week_ahead)+'.th'
        if os.path.exists(path_model):
            try:
                clustering.load_state_dict(torch.load(path_model))
                epochs=POST_TRAIN_EPOCHS  # change epochs b/c we have a pre-trained model
                print('>>>load successful')
            except:
                pass  # TODO: fix this to avoid errors in experiments
        else:
            # path_new_model = './models/'+outputName+'/m_ew'+str(epiweek)+'_i'+str(it)+feat_data+'_'+suffix+'.th'
            epideep.fit_with_dataloader(data_loader_hist_train,num_epoch=epochs)
        torch.save(epideep.state_dict(), path_model)
        # epideep.fit(*epideep_fit_inputs,num_epoch=epochs,pre_train_epochs=pre_train_epochs,model_path=path_new_model,epiweek=epiweek) 
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
                
            def forward(self, x):
                return x
        # replace decoder by identity
        epideep.decoder = Identity()
        epideep.regressor = Identity()
        module_g = buildNetwork([40,16]).to(device)
        module_h = buildNetwork([32,16]).to(device)
        module_f1 = buildNetwork([16,16,16],dropout=DROPOUT_PROB).to(device)
        if recon_emb:  # reconstruct embedding
            module_g_prime = buildNetwork([16,40]).to(device)
            module_h_prime = buildNetwork([16,32]).to(device)
        else:  # reconstruct input data
            module_g_prime = buildNetwork([16,EPIDEEP_SEQ_LEN]).to(device)
            # module_h_prime = buildNetwork([16,int(size_feat_input_data/EPIDEEP_SEQ_LEN)]).to(device)
            module_h_prime = buildNetwork([16,size_feat_input_data]).to(device)
        module_f2 = buildNetwork([16,1]).to(device)
        feat_module = Model(model_hyperparameters,device)

        #Training
        total_loss_per_epoch,preds,actual_values,real_inputs =\
            trainKD(data_loader_train,feat_module,device,data_loader_hist_train,\
                epideep,module_g,module_f1,module_f2,module_h,module_g_prime,module_h_prime,\
                    epiweek,it,k_week_ahead, data_loader_train_overlap,suffix,remove,\
                        alpha,recon_weight,region_graph.laplacian_dict,_beta,_gamma,_lambda,recon_emb,plot)
        # overall,per_region_rmse = feat_module.evaluate(predictions=preds,targets=actual_values,regions=dataset.trainRegions)
        feat_module.plot_training_loss(total_loss_per_epoch)


        print("Train End Date = {}".format(epiweek))
        #print("Train Inputs Hist WILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:3]),np.min(real_inputs[:,:3]),np.mean(real_inputs[:,:3]))) 
        print("Train Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:5]),np.min(real_inputs[:,:5]),np.mean(real_inputs[:,:5])))
        print("Train Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(preds),np.min(preds),np.mean(preds))) 
        
        #=================================== Evaluation ======================================#
        inv_trans_df = dataset.inverse_transform(preds,actual_values,dataset.trainRegions,\
                                            dataset.target_col,transform_targets=True)
        
        preds = inv_trans_df['inv_transformed_predictions'].values.tolist()
        tgts = inv_trans_df['inv_transformed_targets'].values.tolist()
        reg = inv_trans_df[dataset.region_col].values.tolist()

        overall,per_region_rmse = feat_module.evaluate(predictions=preds,targets=tgts,regions=reg,region_col=dataset.region_col)
        feat_module.plot_training_loss(total_loss_per_epoch)
        
        print("Train RMSE Overall = {}, Per Region = {}".format(overall,per_region_rmse))
        #Testing
        # test_preds,test_targets,real_inputs=test(data_loader_test,feat_module)
        test_preds,test_targets,real_inputs=test(data_loader_test,feat_module,module_f1,module_f2,module_h)
        print("Test Inputs COVID MAX = {}, MIN = {}, MEAN = {}".format(np.max(real_inputs[:,:5]),np.min(real_inputs[:,:5]),np.mean(real_inputs[:,:5]))) 
        print("Test Predictions wILI MAX = {}, MIN = {}, MEAN = {}".format(np.max(test_preds),np.min(test_preds),np.mean(test_preds)))

        #Evaluation in test
        
        #Inverse Transform
        inv_trans_df_test = dataset.inverse_transform(test_preds,test_targets,dataset.testRegions,\
                                                    dataset.target_col,transform_targets=False)
        preds = inv_trans_df_test['inv_transformed_predictions'].values.tolist()
        tgts = test_targets
        reg = inv_trans_df_test[dataset.region_col].values.tolist()

        overall_test,per_region_rmse_test = feat_module.evaluate(predictions=preds,\
                                targets=tgts,regions=reg,region_col=dataset.region_col)
        
        print("Test RMSE Overall = {}, Per Region = {}".format(overall_test,per_region_rmse_test))
        print(inv_trans_df_test)
        # import time
        # time.sleep(10)
        # quit()
        def get_results_dict(predictions,targets,regions,region_col):
            """
                @param predictions: Predictions List.
                @param targets: Targets List.
                @param regions: List object containing region name for each prediction, target instance. Same size as predictions , targets.
                @param region_col: The region_column name
                Return RMSE of the model.
                There are two return values: Overall RMSE, and Per-Region RMSE.
            """
            rmse = lambda x,y: np.sqrt(np.nanmean(np.square(x - y)))
            tmp = regions
            tmp = pd.DataFrame(tmp,columns=[region_col])
            tmp['predictions'] = predictions
            tmp['targets'] = targets

            per_region_rmse = OrderedDict()
            per_region_pred = OrderedDict()
            per_region_target = OrderedDict()

            for key,val in tmp.groupby(region_col):
                per_region_rmse[key] = rmse(val['predictions'].values.ravel(),val['targets'].values.ravel())
                per_region_pred[key] = val['predictions'].values.ravel().item()
                per_region_target[key] = val['targets'].values.ravel().item()
            return per_region_pred, per_region_rmse, per_region_target

        predictions['pred'+str(k_week_ahead)], rmse['rmse'+str(k_week_ahead)],\
            targets['val'+str(k_week_ahead)] = get_results_dict(preds,tgts,reg,dataset.region_col)

        k_week_ahead += 1

    for k_week_ahead in range(K+1,5):
        predictions['pred'+str(k_week_ahead)], rmse['rmse'+str(k_week_ahead)],\
            targets['val'+str(k_week_ahead)] = get_results_dict(np.full(len(preds), np.nan),np.full(len(tgts),np.nan),reg,dataset.region_col)

    # save values per region in different files (to determine) 
    
    # epiglobal for epideep + global
    path_rmse = './rmse_results/rmse_'+suffix+'.csv'
    path_res = './rmse_results/results_'+suffix+'.csv'
    save_results(path_rmse, path_res, predictions, rmse, targets, epiweek, it)







if __name__ == "__main__":
    epochs=350
    epochs=400
    year=2019
    data = 'all'
    data = args.data
    remove = args.remove
    alpha = args.alpha
    _beta = args._beta
    _gamma = args._gamma
    _lambda = args._lambda
    if args.recon_emb=='True':
        recon_emb = True
    else:
        recon_emb = False
    if args.plot=='True':
        plot = True
    else:
        plot = False
    recon_weight = args.recon_weight
    suffix='_KD_'+ data

    # RECURRENT=True
    if remove=='recurrent':
        RECURRENT=False
    else:
        RECURRENT=True

    if RECURRENT:
        MODEL_HYPERPARAMS = "./experiment_setup/feature_module/model_specifications/global_recurrent_feature_model"
    else:
        MODEL_HYPERPARAMS = "./experiment_setup/feature_module/model_specifications/global_feature_model"

    from joblib import Parallel, delayed

    feat_data=''
    path=''
    for ew_week in range(9,15+1): 
        week=ew_week+32
        method=0
        for it in range(1):
            TrainPredict('X',year,week,method,path,ew_week,epochs,feat_data,it,suffix,data,remove,alpha,recon_weight,_beta,_gamma,_lambda,recon_emb,plot)
        # runs=10
        # Parallel(n_jobs=runs)(
        #     delayed(TrainPredict)('X',year,week,method,path,ew_week,epochs,feat_data,it,suffix,data,remove,alpha,recon_weight,_beta,_gamma,_lambda,recon_emb,plot) for it in range(1,runs+1))
        