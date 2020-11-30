
import torch
import torch.nn as nn
from rnnAttention import RNN
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
import pickle; import os
from utils import EarlyStopping
from EpiDeep import EpiDeepCOVID
import matplotlib.pyplot as plt
dtype = torch.float



def test(data_loader_test,feat_module,module_f1,module_f2,module_h):
    preds=list()
    targets=list()
    # region_vals=list()
    real_inputs=list()

    feat_module.model.eval()
    module_f1.eval()
    module_f2.eval()
    module_h.eval()

    for batchXReal,batchXCat,batchY in data_loader_test:
        # wILI_preds,wILI_embedding,_,_ = model.predict(batchXReal,batchXCat)
        a,wILI_embedding,_,_ = feat_module.predict(batchXReal,batchXCat)
        out = module_h.forward(wILI_embedding)
        out = module_f1.forward(out)
        preds_target_model = module_f2.forward(out)
        # print(a)
        # preds_target_model = a
        preds.extend(preds_target_model.cpu().data.numpy().ravel().tolist())
        targets.extend(batchY.cpu().data.numpy().ravel().tolist())
        real_inputs.append(batchXReal.cpu().data.numpy())
        
    real_inputs = np.concatenate(real_inputs)
    return preds,targets,real_inputs




def trainKD(data_loader_train,feat_module,device,data_loader_hist_train,epideep,module_g,module_f1,module_f2,module_h,module_g_prime,module_h_prime,ew,it,next,data_loader_train_overlap,suffix,remove,alpha,recon_weight,laplacian_graph,_beta,gamma,_lambda,recon_emb,plot):
    """
        @param feat_module: feature module
        @param epideep: pre-trained epideep with removed last layers
    """
    
    #Hyperparameters
    NUMEPOCHS=feat_module.num_epochs
    NUMEPOCHS=350
    # NUMEPOCHS=50
    LEARNING_RATE=feat_module.learning_rate
    # _lambda=feat_module.laplacian_hyperparameter  # this is for reconstruction
    _lambda=_lambda  # this is for reconstruction
    _beta=_beta # for lapalacian
    _gamma = gamma # for region equity
    _recon_weight = recon_weight
    _alpha = alpha

    #Instantiate Loss and Optimizer.
    criterion = nn.MSELoss()
    params = list(feat_module.model.parameters()) + list(module_g.parameters()) +\
        list(module_f1.parameters()) + list(module_f2.parameters()) + list(module_h.parameters()) +\
            list(module_h_prime.parameters()) + list(module_g_prime.parameters())

    
    """
        Alternating training
    """
    # make the batches real
    list_batches_feat_module_overlap = [(a,b,c,d, batchXReal,batchXCat,batchY,batchRegions) for (a,b,c,d, batchXReal,batchXCat,batchY,batchRegions) in data_loader_train_overlap] 
    list_batches_epideep = [(a,b,c,d) for (a,b,c,d) in data_loader_hist_train] 
    
    #Bookkeeping Structures
    wILI_loss_per_epoch=list()
    total_loss_per_epoch=list()
    embedding_loss_per_epoch=list()
    embeddings_per_region=dict() #Capture Embeddings Per Region At The Final Epoch
    source_loss_per_epoch=list()
    target_loss_per_epoch=list()
    
    # optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),lr=LEARNING_RATE)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),lr=feat_module.learning_rate,weight_decay=feat_module.l2_reg_hyperparameter)
    for epoch in range(NUMEPOCHS):
        
        wILI_loss_per_batch=list()
        embedding_loss_per_batch=list()
        total_loss_per_batch=list()
        source_loss_per_batch = list()
        target_loss_per_batch = list()
        
        #Record predictions and actual values only from the last epoch.
        preds=list()
        actual_values=list()
        region_vals=list()
        real_inputs=list()
        
        total_loss = torch.tensor([0.], dtype=dtype, device=device)
        N=5
        # they won't change
        module_g.train()
        for param in module_g.parameters():
            param.requires_grad = True
        
        if remove!='epideep':
            for _ in range(1):
                source_loss = torch.tensor([0.], dtype=dtype, device=device)
                for _ in range(N):
                    # NOTE: we could optimize jointly, but let's try alternating
                    # source model: do a full pass with epideep 
                    batch_idx = np.random.choice(len(list_batches_epideep), 1).item()
                    batch_clustering_query_length,_,batch_rnn_data,batch_rnn_label = list_batches_epideep[batch_idx]  
                    epi_emb = epideep.predict(batch_clustering_query_length,batch_rnn_data)
                    out = module_g.forward(epi_emb)
                    out = module_f1.forward(out)
                    source_reconstructed = module_g_prime.forward(out)
                    preds_source_model = module_f2.forward(out)
                    
                    #Backpropagate and Update model
                    iter_source_loss = criterion(preds_source_model.squeeze(),batch_rnn_label.squeeze())
                    if recon_emb:  # reconstruct embedding
                        iter_source_recon_loss = criterion(source_reconstructed.squeeze(),epi_emb.squeeze())
                    else:
                        iter_source_recon_loss = criterion(source_reconstructed.squeeze(),batch_clustering_query_length.squeeze())
                    source_loss += iter_source_loss + _recon_weight*iter_source_recon_loss
                    # print(iter_source_recon_loss,iter_source_loss)
                    source_loss_per_batch.append(iter_source_loss.item() + _recon_weight*iter_source_recon_loss.item())
                optim.zero_grad()
                source_loss.backward()
                optim.step()

            # they won't change
            # epideep.eval()
            module_g.eval()
            for param in module_g.parameters():
                param.requires_grad = False

            # predict in the whole dataset to get nu
            module_g.eval()
            module_f1.eval()
            module_f2.eval()
            _max=-1*np.inf; _min=np.inf
            for clustering_query_length,_,rnn_data,_,_,_,Y,_ in list_batches_feat_module_overlap:
                out = module_g.forward(epideep.predict(clustering_query_length,rnn_data))
                out = module_f1.forward(out)
                preds_source_model = module_f2.forward(out)
                source_error_all_data = F.mse_loss(preds_source_model.squeeze(),Y.squeeze(),reduction='none')
                if source_error_all_data.max() > _max:
                    _max = source_error_all_data.max()
                if source_error_all_data.min() < _min:
                    _min = source_error_all_data.min()
            nu = _max - _min
        if remove=='epideep' or remove=='KD':  # removing epideep automatically removes KD
            nu = np.inf
            # _alpha = 1.
            _alpha = 0.
            source_loss = torch.tensor([0.], dtype=dtype, device=device)


        for _ in range(2):
            # target model:
            target_loss = torch.tensor([0.], dtype=dtype, device=device, requires_grad=True)
            for _ in range(N):
                batch_idx = np.random.choice(len(list_batches_feat_module_overlap), 1).item()
                batch_clustering_query_length,_,batch_rnn_data,batch_rnn_label,\
                    batchXReal,batchXCat,batchY,batchRegions = list_batches_feat_module_overlap[batch_idx]
                # epideep
                module_g.eval()
                module_f1.eval()  # to turn off dropout and batchnorm
                module_f2.eval()
                hint_source_emb = module_g.forward(epideep.predict(batch_clustering_query_length,batch_rnn_data))
                out = module_f1.forward(hint_source_emb)
                hint_source_emb = torch.tensor(hint_source_emb.data, requires_grad=False)
                preds_source_model = module_f2.forward(out)
                module_g.train()
                module_f1.train()
                module_f2.train()
                # print(batchXReal.shape,batchXCat.shape)
                wILI_preds,wILI_embedding,region_encoding_reconstructed,region_embedding = feat_module.predict(batchXReal,batchXCat)
                # print(wILI_embedding.shape)
                # print(wILI_embedding)
                hint_target_emb = module_h.forward(wILI_embedding)
                out = module_f1.forward(hint_target_emb)
                target_reconstructed = module_h_prime.forward(out)
                preds_target_model = module_f2.forward(out)
                if remove!='region_reconstruction':
                    region_embedding_reconstruction_loss = criterion(region_encoding_reconstructed,batchXCat)
                else:
                    region_embedding_reconstruction_loss = torch.tensor([0.], dtype=dtype, device=device, requires_grad=False)
                
                if remove!='reconstruction':
                    if recon_emb:  # reconstruct embedding
                        iter_target_recon_loss = criterion(target_reconstructed.squeeze(),wILI_embedding.squeeze())
                    else:
                        a=target_reconstructed.squeeze()
                        b=batchXReal.view_as(target_reconstructed).squeeze()
                        iter_target_recon_loss = criterion(target_reconstructed.squeeze(),batchXReal.view_as(target_reconstructed).squeeze())
                else:
                    iter_target_recon_loss = torch.tensor([0.], dtype=dtype, device=device, requires_grad=False)
                

                ############## Laplacian Reg. #################
                
                if remove!='laplacian':
                    lap_reg = 0.0 #Initialize Laplacian Regularization Loss to 0.0

                #Stack such that each row of the lap_mat variable corresponds to the row in the laplacian matrix per region.
                    lap_mat = np.vstack([laplacian_graph[r] for r in batchRegions.cpu().data.numpy().tolist()])
                    lap_mat = Variable(torch.from_numpy(lap_mat).float()).to(device) 
                    lap_reg = torch.matmul(torch.transpose(region_embedding,0,1),lap_mat)
                    lap_reg =  torch.matmul(lap_reg,region_embedding)
                    _eye = torch.eye(lap_reg.size(0)).to(device)  #Create identity mat. and move to `device`. Used for Hadamard prod. with lap_reg for trace calc. 
                    lap_reg = torch.sum(_eye*lap_reg).pow(2)  #Square the sum.  (Formulation, everything but the square is from: (Climate Multi-model Regression Using Spatial Smoothing))
                else:
                    lap_reg = torch.tensor([0.], dtype=dtype, device=device, requires_grad=False)
                ############## Laplacian Reg. End #############

                wILI_loss = criterion(preds_target_model.squeeze(),batchY.squeeze())

                # imiation loss
                pred_source_static = torch.tensor(preds_source_model.squeeze().data, requires_grad=False)
                source_error = F.mse_loss(pred_source_static.squeeze(),batchY.squeeze(),reduction='none')
                phi = (1 - source_error/ nu)
                hint_loss = torch.mean(phi*torch.norm(hint_source_emb-hint_target_emb,dim=1))
                imitation_loss = torch.mean(phi * F.mse_loss(pred_source_static, preds_target_model.squeeze(), reduction='none')) 
                # previous loss
                iter_loss =  wILI_loss + _alpha*imitation_loss + _alpha*hint_loss +\
                     _lambda*region_embedding_reconstruction_loss + _beta*lap_reg + _recon_weight*iter_target_recon_loss
                target_loss = target_loss + iter_loss
                target_loss_per_batch.append(iter_loss.item())
            optim.zero_grad()
            target_loss.backward(retain_graph=True)
            optim.step()
        # print(target_loss_per_batch)
        
        total_loss += source_loss + target_loss
 
        total_loss_per_batch.append(total_loss.item())
        embedding_loss_per_batch.append(region_embedding_reconstruction_loss.item())
        
        
        wILI_loss_per_epoch.append(np.mean(wILI_loss_per_batch))
        embedding_loss_per_epoch.append(np.mean(embedding_loss_per_batch))
        total_loss_per_epoch.append(np.mean(total_loss_per_batch))
        source_loss_per_epoch.append(np.mean(source_loss_per_batch))
        target_loss_per_epoch.append(np.mean(target_loss_per_batch))
        # print('s',source_loss_per_epoch[-1])
        # print('t',target_loss_per_epoch[-1])
    
    feat_module.model.eval()
    module_f1.eval()
    module_f2.eval()
    module_g.eval()
    module_h.eval()

    print("Preds Size = {}, Actual Values Size = {}".format(len(preds),len(actual_values)))
    

    # Plot 2
    # predict in training
    preds=list()
    targets=list()
    real_inputs=list()
    for batchXReal,batchXCat,batchY,_ in data_loader_train:
        # wILI_preds,wILI_embedding,_,_ = model.predict(batchXReal,batchXCat)
        _,wILI_embedding,_,_ = feat_module.predict(batchXReal,batchXCat)
        out = module_h.forward(wILI_embedding)
        out = module_f1.forward(out)
        preds_target_model = module_f2.forward(out)
        preds.extend(preds_target_model.cpu().data.numpy().ravel().tolist())
        targets.extend(batchY.cpu().data.numpy().ravel().tolist())
        real_inputs.append(batchXReal.cpu().data.numpy())
    if plot:
        fig,ax=plt.subplots(1,1,figsize=(12,8))
        ax.plot(targets,preds,'o',c='b')
        ax.set_xlabel('targets')
        ax.set_ylabel('preds')
        ax.legend(fontsize=16)
        ax.set_title("Predictions vs targets",fontsize=16)
        plt.savefig('./figures/'+suffix+'/PvsT_ew'+str(ew)+'_next'+str(next)+'_'+str(it)+suffix+'.png')
        plt.close()
    
    actual_values = targets
    real_inputs = np.concatenate(real_inputs)
    return total_loss_per_epoch,preds,actual_values,real_inputs