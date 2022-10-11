import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import tqdm
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import Dataset, TensorDataset
import argparse
import os
from Model import ER_MLP, KGMTL
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np


def eval_matrics(y_test, y_pred):

    MSE = mean_squared_error(y_test, y_pred)
    print('MSE=',MSE)
    RMSE =np.sqrt(MSE)
    print('RMSE=',RMSE)
    MAE= mean_absolute_error(y_test, y_pred)
    print('MAE=',MAE)

    R2=1-MSE/np.var(y_test)
    print("R2=", R2)


def evaluation(triple_loader, head_loader, tail_loader,device,mymodel):
    pred_rel=[]; pred_head=[]; pred_tail=[]; target_head=[]; ent=[]; attr=[]
    #model.eval()
    for x, y in triple_loader:                         # iterate through the dataloader
        x = x.to(device)
        with torch.no_grad(): 
            pred_triple = mymodel.StructNet_forward(x[:,0], x[:,1], x[:,2])
            pred_rel.append(pred_triple.detach().cpu()) 
            target_rel = y[:,0] 
 
    for x, y in head_loader:
        x = x.to(device)
        with torch.no_grad():
            pred_att_h = mymodel.AttrNet_h_forward(x[:,0], x[:,1])
            pred_head.append(pred_att_h.detach().cpu())
            target_head.append(y)
            ent.append(x[:,0].detach().cpu())
            attr.append(x[:,1].detach().cpu())
    preds_head = torch.cat(pred_head,0).numpy().reshape((-1,1)) 
    targets_head = torch.cat(target_head,0).numpy().reshape((-1,1))
    attrs= torch.cat(attr,0).numpy().reshape((-1,1))
    evs= torch.cat(ent,0).numpy().reshape((-1,1))
    #print('from eval.py',preds_head, sep='\t')
    #return preds_head, targets_head
    table = np.concatenate((evs, attrs, preds_head, targets_head),axis=1)
    eval_matrics(preds_head, targets_head)

    return table 



    # for x, y in tail_loader:
    #     x = x.to(device)
    #     with torch.no_grad():
    #         pred_att_t = mymodel.AttrNet_h_forward(x[:,0], x[:,1])
    #         pred_tail.append(pred_att_t.detach().cpu())
    #         target_tail=y[:,0]

    # preds_rel = torch.cat(pred_rel, dim=0).numpy()
    # preds_head = torch.cat(pred_head, dim=0).numpy()
    # preds_tail = torch.cat(pred_tail, dim=0).numpy()

    # targets_rel = torch.cat(target_rel, dim=0).numpy()
    # targets_head = torch.cat(target_head, dim=0).numpy()
    # targets_tail = torch.cat(target_tail, dim=0).numpy()


    #return preds_rel, preds_head, preds_tail 

def save_result(eval_result, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    # with open(file, 'w') as fp:
    #     writer = csv.writer(fp)
    #     writer.writerow(['idx', 'tested_pred'])
    #     for i, p in enumerate(preds):
    #         writer.writerow([i, p])

    # save result 
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['idx','e','a','pred_h','target_h'])
        for i, [e,a,p,t] in enumerate(eval_result[:]):
            writer.writerow([i, e, a, p, t])

def eval_headattr(head_loader, device, mymodel):
    pred_head=[]; target_head=[]; ent=[]; attr=[]
    #model.eval()
    for x, y in head_loader:
        x = x.to(device)
        with torch.no_grad():
            pred_att_h = mymodel.AttrNet_h_forward(x[:,0], x[:,1])
            pred_head.append(pred_att_h.detach().cpu())
            target_head.append(y)
            ent.append(x[:,0].detach().cpu())
            attr.append(x[:,1].detach().cpu())
    preds_head = torch.cat(pred_head,0).numpy().reshape((-1,1)) 
    targets_head = torch.cat(target_head,0).numpy().reshape((-1,1))
    attrs= torch.cat(attr,0).numpy().reshape((-1,1))
    evs= torch.cat(ent,0).numpy().reshape((-1,1))
    #print('from eval.py',preds_head, sep='\t')
    #return preds_head, targets_head
    table = np.concatenate((evs, attrs, preds_head, targets_head),axis=1)
    eval_matrics(preds_head, targets_head)

    return table 


    
    
    



