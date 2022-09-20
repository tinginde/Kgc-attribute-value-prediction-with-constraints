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

def evaluation(triple_loader, head_loader, tail_loader,device,mymodel=KGMTL):
    pred_rel=[]; pred_head=[]; pred_tail=[]
    #model.eval()
    for x, y in triple_loader:                         # iterate through the dataloader
        x = x.to(device)
        with torch.no_grad(): 
            pred_triple = mymodel.StructNet_forward(x[:,0], x[:,1], x[:,2])
            pred_rel.append(pred_triple.detach().cpu())  
 
    for x, y in head_loader:
        x = x.to(device)
        with torch.no_grad():
            pred_att_h = mymodel.AttrNet_h_forward(x[:,0], x[:,1])
            pred_head.append(pred_att_h.detach().cpu())


    for x, y in tail_loader:
        x = x.to(device)
        with torch.no_grad():
            pred_att_t = mymodel.AttrNet_h_forward(x[:,0], x[:,1])
            pred_tail.append(pred_att_t.detach().cpu())

    preds_rel = torch.cat(pred_rel, dim=0).numpy()
    preds_head = torch.cat(pred_head, dim=0).numpy()
    preds_tail = torch.cat(pred_tail, dim=0).numpy()

    return preds_rel, preds_head, preds_tail    

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['idx', 'tested_pred'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
    
    



