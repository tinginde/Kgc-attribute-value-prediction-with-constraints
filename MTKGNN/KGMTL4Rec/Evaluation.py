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


def make_train_step(model, loss_fn, loss_mse, optimizer):
    # Builds function that performs a step in the train loop
    def train_step_triplet(x, y):  
        output1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
        loss_1 = loss_fn(output1, torch.reshape(y, (-1,1)))
        return loss_1
    def AttrNet_h_forward(x, y):   
        output2 = model.AttrNet_h_forward(x[:,0], x[:,1])
        loss_2 = loss_mse(output2, torch.reshape(y.float(), (-1,1)))
        return loss_2
    def AttrNet_t_forward(x, y): 
        output3 = model.AttrNet_t_forward(x[:,0], x[:,1])
        loss_3 = loss_mse(output3, torch.reshape(y.float(), (-1,1)))
        return loss_3
    def param_update(loss):
        loss.backward()
        optimizer.step()
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step_triplet, AttrNet_h_forward, AttrNet_t_forward, param_update

# make new eval method for testing
# need to create triples for testing(x_test_triples, x_head, x_tail)
def predict(test_loader, model, device):
    preds = []
    for x in test_loader:
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x[:,0],x[:,1])                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

