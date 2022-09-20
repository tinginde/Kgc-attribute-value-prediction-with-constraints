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
import sys,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append('./MTKGNN/KGMTL4Rec')
sys.path.append('LiterallyWikidata')
from Model import KGMTL
from Evaluation import *
from Data_Processing_copy_less import *
# For plotting learning curve
#from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description='KGMTL4REC')

    parser.add_argument('-ds', type=str, required=False, default="LiterallyWikidata/")
    parser.add_argument('-epochs', type=int, required=False, default=10)
    parser.add_argument('-batch_size', type=float, required=False, default=128
    )
    parser.add_argument('-lr', type=float, required=False, default=10e-3)
    parser.add_argument('-model_path', type=str, required=False, default='MLT')
    parser.add_argument('-emb_size', type=int, required=False, default=50)
    parser.add_argument('-hidden_size', type=int, required=False, default=100)
    parser.add_argument('-nb_items', type=int, required=False, default=138)
    parser.add_argument('-Ns', type=int, required=False, default=3)
    parser.add_argument('-device', type=str, required=False, default="cuda:0")
    parser.add_argument('-nb_hist', type=int, required=False, default=1)
    parser.add_argument('-hit', type=int, required=False, default=10)
    args = parser.parse_args()

    ds_path = args.ds
    epochs = args.epochs
    learning_rate = args.lr
    model_path = args.model_path
    emb_size = args.emb_size
    hidden_size = args.hidden_size
    nb_items = args.nb_items
    batch_size = args.batch_size
    Ns = args.Ns



    ##****** Set Device ******
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 

    device = torch.device(dev)  
    
    # built init data
    KGMTL_Data_local = KGMTL_Data(ds_path,Ns=3)
    tot_entity = len(KGMTL_Data_local.entities)
    tot_rel = len(KGMTL_Data_local.relations)
    tot_attri = len(KGMTL_Data_local.attributes)
    # # Now we can create a model and send it at once to the device
    model = KGMTL(tot_entity, tot_rel, tot_attri , emb_size, hidden_size)
    # torch.cuda.empty_cache()
    model.to(device)
    # # We can also inspect its parameters using its state_dict
    print(model)
    
    # # check number of attr set triples
    print(f'train att set: {len(KGMTL_Data_local.train_attri_data)}')
    print(f'valid att set: {len(KGMTL_Data_local.valid_attri_data)}')

    # ## Define losses, optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    ## Load REL triples for task1
    X_train_triples, y_train_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.train_rel_data)
    print(f'train rel set: {len(X_train_triples)}')
    X_val_triplets, y_val_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.val_rel_data)
    print(f'val rel set: {len(X_val_triplets)}')
    X_test_triplets, y_test_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.test_rel_data)
    print(f'val rel set: {len(X_test_triplets)}')

    X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.train_rel_data)
    print(f'X_train_head_attr: {len(X_train_head_attr)}')
    X_val_head_attr, X_val_tail_attr, y_val_head_attr, y_val_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.val_rel_data)
    print(f'X_val_head_attr: {len(X_val_head_attr)}')
    
    X_test_head_attr, X_test_tail_attr, y_test_head_attr, y_test_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.test_rel_data)
    print(f'X_test_head_attr: {len(X_test_head_attr)}')
    
    # Put triples into TensorDataset
    train_loader_triplets, train_loader_head_attr, train_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_train_triples, y_train_triplets, 
    X_train_head_attr, y_train_head_attr, 
    X_train_tail_attr, y_train_tail_attr, batch_size)
    
    valid_loader_triplets, valid_loader_head_attr, valid_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_val_triplets, y_val_triplets, 
    X_val_head_attr, y_val_head_attr, 
    X_val_tail_attr, y_val_tail_attr, batch_size, mode='test')

    test_loader_triplets, test_loader_head_attr, test_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_test_triplets, y_test_triplets, 
    X_test_head_attr, y_test_head_attr, 
    X_test_tail_attr, y_test_tail_attr, batch_size, mode='test')

    ## Training the model
    tr_loss = []
    val_loss_fn = []
    val_loss_mse = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train() 
    for epoch in tqdm.tqdm(range(epochs)):
        loss_1_epoch = []; loss_2_epoch = []; loss_3_epoch = []
        for x_batch_triplets, y_batch_triplets in train_loader_triplets:
            optimizer.zero_grad()
            x,y= x_batch_triplets.to(device), y_batch_triplets.to(device)
            output1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
            loss_1 = loss_fn(output1, y)
            loss_1.backward()
            optimizer.step()
            loss_1_epoch.append(loss_1.detach().cpu().item())
            ##
        print('epoch {}, Struct Training loss {}'.format(epoch, np.mean(loss_1_epoch)))

        for x_batch_head_attr, y_batch_head_attr in train_loader_head_attr:
            optimizer.zero_grad()
            x,y= x_batch_head_attr.to(device), y_batch_head_attr.to(device)
            output2 = model.AttrNet_h_forward(x[:,0], x[:,1])
            loss_2 = loss_mse(output2, torch.reshape(y.float(), (-1,1)))
            loss_2.backward()
            optimizer.step()
            loss_2_epoch.append(loss_2.detach().cpu().item())
        ##
        print('epoch {}, Head Reg Training loss {}'.format(epoch, np.mean(loss_2_epoch)))
        for x_batch_tail_attr, y_batch_tail_attr in train_loader_tail_attr:
            optimizer.zero_grad()
            x,y= x_batch_tail_attr.to(device), y_batch_tail_attr.to(device)
            output3 = model.AttrNet_h_forward(x[:,0], x[:,1])
            loss_3 = loss_mse(output3, torch.reshape(y.float(), (-1,1)))
            loss_3.backward()
            optimizer.step()
            loss_3.item()
            loss_3_epoch.append(loss_3.detach().cpu().item())
        print('epoch {}, Tail Reg Training loss {}'.format(epoch, np.mean(loss_3_epoch)))

        ## Total loss
        print('epoch {}, SUM Training loss {}'.format(epoch, np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch)))
        tr_loss.append(np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch) )
        
       
        model.eval()
        for x, y in valid_loader_triplets:                         # iterate through the dataloader
            x, y = x.to(device), y.to(device) 
            with torch.no_grad(): 
                pred_1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
                loss_1 = loss_fn(pred_1, y)
            val_loss_fn.append(loss_1.detach().cpu().item()) 
        print('epoch {}, Validation loss_rel {}'.format(epoch, np.mean(val_loss_fn)))

        for x, y in valid_loader_head_attr:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred_2 = model.AttrNet_h_forward(x[:,0], x[:,1])
                loss_2 = loss_mse(pred_2, y)
            val_loss_mse.append(loss_2.detach().cpu().item())
        print('epoch {}, Validation loss_head {}'.format(epoch, np.mean(val_loss_mse)))
        #保存權重
        #torch.save(model.state_dict(),'KGMTL4Rec/saved_model/model_{}_{}_{}.pt'.format(epochs, batch_size,learning_rate))
        model.train()
    
    #test model
    model.eval()
    preds1, preds2, preds3 = evaluation(test_loader_triplets, test_loader_head_attr, test_loader_tail_attr, device , mymodel=model) 
    save_pred(preds1, 'predicted_result/epoch{}_preds_rel.csv'.format(epochs))
    save_pred(preds2, 'predicted_result/epoch{}_preds_att_head.csv'.format(epochs)) 
    save_pred(preds3, 'predicted_result/epoch{}_preds_att_tail.csv'.format(epochs))        

if __name__ == '__main__':
    main()    
    
