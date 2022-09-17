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
sys.path.append('KGMTL4Rec/KGMTL4Rec')
sys.path.append('KGMTL4Rec/src')
from Model import ER_MLP, KGMTL
from Evaluation import *
from Data_Processing_copy_less import *
# For plotting learning curve
#from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description='KGMTL4REC')

    parser.add_argument('-ds', type=str, required=False, default="LiterallyWikidata/")
    parser.add_argument('-epochs', type=int, required=False, default=1)
    parser.add_argument('-batch_size', type=float, required=False, default=128)
    parser.add_argument('-lr', type=float, required=False, default=0.0001)
    parser.add_argument('-model_path', type=str, required=False, default='MLT')
    parser.add_argument('-emb_size', type=int, required=False, default=128)
    parser.add_argument('-hidden_size', type=int, required=False, default=64)
    parser.add_argument('-word_embd_size', type=int, required=False, default=300)
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
    word_embd_size = args.word_embd_size
    nb_items = args.nb_items
    batch_size = args.batch_size
    Ns = args.Ns
    hit = args.hit
    nb_hist = args.nb_hist



    ##****** Set Device ******
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 

    device = torch.device(dev)  
    
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
    
    # ## Define losses
    loss_fn = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # # Creates the train_step function for our model, loss function and optimizer
    train_step_triplet, train_step_head_att, train_step_tail_att, param_update = make_train_step(model, loss_fn, loss_mse, optimizer)

    ## Load Data
    # KGMTL_Data_local = KGMTL_Data(ds_path)
    print(f'train att set: {len(KGMTL_Data_local.train_attri_data)}')
    print(f'valid att set: {len(KGMTL_Data_local.valid_attri_data)}')
    print(f'test att set:  {len(KGMTL_Data_local.test_attri_data)}')
  
    X_train_triples, y_train_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.train_rel_data)
    print(f'train rel set: {len(X_train_triples)}')
    X_val_triplets, y_val_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.val_rel_data)
    print(f'val rel set: {len(X_val_triplets)}')
    
    ##debug
    # dicte2av = KGMTL_Data_local.create_dicte2av(KGMTL_Data_local.attri_data)
    # print(dicte2av)
    
    X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.train_rel_data)
    # print('x_train_head', X_train_head_attr)
    print(f'train att set: {len(X_train_head_attr)}')
    X_val_head_attr, X_val_tail_attr, y_val_head_attr, y_val_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.val_rel_data)
    print(f'val att set: {len(X_val_head_attr)}')
    
    #X_test_head_attr, X_test_tail_attr, y_test_head_attr, y_test_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.test_rel_data,KGMTL_Data_local.test_attri_data)
    #print(f'test att set: {len(X_test_head_attr)}')
    
    train_loader_triplets, train_loader_head_attr, train_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_train_triples, y_train_triplets, 
    X_train_head_attr, y_train_head_attr, 
    X_train_tail_attr, y_train_tail_attr, batch_size)
    
    valid_loader_triplets, valid_loader_head_attr, valid_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_val_triplets, y_val_triplets, 
    X_val_head_attr, y_val_head_attr, 
    X_val_tail_attr, y_val_tail_attr, batch_size)

 
    ## Training the model
    tr_loss = []
    val_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train() 
    for epoch in tqdm.tqdm(range(epochs)):
        loss_1_epoch = []; loss_2_epoch = []; loss_3_epoch = []
        for x_batch_triplets, y_batch_triplets in train_loader_triplets:
            optimizer.zero_grad()
            loss1 = train_step_triplet(x_batch_triplets.to(device), y_batch_triplets.to(device))
            loss_1 = param_update(loss1)
            loss_1_epoch.append(loss_1)
            ##
        print('epoch {}, Struct Training loss {}'.format(epoch, np.mean(loss_1_epoch)))\

        for x_batch_head_attr, y_batch_head_attr in train_loader_head_attr:
            optimizer.zero_grad()
            loss2 = train_step_head_att(x_batch_head_attr.to(device), y_batch_head_attr.to(device))
            loss_2 = param_update(loss2)
            loss_2_epoch.append(loss_2)
        ##
        print('epoch {}, Head Reg Training loss {}'.format(epoch, np.mean(loss_2_epoch)))
        for x_batch_tail_attr, y_batch_tail_attr in train_loader_tail_attr:
            optimizer.zero_grad()
            loss3 = train_step_tail_att(x_batch_tail_attr.to(device), y_batch_tail_attr.to(device))
            loss_3 = param_update(loss3)
            loss_3_epoch.append(loss_3)
        print('epoch {}, Tail Reg Training loss {}'.format(epoch, np.mean(loss_3_epoch)))

        ## Total loss
        print('epoch {}, SUM Training loss {}'.format(epoch, np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch)))
        tr_loss.append(np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch) )
        
       
        model.eval()
#試作relation部分
        # for x, y in valid_loader_triplets:                         # iterate through the dataloader
        #     x, y = x.to(device), y.to(device) 
        #     with torch.no_grad(): 
        #         pred = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
        #         rel_loss = model.cal_loss(pred, torch.reshape(y, (-1,1)))  # compute loss
        # total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
        # total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

        for x, y in valid_loader_head_attr:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model.AttrNet_h_forward(x[:,0], x[:,1])
                loss = loss_mse(pred, y)
                #print('epoch {}, Validation loss {}'.format(epoch, loss))
                val_loss.append(loss.detach().cpu().item())
            
        mean_valid_loss = sum(val_loss)/len(val_loss)
        print('epoch {}, Validation loss {}'.format(epoch, mean_valid_loss))
        #torch.save(model.state_dict(),'KGMTL4Rec/saved_model/')
        #model.train()
    
    x_tensor_test = torch.from_numpy(KGMTL_Data_local.test_attri_data)
    test_dataset = TensorDataset(x_tensor_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, model.AttrNet_h_forward, device) 
    save_pred(preds, 'pred.csv')


if __name__ == '__main__':
    main()    
    
