import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import Dataset, TensorDataset
import argparse
import sys,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append('./MTKGNN/KGMTL4Rec')
sys.path.append('LiterallyWikidata')
from Model_onlyATT import KGMTL
from Evaluation import *
from Data_Processing_copy_less import *
import math

def main():
    parser = argparse.ArgumentParser(description='KGMTL4REC')

    parser.add_argument('-ds', type=str, required=False, default="LiterallyWikidata/")
    parser.add_argument('-epochs', type=int, required=False, default=500)
    parser.add_argument('-batch_size', type=float, required=False, default=500
    )
    parser.add_argument('-lr', type=float, required=False, default=0.001)
    parser.add_argument('-model_path', type=str, required=False, default='MLT')
    parser.add_argument('-emb_size', type=int, required=False, default=128)
    parser.add_argument('-hidden_size', type=int, required=False, default=100)
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
    
    ## Load REL triples for task1
    X_train_triples, y_train_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.train_rel_data)
    print(f'train rel set: {len(X_train_triples)}')
    X_val_triplets, y_val_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.val_rel_data)
    print(f'val rel set: {len(X_val_triplets)}')
    # X_test_triplets, y_test_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.test_rel_data)
    # print(f'val rel set: {len(X_test_triplets)}')
    
    ## Load ATT triples 
    X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.train_rel_data)
    print(f'X_train_head_attr: {len(X_train_head_attr)}')
    X_val_head_attr, X_val_tail_attr, y_val_head_attr, y_val_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.val_rel_data)
    print(f'X_val_head_attr: {len(X_val_head_attr)}')
    # X_test_head_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.test_rel_data)
    # print(f'X_test_head_attr: {len(X_test_head_attr)}')
    
    # Put triples into TensorDataset
    train_loader_triplets, train_loader_head_attr, train_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    X_train_triples, y_train_triplets, 
    X_train_head_attr, y_train_head_attr, 
    X_train_tail_attr, y_train_tail_attr, batch_size)
    
    # valid_loader_triplets, valid_loader_head_attr, valid_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    # X_val_triplets, y_val_triplets, 
    # X_val_head_attr, y_val_head_attr, 
    # X_val_tail_attr, y_val_tail_attr, batch_size, mode='test')

    # test_loader_triplets, test_loader_head_attr, test_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
    # X_test_triplets, y_test_triplets, 
    # X_test_head_attr, y_test_head_attr, 
    # X_test_tail_attr, y_test_tail_attr, batch_size, mode='test')
    valid_loader_head_attr = KGMTL_Data_local.eval_loader(KGMTL_Data_local.valid_attri_data, batch_size)

    ## Training the model
    loss_record = {'rel_train':[],'rel_valid':[],'att_h_train':[],'att_t_train':[],'att_h_val':[],'att_t_val':[],'ast_train':[]}
    best_mse = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    early_stop_count = 0
    config = {'early_stop':15} 

    for epoch in range(epochs):
        model.train() 
        #loss_1_epoch = []; loss_2_epoch = []; loss_3_epoch = []; loss_4_ast= []
        for x_batch_triplets, y_batch_triplets in train_loader_triplets:
            optimizer.zero_grad()
            x,y= x_batch_triplets.to(device), y_batch_triplets.to(device)
            output1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
            loss_1 = model.loss_fn(output1, y)
            loss_1.backward()
            optimizer.step()
            loss_record['rel_train'].append(loss_1.detach().cpu().item())
            ##
        

        for x_batch_head_attr, y_batch_head_attr in train_loader_head_attr:
            optimizer.zero_grad()
            x,y= x_batch_head_attr.to(device), y_batch_head_attr.to(device)
            output2 = model.AttrNet_h_forward(x[:,0], x[:,1])
            loss_2 = model.cal_loss(output2, torch.reshape(y.float(), (-1,1)))
            loss_2.backward()
            optimizer.step()
            loss_record['att_h_train'].append(loss_2.detach().cpu().item())
        ##

        for x_batch_tail_attr, y_batch_tail_attr in train_loader_tail_attr:
            optimizer.zero_grad()
            x,y= x_batch_tail_attr.to(device), y_batch_tail_attr.to(device)
            output3 = model.AttrNet_h_forward(x[:,0], x[:,1])
            loss_3 = model.cal_loss(output3, torch.reshape(y.float(), (-1,1)))
            loss_3.backward()
            optimizer.step()
            loss_record['att_t_train'].append(loss_3.detach().cpu().item())


        ## Total loss
        print('epoch {}, SUM Training loss {}'.format(epoch, np.mean(loss_record['rel_train']) +  np.mean(loss_record['att_h_train']) + np.mean(loss_record['att_t_train'])))


        for k in range(4):
            pred_left, pred_right, target = model.forward_AST(batch_size)
            loss_AST = model.cal_loss(pred_left, target) + model.cal_loss(pred_right, target)
            loss_AST.backward()
            optimizer.step()
            loss_record['ast_train'].append(loss_AST.detach().cpu().item())
        print('epoch {}, training AST loss {}'.format(epoch, np.mean(loss_record['ast_train'])))
        # with open('AST_prediction', 'w') as fp:
        #         writer = csv.writer(fp)
        #         writer.writerow(['idx', 'ast_pred'])
        #         for i, p in enumerate(pred_left.detach().cpu()):
        #             writer.writerow([i, p])
  

        model.eval()
        # for x, y in valid_loader_triplets:                         # iterate through the dataloader
        #     x, y = x.to(device), y.to(device) 
        #     with torch.no_grad(): 
        #         pred_1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
        #         loss_1 = model.loss_fn(pred_1, y)
        #     loss_record['rel_valid'].append(loss_1.detach().cpu().item())
        # print('epoch {}, Training loss_rel {}'.format(epoch, np.mean(loss_record['rel_train']))) 
        # print('epoch {}, Validation loss_rel {}'.format(epoch, np.mean(loss_record['rel_valid'])))

        for x, y in valid_loader_head_attr:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred_2 = model.AttrNet_h_forward(x[:,0], x[:,1])
                loss_2 = model.cal_loss(pred_2, y)
            loss_record['att_h_val'].append(loss_2.detach().cpu().item())
        print('epoch {}, Training loss_head {}'.format(epoch, np.mean(loss_record['att_h_train'])))
        print('epoch {}, Validation loss_head {}'.format(epoch, np.mean(loss_record['att_h_val'])))
        #保存model
        if np.mean(loss_record['att_h_val']) < best_mse: 
            best_mse = np.mean(loss_record['att_h_val'])
            print('Better Performance! Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch , best_mse))
            
            torch.save(model.state_dict(),'exp_pretrained/saved_model/model_{}_{}_{}_0109.pt'.format(epochs, batch_size,learning_rate))
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        epoch += 1
        
        if early_stop_count == config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    with open('exp_pretrained/loss_record/model_{}_{}_{}_0109.pickle'.format(epochs, batch_size,learning_rate),'wb') as fw:
        pickle.dump(loss_record,fw,protocol=pickle.HIGHEST_PROTOCOL)
        
    #test model
    model.eval()
    table = eval_headattr(valid_loader_head_attr, device , mymodel=model) 
    save_result(table, 'exp_pretrained/predicted_result/model_{}_{}_att_head_0109.csv'.format(epochs,batch_size)) 

    # 

if __name__ == '__main__':
    main()

    
