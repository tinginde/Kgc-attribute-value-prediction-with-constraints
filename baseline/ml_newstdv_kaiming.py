# -*- coding: utf-8 -*-
"""ML2021Spring - HW1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G2_nyhkScgWI_AjBEq6m3bjbh3qQ7-SF

Author: Heng-Jui Chang

"""
import math
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For data preprocess
import pandas as pd
import numpy as np
import csv
import os

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


myseed = 80215  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# **Some Utilities**

You do not need to modify this part.
"""

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

"""# **Preprocess**

We have three kinds of datasets:
* `train`: for training
* `dev`: for validation
* `test`: for testing (w/o target value)
"""

train_attri_data = pd.read_csv('LiterallyWikidata/files_needed/train_attri_data.csv')
valid_attri_data = pd.read_csv('LiterallyWikidata/files_needed/valid_attri_data.csv')
test_attri_data = pd.read_csv('LiterallyWikidata/files_needed/test_attri_data.csv')
# num_lit=np.load('LiterallyWikidata/files_needed/num_lit_std.npy')
train_attri_data=train_attri_data[['e','a','new_stdv']]
valid_attri_data=valid_attri_data[['e','a','new_stdv']]
test_attri_data=test_attri_data[['e','a','new_stdv']]
## constraint needed:
# pop_idx = dict_all_2_idx['P1082']
# gdp = dict_all_2_idx['P4010']
# nominal_gdp = dict_all_2_idx['P2131']
# nominal_gdp_per = dict_all_2_idx['P2132']
# gdp_per = dict_all_2_idx['P2299']
# date_of_birth = dict_all_2_idx['P569']
# date_of_death = dict_all_2_idx['P570']
# area = ['P2046']
# net_profit = dict_all_2_idx['P2295']
# retirement_age = dict_all_2_idx['P3001']
# age_of_majority = dict_all_2_idx['P2997']
# work_start = dict_all_2_idx['P2031']
# work_end = dict_all_2_idx['P2032']


## Load pretrain embedding
emb_ent = torch.load('LiterallyWikidata/files_needed/pretrained_kge/pretrained_complex_entemb.pt')
embedding_e = torch.nn.Embedding.from_pretrained(emb_ent)
## Preparing ent2idx
list_ent_ids =[]
with open('LiterallyWikidata/files_needed/list_ent_ids.txt','r') as f:
    for line in f:
        list_ent_ids.append(line.strip())
ent2idx = {e:i for i,e in enumerate(list_ent_ids)}

## Preparing ent embedding
def emb_e(attri_data):
    attri_data['ent_idx']= attri_data['e'].map(ent2idx)
    input_e = torch.LongTensor(attri_data['ent_idx'].to_numpy())
    entity_embedding = embedding_e(input_e)
    return entity_embedding

## Preparing att embedding
att2idx = {a:i for i,a in enumerate(train_attri_data['a'].unique())}
embedding_a = torch.nn.Embedding(len(train_attri_data['a'].unique()),128,padding_idx=0)
def emb_a(attri_data):
    attri_data['a_idx']=attri_data['a'].map(att2idx)
    input_a = torch.LongTensor(attri_data['a_idx'].to_numpy())
    attribute_embedding = embedding_a(input_a)
    return attribute_embedding


## concat two embedding
def x_data(attri_data):
    entity_embedding = emb_e(attri_data)
    attribute_embedding = emb_a(attri_data)
    x_data = torch.cat([entity_embedding,attribute_embedding],dim=1).detach().numpy()
    return x_data 

X_trainset, X_validset, X_testset = x_data(train_attri_data) , x_data(valid_attri_data), x_data(test_attri_data)
y_trainset, y_validset,y_testset = train_attri_data.loc[:,'new_stdv'].to_numpy(),valid_attri_data.loc[:,'new_stdv'].to_numpy(),test_attri_data.loc[:,'new_stdv'].to_numpy()

"""# **Setup Hyper-parameters**

`config` contains hyper-parameters for training and the path to save your model.
"""

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models_newstdv802_kaiming/', exist_ok=True)  # The trained model will be saved to ./models/

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 500,                # maximum number of epochs
    'batch_size': 100,               # mini-batch size for dataloader
    'learning_rate':0.0001,
    'early_stop': 30,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models_newstdv802_kaiming/' , # your model will be saved here
}




# from sklearn.model_selection import train_test_split
# X_trainset, X_validset, y_trainset, y_validset = train_test_split(x_data, y,test_size=0.2, random_state=802)
# X_validset, X_testset, y_validset, y_testset = train_test_split(X_validset, y_validset,test_size=0.5, random_state=802)
# train_attri_data, valid_attri_data = train_test_split(attri_data, test_size=0.2,stratify=attri_data['a'],
#                                                                     random_state=802)
# valid_attri_data, test_attri_data = train_test_split(valid_attri_data, test_size=0.5,stratify=valid_attri_data['a'],
#                                                                     random_state=802)


"""## **Dataset**

"""

class KGMTL_Data(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
        

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)





"""## **DataLoader**

A `DataLoader` loads data from a given `Dataset` into batches.

"""

train_set =KGMTL_Data(X_trainset,y_trainset)
valid_set =KGMTL_Data(X_validset,y_validset)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

"""# **Deep Neural Network**

`NeuralNet` is an `nn.Module` designed for regression.
The DNN consists of 2 fully-connected layers with ReLU activation.
This module also included a function `cal_loss` for calculating loss.

"""

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)

"""# **Train/Dev/Test**

## **Training**
"""

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],momentum=0.9, weight_decay=1e-6) 

    min_mse = math.inf
    loss_record = {'train_batch':[],'train': [], 'mean_valid_loss':[],'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0

    
    for epoch in range(n_epochs):
        model.train() 
        
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(tr_set, position=0, leave=True)
        
        # set model to training mode
        for x, y in train_pbar:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            #print(f'pred:{pred}')
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())
        print('Epoch = {:4d}, Training loss = {:.4f}'.format(epoch + 1,np.mean(loss_record['train'])))
        loss_record['train_batch'].append(np.mean(loss_record['train']))

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path']+'model_newstdv.pt')  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def eval_matrics(y_test, y_pred):

    MSE = mean_squared_error(y_test, y_pred)
    print('MSE=',MSE)
    RMSE =np.sqrt(MSE)
    print('RMSE=',RMSE)
    MAE= mean_absolute_error(y_test, y_pred)
    print('MAE=',MAE)

    R2=1-MSE/np.var(y_test)
    print("R2=", R2)




"""## **Validation**"""

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss

"""## **Testing**"""

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []; y_b=[]
    for x,y in tt_set:                            # iterate through the dataloader
        x ,y = x.to(device), y.to(device)                          # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())
            y_b.append(y.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy().reshape(-1,1)     # concatenate all predictions and convert to a numpy array
    y_b= torch.cat(y_b,0).numpy().reshape(-1,1) 
    table  = np.concatenate((preds, y_b),axis=1)
    eval_matrics(y_b,preds)
    return table

"""# **Testing**
The predictions of your model on testing set will be stored at `pred.csv`.
"""

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'pred_y','target_y'])
        for i, [p,t] in enumerate(preds[:]):
            writer.writerow([i, p, t])

"""# **Load data and model**"""

model = NeuralNet(256).to(device)  # Construct model and move to device
print(model)
"""# **Start Training!**"""
import pickle
model_loss, model_loss_record = train(train_loader, valid_loader, model, config, device)

#save loss record for plt
with open(config['save_path']+'model_newstdv.pickle','wb') as fw:
    pickle.dump(model_loss_record,fw,protocol=pickle.HIGHEST_PROTOCOL)
#plot_learning_curve(model_loss_record, title='deep model')

print('model_loss min_mse:',model_loss)

del model
model = NeuralNet(256).to(device)
ckpt = torch.load(config['save_path']+'model_newstdv.pt', map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
#plot_pred(dv_set, model, device)  # Show prediction on the validation set



preds = test(valid_loader, model, device)  # predict 
save_pred(preds, config['save_path']+'preds_result_model_newstdv')     # save prediction file to pred.csv
