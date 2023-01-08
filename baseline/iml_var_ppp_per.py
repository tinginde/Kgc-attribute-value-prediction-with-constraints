#!/usr/bin/env python
# coding: utf-8

# # **Import Some Packages**

# In[ ]:


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For data preprocess
import pandas as pd
import numpy as np
import csv
import os

from tqdm import tqdm

import math
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# # **Some Utilities**
# 
# You do not need to modify this part.

# In[ ]:


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['mean_train_loss'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['mean_train_loss']) // len(loss_record['mean_valid_loss'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['mean_train_loss'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['mean_valid_loss'], c='tab:cyan', label='dev')
    plt.ylim(10e+2,5*10e+5)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
    


# In[ ]:


def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds


# # Loading data

# In[ ]:


attri_data = pd.read_csv('LiterallyWikidata/files_needed/train_attri_data_minmax.csv')
attri_data_valid_v = pd.read_csv('LiterallyWikidata/files_needed/valid_attri_data_minmax.csv')
attri_data_test = pd.read_csv('LiterallyWikidata/files_needed/test_attri_data_minmax.csv')



# # 準備變數

# In[ ]:


var_name=["population","GDP (PPP)","PPP GDP per capita",
      "date of birth","date of death",
      "area","work period (start)","work period (end)",
      "coordinate location(latitude)","coordinate location(logtitude)","height"]
list_var = ['P1082','P4010','P2299','P569','P570','P2046','P2031','P2032','P625_Latitude','P625_Longtiude','P2048']


# # Making input data
# ## ent --> pretrained; att-->initial emb

# In[ ]:


# 用kgeemb順序
ent2idx ={}
with open('LiterallyWikidata/files_needed/list_ent_ids.txt','r') as fr:
    for i, word in enumerate(fr.readlines()):
        ent2idx[word.strip()] = i

#先用沒有標準化 y 
#attri_data_std_v = attri_data[['e','a','new_stdv']]

# att2idx = {}
# #rel2idx = {v:k for k,v in enumerate(relations['label'].unique())}

# with open('../LiterallyWikidata/files_needed/attribute.txt','r') as fr:
#     for i, word in enumerate(fr.readlines()):
#         att2idx[word.strip()] = i
        
att2idx = {v:k for k,v in enumerate(attri_data['a'].unique())}


# In[ ]:


#loading pre-trained embedding
emb_ent = torch.load('LiterallyWikidata/files_needed/pretrained_kge/pretrained_complex_entemb.pt')
embedding_e = torch.nn.Embedding.from_pretrained(emb_ent)
# input_e = torch.LongTensor(attri_data['ent_idx'].to_numpy())
# entity_embedding = embedding_e(input_e)*math.sqrt(2./128)


# In[ ]:


attri_data['a_idx']=attri_data['a'].map(att2idx)
attri_data['e_idx']=attri_data['e'].map(ent2idx)
attri_data_valid_v['a_idx']=attri_data_valid_v['a'].map(att2idx)
attri_data_valid_v['e_idx']=attri_data_valid_v['e'].map(ent2idx)
# attri_data_test['a_idx']=attri_data_test['a'].map(att2idx)
# attri_data_test['e_idx']=attri_data_test['e'].map(ent2idx)


# In[ ]:


attri_data_train=attri_data[['e','a','minmax']]
attri_data_valid=attri_data_valid_v[['e','a','minmax']]
# attri_data_test = attri_data_test[['e','a','minmax']]


# In[ ]:


#attri_valid_new = pd.concat([attri_data_valid,attri_data_test],axis=0)


# In[ ]:


# 做矩陣，ent * att， 交叉為值 v
def numeric_literal_array(data, ent2idx, att2idx):
    #'LiterallyWikidata/LitWD48K/train_attri_data'
    df_all = data

    # Resulting file
    num_lit = np.zeros([len(ent2idx), len(att2idx)],dtype=np.float32)

# Create literal wrt vocab
    for i, (s, p, lit) in enumerate(df_all.values):
        try:
            num_lit[ent2idx[s], att2idx[p]] = lit
        except KeyError:
            continue
    return num_lit


# num_lit shape (47998, 86)


# In[ ]:


#v值沒有標準化
num_lit = numeric_literal_array(attri_data[['e','a','v']], ent2idx, att2idx)

num_lit_valid = numeric_literal_array(attri_data_valid_v[['e','a','v']], ent2idx, att2idx)
print(num_lit.shape, num_lit_valid.shape)




# In[ ]:


#值用標準化的
num_lit_minmax = numeric_literal_array(attri_data_train, ent2idx, att2idx)
num_lit_minmax_valid = numeric_literal_array(attri_data_valid, ent2idx, att2idx)
#print(num_lit_stdv.shape)


# In[ ]:


# ## constraint needed:
#pop_idx = att2idx['P1082']
#gdp = att2idx['P4010']
#nominal_gdp = att2idx['P2131']
# nominal_gdp_per = att2idx['P2132']
gdp_per = att2idx['P2299']
# date_of_birth = att2idx['P569']
# date_of_death = att2idx['P570']
# area = ['P2046']
# work_start = att2idx['P2031']
# work_end = att2idx['P2032']
#longitude = att2idx['P625_Longtiude']

var_predict=gdp_per



# # 做x_list under principle:
# ## 1.先確定有true資料: ent的var有值
# ## 2.取其他的特徵: var以外的值存到inner_x_list




# def 過程
def create_x_list(var_idx,num_lit):
    x_list=[]

    for i, ent in enumerate(num_lit):
        if ent[var_idx] == 0:
            pass
        else:
            inner_x_list=[]

            for j in range(len(ent)):
#                 if j == var2_idx and ent[j]==0:
#                     #如果為0，補中位數
#                     inner_x_list.append(1)
#                 else:
                inner_x_list.append(ent[j])
            inner_x_list.append(i)
            x_list.append(inner_x_list)
    return x_list


# In[ ]:
def make_inputdata(var, num_lit,num_lit_valid,target,target_val):
    x_list =create_x_list(var,num_lit)
    x_list_valid=create_x_list(var,num_lit_valid)
    df_train =pd.DataFrame(x_list)
    df_valid = pd.DataFrame(x_list_valid)
    # 刪掉全為0值的feature
    df_train=df_train.loc[:,(df_train!=0).any(axis=0)]
    cols = [i for i in df_train.columns if i != var]
    ent = df_train.iloc[:,-1].values
    ent_val = df_valid.iloc[:,-1].values
    y_train = np.array([target[e][var] for e in list(ent)])
    y_valid = np.array([target_val[e][var] for e in list(ent_val)])

    x_train = df_train[cols]
    x_valid = df_valid[cols]

    return x_train, y_train, x_valid, y_valid

x_train, y_train, x_valid, y_valid = make_inputdata(var_predict, num_lit_minmax, num_lit_minmax_valid,num_lit,num_lit_valid)

input_e = torch.LongTensor(x_train.iloc[:,-1].to_numpy())
entity_embedding = embedding_e(input_e)*math.sqrt(2./128)
input_e_val = torch.LongTensor(x_valid.iloc[:,-1].to_numpy())
entity_embedding_val = embedding_e(input_e_val)*math.sqrt(2./128)

x_att=torch.FloatTensor(x_train.iloc[:,0:-1].to_numpy())
x_att_val=torch.FloatTensor(x_valid.iloc[:,0:-1].to_numpy())
x_train = torch.cat([x_att,entity_embedding],dim=1)
x_valid = torch.cat([x_att_val,entity_embedding_val],dim=1)

# In[ ]:


# x_train.iloc[:,-1].to_numpy()
# x_valid.iloc[:,-1].to_numpy()




#[list(att2idx.keys())[list(att2idx.values()).index(i)] for i in select_feature]


# In[ ]:


print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)


# In[ ]:


#data=pd.DataFrame(x_list,columns=list(range(len(x_list[0]))))


# In[ ]:


# y_list=list()
# for ent2 in num_lit:
#     if ent2[gdp] ==0:
#         pass
#     else:
#         y_list.append(ent2[gdp])


# In[ ]:


# for i in range(len(x_list)):
#     inner_x = x_list[i]
#     inner_x.append(y_list[i])
# x_list.append(inner_x)


# In[ ]:


# def select_feat(train_data, valid_data, select_all=True):
#     '''Selects useful features to perform regression'''

#     sc = StandardScaler()
#     train_data = sc.fit_transform(train_data)
#     valid_data = sc.transform(valid_data)
    
#     y_train, y_valid = train_data[:,-1], valid_data[:,-1]
#     raw_x_train, raw_x_valid = train_data[:,:-1], valid_data[:,:-1]
    

#     if select_all:
#         feat_idx = list(range(raw_x_train.shape[1]))
#     else:
#         feat_idx = select_feature # TODO: Select suitable feature columns.
        
#     return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], y_train, y_valid


# In[ ]:


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


device = get_device()                 # get the current available device ('cpu' or 'cuda')
#os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'n_epochs': 500,                # maximum number of epochs
    'batch_size': 32,               # mini-batch size for dataloader
    'learning_rate':1e-3,
    'early_stop': 15,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': './baseline/iml_pt/model_{}_nocon_pretraine_minmax.pt'.format(var_predict) , # your model will be saved here
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
}
    


# In[ ]:


# Set seed for reproducibility
same_seed(config['seed'])


# # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# # test_data size: 1078 x 117 (without last day's positive rate)
# train_data, valid_data = x_list_train, x_list_valid
# #train_data, valid_data = train_valid_split(x_list, config['valid_ratio'], config['seed'])

# # Print out the data size.
# print(f"""train_data size: {train_data.shape} 
# valid_data size: {valid_data.shape} """)
# # test_data size: {test_data.shape}""")


# # Select features
# x_train, x_valid, y_train, y_valid = select_feat(train_data, valid_data, config['select_all'])

# # Print out the number of features.
# print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset = KGMTL_Data(x_train, y_train),                                             KGMTL_Data(x_valid, y_valid)

#print('train_dataset', train_dataset[120])

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)





class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.Tanh(),
            nn.Dropout(0.5),
#             nn.Linear(64,64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
            nn.Linear(200, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        x = self.layers(x)
        x = x.squeeze(1)
        return x

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        
        return self.criterion(pred, target) + 1000


# # **Preprocess**
# 
# We have three kinds of datasets:
# * `train`: for training
# * `dev`: for validation
# * `test`: for testing (w/o target value)

# In[ ]:


loss_record={'train': [], 'dev': [],'mean_train_loss':[],'mean_valid_loss':[]} 

def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    #optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    #writer = SummaryWriter() # Writer of tensoboard
#     if not os.path.isdir('./models_var'):
#         os.mkdir('./models_var') # Create directory of saving models.
    
    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)   
            #print(f'-------predict: {pred}, y: {y}----------') 

            #x_constraint = torch.tensor([(x[i][cols.index(3)]*x[i][cols.index(46)]) for i in range(len(x))])
            #x_constraint = torch.tensor([x[i][pop_idx]*x[i][gdp_per] for i in range(len(x))])
            #print(x_constraint)
            #x_constraint = x_constraint.to(device)
            loss = model.criterion(pred, y)
            #loss = criterion(pred, y) + criterion(pred, x_constraint)
                    #x_constraint = 1000
                     
            # criterion(pred,x_constraint)
                # ((pred-x[0]*x[18])**2) 
                
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record["train"].append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record["train"])/len(loss_record["train"])
        #writer.add_scalar('Loss/train', mean_train_loss, step)
        loss_record['mean_train_loss'].append(mean_train_loss)

        model.eval() # Set your model to evaluation mode.
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                # print(f'x: {x}')
                loss = criterion(pred, y)

            loss_record["dev"].append(loss.item())
            
        mean_valid_loss = sum(loss_record["dev"])/len(loss_record["dev"])
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        #writer.add_scalar('Loss/valid', mean_valid_loss, step)
        loss_record['mean_valid_loss'].append(mean_valid_loss)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# ## **Validation**

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def eval_matrics(y_test, y_pred):

    MSE = mean_squared_error(y_test, y_pred)
    print('MSE=',MSE)
    RMSE =np.sqrt(MSE)
    print('RMSE=',RMSE)
    MAE= mean_absolute_error(y_test, y_pred)
    print('MAE=',MAE)

    R2=1-MSE/np.var(y_test)
    print("R2=", R2)


# ## **Testing**

# In[ ]:


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


# In[ ]:




# # **Load data and model**

# In[ ]:


model = NeuralNet(input_dim=x_train.shape[1]).to(device)  # Construct model and move to device
print(model)


# # **Start Training!**

# In[ ]:


trainer(train_loader, valid_loader, model, config, device)


# In[ ]:
#save loss record for plt
import pickle
with open('baseline/iml_var/loss_record_{}_tanh.pickle'.format(var_predict),'wb') as fw:
    pickle.dump(loss_record,fw,protocol=pickle.HIGHEST_PROTOCOL)

#max(loss_record["mean_valid_loss"]),min(loss_record["mean_valid_loss"])


# In[ ]:


#plot_learning_curve(loss_record, title='deep model')


# In[ ]:


#config['save_path']


# In[ ]:


del model
model = NeuralNet(input_dim=x_train.shape[1]).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)


# In[ ]:
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'pred_y','target_y'])
        for i, [p,t] in enumerate(preds[:]):
            writer.writerow([i, p, t])

preds = test(valid_loader, model, device) 
#print(preds)
save_pred(preds, 'baseline/iml_var/pred_result_{}_minmax_tanh'.format(var_predict))


# # In[ ]:


# y_valid


# # In[ ]:


# pd.set_option('display.float_format', lambda x: '%.3f' % x)


# # In[ ]:


# df2=pd.DataFrame(preds,columns=['predict','target'])


# # In[ ]:


# df2.describe()


# In[ ]:


# lim=2200
# preds=df2['predict']
# targets=df2['target']
# figure(figsize=(5, 5))
# plt.scatter(targets, preds, c='r', alpha=0.5)
# plt.plot([-10, lim], [-10, lim], c='b')
# plt.xlim(0, lim)
# plt.ylim(0, lim)
# plt.xlabel('ground truth value')
# plt.ylabel('predicted value')
# plt.title('Ground Truth v.s. Prediction')
# plt.show()


# GDP value prediction
# input: entities's GDP_per, pop and 27 var values, no ent embddings 
# adding gdp_per * pop constraint
# tried: lambda 0.0075, 0.2, 1000 cannot make any effect
# tried2: pred<0 cons=1000 it run into some technical isse
# value: no normalized
# problem: after one epoch cannot updated
# result: loss is large but training loss decreased slowly, with constraint no effect
# reason why: data is small(125), valid features is a lot of 0

# In[ ]:





# # **Testing**
# The predictions of your model on testing set will be stored at `pred.csv`.

# In[ ]:




# preds = test(valid_loader, model, device)  # predict COVID-19 cases with your model
# print('pred')         # save prediction file to pred.csv


# # **Reference**
# This code is completely written by Heng-Jui Chang @ NTUEE.  
# Copying or reusing this code is required to specify the original author. 
# 
# E.g.  
# Source: Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
# 
