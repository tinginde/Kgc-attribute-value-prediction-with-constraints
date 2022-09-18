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
from torch.utils.data import Dataset, TensorDataset, random_split
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from Model import ER_MLP, KGMTL
from Evaluation import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



class KGMTL_Data():
    
    def __init__(self, ds_path, Ns):
        self.Ns = Ns
        ## Load Data for Relnet
        self.train_rel_data = pd.read_csv(ds_path + 'LitWD1K/train.txt', sep='\t', names=['s', 'p', 'o'])
        self.val_rel_data = pd.read_csv(ds_path + 'LitWD1K/valid.txt', sep='\t', names=['s', 'p', 'o'])
        self.test_rel_data = pd.read_csv(ds_path + 'LitWD1K/test.txt', sep='\t', names=['s', 'p', 'o'])
        
        ## Load Data for Attnet
        self.attri_data = pd.read_csv(ds_path + 'files_needed/numeric_literals_final_ver04', sep='\t')
        #self.attri_data = attri_data.to_numpy() 
#       self.attri_data = self.attri_data.sample(frac=0.5, random_state=42)
        self.train_attri_data, valid_attri_data = train_test_split(self.attri_data, test_size=0.2,stratify=self.attri_data['a'],
                                                                    random_state=802)
        self.valid_attri_data, self.test_attri_data = train_test_split(valid_attri_data, test_size=0.5,stratify=valid_attri_data['a'],
                                                                    random_state=802)   
        #self.att2df = pd.read_pickle(ds_path + 'LitWD48K/one_attri_one_df.pkl') 

        ## Group Entities, relations, attributes
        self.entities = pd.read_csv(ds_path + 'Entities/entity_labels_en.txt', sep='\t', names=['label', 'name'])
        self.relations = pd.read_csv(ds_path + 'Relations/relation_labels_en.txt', sep='\t', names=['label', 'name'])
        self.attributes = self.attri_data['a'].unique()

        ## Dict Entites and relations
        # look like{ent:idx,rel:idx,att:idx}
        # self.dict_ent_2_idx = dict(zip(self.entities['label'], np.arange(0, len(self.entities), 1)))
        # self.dict_rel_2_idx = dict(zip(self.relations['label'], np.arange(0, len(self.relations), 1)))
        # self.dict_att_2_idx = dict(zip(self.attributes, np.arange(0, len(self.attributes), 1)))

        # ## Dict contains all Graph objects
        # self.dict_all_2_idx = {}
        # self.dict_all_2_idx.update(self.dict_ent_2_idx)
        # self.dict_all_2_idx.update(self.dict_rel_2_idx)
        # self.dict_all_2_idx.update(self.dict_att_2_idx)
        #self.dict_all_2_idx = np.load(ds_path+'files_needed/dict_all_2_idx.npy',allow_pickle=True).item()
        
        with open(ds_path+'files_needed/saved_all2idx.pkl', 'rb') as f:
            self.dict_all_2_idx = pickle.load(f)


# create data for train and valid 
    def create_triplets_data(self, rel_data):
        
        ## Dict contains all Graph objects
        # dict_all_2_idx = {}
        # dict_all_2_idx.update(self.dict_ent_2_idx)
        # dict_all_2_idx.update(self.dict_rel_2_idx)
        # dict_all_2_idx.update(self.dict_att_2_idx)
        
        ## Construct neg triplets
        list_all_rel = self.relations['label'].unique().tolist()
        idx_rel = self.dict_all_2_idx[random.choice(list_all_rel)]
        
        X_all_pos = np.empty([rel_data.shape[0], rel_data.shape[1]], dtype=int)
        for i, el in enumerate(rel_data.values):
            X_all_pos[i] = [self.dict_all_2_idx[el_] for el_ in el]
        y_all_pos = np.ones((X_all_pos.shape[0],1))
        ## Construct negative instances for all other data (Ns = 1)
        list_all_ent_j = np.unique(X_all_pos[:,2])
        ## Create dict of train instances
        dict_positive = dict()
        for el in X_all_pos:
            ent_i = el[0]
            if not(ent_i in dict_positive):
                dict_positive[ent_i] = [el[2]]
            else:
                l = dict_positive[ent_i]
                l.append(el[2])
                dict_positive[ent_i] = l
        ## Create the neg instance
        dict_neg_instances = dict()
        for key in dict_positive:
            l=list()
            for i in range(1000):
                el = random.choice(list_all_ent_j)
                if not(el in dict_positive[key]): #and not(el in dict_val_positive_user2item):
                    l.append(el)
                if len(l)==self.Ns:
                    break
            dict_neg_instances[key] = l
        ## Create X_train neg sample
        X_all_neg = np.empty([len(dict_neg_instances)*self.Ns, rel_data.shape[1]], dtype=int)
        k = 0
        for key in dict_neg_instances:
            for i in range(self.Ns):
                X_all_neg[k+i] = [key, idx_rel, dict_neg_instances[key][i]]
            k = k + self.Ns
        y_all_neg = np.zeros((X_all_neg.shape[0],1))

        ## Concatenate positive and negative instances
        X_triplets = np.concatenate((X_all_pos, X_all_neg), axis=0)
        y_triplets = np.concatenate((y_all_pos, y_all_neg), axis=0)
        
        return X_triplets, y_triplets

    def create_attr_net_data(self,rel_data):
        #要做dict:{el[0]:[attri,v],[attri,v]}
        dict_e2rv = dict()
        for el in self.train_attri_data.values:
            attri = self.dict_all_2_idx[el[1]]
            v = round(el[2],5)
            e = self.dict_all_2_idx[el[0]]
            if e in dict_e2rv:
                l = dict_e2rv[e]
                l.append([attri,v])
                dict_e2rv[e] = l
            else:
                dict_e2rv[e] = [[attri,v]]
        
        X_list_head_attr = list()
        y_list_head_attr = list()
        X_list_tail_attr = list()
        y_list_tail_attr = list()
        ##
        # X_all_pos_r: triples filter out same entities in valid and test sets
        # X_all_pos: no filter out
        #rel_data = self.train_rel_data
        test_att_all=pd.concat([self.valid_attri_data, self.test_attri_data], ignore_index=True)
        overlap = np.intersect1d(rel_data['s'].unique(),test_att_all['e'].unique())
        rel_data_nooverlap = rel_data[~rel_data['s'].isin(overlap)]
        X_all_pos_r = np.empty([rel_data_nooverlap.shape[0], rel_data_nooverlap.shape[1]], dtype=int)
        for i, el in enumerate(rel_data_nooverlap.values):
            X_all_pos_r[i] = [self.dict_all_2_idx[el_] for el_ in el]

        for i, triple in enumerate(X_all_pos_r):
            ei = triple[0]
            ej = triple[2]
            if ei in dict_e2rv:
                l_vals = dict_e2rv[ei]
                for el in l_vals:
                    ai = el[0]
                    vi = el[1]
                    X_list_head_attr.append([ei, ai])
                    y_list_head_attr.append([vi])
            if ej in dict_e2rv:
                l_vals = dict_e2rv[ej]
                for el in l_vals:
                    vj = el[1]
                    aj = el[0]
                    X_list_tail_attr.append([ej, aj])
                    y_list_tail_attr.append([vj])
        # x ([e,a]) y([v])
        X_head_attr = np.array(X_list_head_attr).reshape((len(X_list_head_attr), 2))
        X_tail_attr = np.array(X_list_tail_attr).reshape((len(X_list_tail_attr), 2))
        ##
        y_head_attr = np.array(y_list_head_attr).reshape((len(X_list_head_attr), 1))
        y_tail_attr = np.array(y_list_tail_attr).reshape((len(X_list_tail_attr), 1))
        return X_head_attr, X_tail_attr, y_head_attr, y_tail_attr
    
    def create_val_triple_data(self):
        # dict_all_2_idx = {}
        # dict_all_2_idx.update(self.dict_ent_2_idx)
        # dict_all_2_idx.update(self.dict_rel_2_idx)

        ## Construct positive instances
        X_val_pos = np.empty([self.val_rel_data.shape[0], self.val_rel_data.shape[1]], dtype=int)
        for i, el in enumerate(self.val_rel_data.values):
            X_val_pos[i] = [self.dict_all_2_idx[el_] for el_ in el]
        y_val_pos = np.ones((X_val_pos.shape[0],1))
        ## create pos dict {ent_i:[ent_j...]}
        list_val_ej = np.unique(X_val_pos[:,2])
        dict_val_pos = dict()
        for el in X_val_pos:
            pos_s = el[0]
            if not(pos_s in dict_val_pos):
                dict_val_pos[pos_s] = [el[2]]
            else:
                l = dict_val_pos[pos_s]
                l.append(el[2])
                dict_val_pos[pos_s] = l
        ### Construct negative instances        
        dict_val_neg = dict()
        all_val_rel = self.val_rel_data.p.unique().tolist()
        neg_rel = self.dict_all_2_idx[random.choice(all_val_rel)]
        for key in dict_val_pos:
            list_neg_ins = list()
            for i in range(100):
                neg_ej = random.choice(list_val_ej)
                if not(neg_rel in dict_val_pos[key]):
                    list_neg_ins.append(neg_ej)
                if len(list_neg_ins)==self.Ns:
                    break
            dict_val_neg[key] =list_neg_ins
        
        X_val_neg = np.empty([len(dict_val_neg)*self.Ns, self.val_rel_data.shape[1]], dtype=int)
        k = 0
        for key in dict_val_neg:
            for i in range(self.Ns):
                X_val_neg[k+i] = [key, neg_rel, dict_val_neg[key][i]]
            k = k + self.Ns
        y_val_neg = np.zeros((X_val_neg.shape[0],1))

        ## Concatenate positive and negative instances
        X_val_triplets = np.concatenate((X_val_pos, X_val_neg), axis=0)
        y_val_triplets = np.concatenate((y_val_pos, y_val_neg), axis=0)
        
        return X_val_triplets, y_val_triplets # return X_val_neg, y_val_neg


    
    def create_pytorch_data(self, X_triplets, y_triplets, X_head_attr, y_head_attr,
                           X_tail_attr, y_tail_attr,
                           batch_size,mode='train'):
        if mode =='test':
            x_tensor_triplets = torch.from_numpy(X_triplets)
            y_tensor_triplets = torch.from_numpy(y_triplets)
            data_triplets = TensorDataset(x_tensor_triplets, y_tensor_triplets)
            loader_triplets = DataLoader(dataset=data_triplets, batch_size=batch_size, shuffle=False)
            x_tensor_head_attr = torch.from_numpy(X_head_attr)
            y_tensor_head_attr = torch.from_numpy(y_head_attr)
            data_head_attr = TensorDataset(x_tensor_head_attr, y_tensor_head_attr)
            loader_head_attr = DataLoader(dataset=data_head_attr, batch_size=batch_size, shuffle=False)
            ##
            x_tensor_tail_attr = torch.from_numpy(X_tail_attr)
            y_tensor_tail_attr = torch.from_numpy(y_tail_attr)
            data_tail_attr = TensorDataset(x_tensor_tail_attr, y_tensor_tail_attr)
            loader_tail_attr = DataLoader(dataset=data_tail_attr, batch_size=batch_size, shuffle=False)
        else:
            x_tensor_triplets = torch.from_numpy(X_triplets)
            y_tensor_triplets = torch.from_numpy(y_triplets)
            data_triplets = TensorDataset(x_tensor_triplets, y_tensor_triplets)
            loader_triplets = DataLoader(dataset=data_triplets, batch_size=batch_size, shuffle=True)
            ##
            x_tensor_head_attr = torch.from_numpy(X_head_attr)
            y_tensor_head_attr = torch.from_numpy(y_head_attr)
            data_head_attr = TensorDataset(x_tensor_head_attr, y_tensor_head_attr)
            loader_head_attr = DataLoader(dataset=data_head_attr, batch_size=batch_size, shuffle=True)
            ##
            x_tensor_tail_attr = torch.from_numpy(X_tail_attr)
            y_tensor_tail_attr = torch.from_numpy(y_tail_attr)
            data_tail_attr = TensorDataset(x_tensor_tail_attr, y_tensor_tail_attr)
            loader_tail_attr = DataLoader(dataset=data_tail_attr, batch_size=batch_size, shuffle=True)
        
        return loader_triplets, loader_head_attr, loader_tail_attr
    
    def test_dataset(self):
        # test att data
        #self.test_attri_data['map_e'] = self.test_attri_data['e'].map(self.dict_all_2_idx)
        #self.test_attri_data['map_a'] = self.test_attri_data['a'].map(self.dict_all_2_idx)
        test_data = self.test_attri_data[['map_e','map_a']].values
        x_tensor_data = torch.from_numpy(test_data)
        test_dataset =TensorDataset(x_tensor_data)
        return test_dataset

# class LitWD(Dataset):
#     def __init___(self, triple_file, mode='train'):
#         self.triples = pd.read_csv(triple_file, sep="\t")
#         self.entities = self.triples[0].to_numpy()
#         self.attributes = self.triples[1].to_numpy()
#         self.value = self.triples[2].to_numpy(dtype=float)
#         self.numlit_matirix = num_lit
#         self.mode = mode

#     def __getitem__(self, idx):
#         self.sample = idx

#     def __len__(self):
#         return len(self.triples)


