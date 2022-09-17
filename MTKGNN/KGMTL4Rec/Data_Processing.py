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




# def train_valid_split(dataset, valid_ratio, seed):
#     '''split numerical tuples into train and valid'''
#     valid_set_size = int(valid_ratio * len(dataset))
#     train_set_size = len(dataset) - valid_set_size
#     train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator = torch.Generator().manual_seed(seed))
#     return np.array(train_set), np.array(valid_set)

class KGMTL_Data():
    
    def __init__(self, ds_path, Ns):
        self.Ns = Ns
        ## Load Data for Relnet
        self.train_rel_data = pd.read_csv(ds_path + 'LitWD48K/train.txt', sep='\t', names=['s', 'p', 'o'])
        self.val_rel_data = pd.read_csv(ds_path + 'LitWD48K/valid.txt', sep='\t', names=['s', 'p', 'o'])
        self.test_rel_data = pd.read_csv(ds_path + 'LitWD48K/test.txt', sep='\t', names=['s', 'p', 'o'])
        
        ## Load Data for Attnet
        self.attri_data = pd.read_csv(ds_path + 'LitWD48K/numeric_literals_final', sep='\t')
        self.train_attri_data, valid_attri_data = train_test_split(self.attri_data, test_size=0.2,stratify=self.attri_data['a'],random_state=802)
        self.valid_attri_data, self.test_attri_data = train_test_split(valid_attri_data, test_size=0.5,stratify=valid_attri_data['a'],random_state=802)   

        ## Group Entities, relations, attributes
        self.entities = pd.read_csv(ds_path + 'Entities/entity_labels_en.txt', sep='\t', names=['label', 'name'])
        self.relations = pd.read_csv(ds_path + 'Relations/relation_labels_en.txt', sep='\t', names=['label', 'name'])
        self.attributes = self.attri_data.a.value_counts().index

        ## Dict Entites and relations
        self.dict_ent_2_idx = dict(zip(self.entities['label'], np.arange(0, len(self.entities), 1)))
        self.dict_rel_2_idx = dict(zip(self.relations['label'], np.arange(0, len(self.relations), 1)))
        self.dict_att_2_idx = dict(zip(self.attributes, np.arange(0, len(self.attributes), 1)))


    def create_triplets_data(self):
        
        ## Dict contains all Graph objects
        dict_all_2_idx = {}
        dict_all_2_idx.update(self.dict_ent_2_idx)
        dict_all_2_idx.update(self.dict_rel_2_idx)
        dict_all_2_idx.update(self.dict_att_2_idx)
        
        ## Construct neg triplets
        list_all_rel = self.relations['label'].unique().tolist()
        idx_rel = self.dict_rel_2_idx[random.choice(list_all_rel)]
        
        X_train_all_pos = np.empty([self.train_rel_data.shape[0], self.train_rel_data.shape[1]], dtype=int)
        for i, el in enumerate(self.train_rel_data.values):
            X_train_all_pos[i] = [dict_all_2_idx[el_] for el_ in el]
        y_train_all_pos = np.ones((X_train_all_pos.shape[0],1))
        ## Construct negative instances for all other data (Ns = 1)
        list_all_ent_j = np.unique(X_train_all_pos[:,2])
        ## Create dict of train instances
        dict_train_positive = dict()
        for el in X_train_all_pos:
            ent_i = el[0]
            if not(ent_i in dict_train_positive):
                dict_train_positive[ent_i] = [el[2]]
            else:
                l = dict_train_positive[ent_i]
                l.append(el[2])
                dict_train_positive[ent_i] = l
        ## Create the neg instance
        dict_neg_instances = dict()
        for key in dict_train_positive:
            l=list()
            for i in range(1000):
                el = random.choice(list_all_ent_j)
                if not(el in dict_train_positive[key]): #and not(el in dict_val_positive_user2item):
                    l.append(el)
                if len(l)==self.Ns:
                    break
            dict_neg_instances[key] = l
        ## Create X_train neg sample
        X_train_all_neg = np.empty([len(dict_neg_instances)*self.Ns, self.train_rel_data.shape[1]], dtype=int)
        k = 0
        for key in dict_neg_instances:
            for i in range(self.Ns):
                X_train_all_neg[k+i] = [key, idx_rel, dict_neg_instances[key][i]]
            k = k + self.Ns
        y_train_all_neg = np.zeros((X_train_all_neg.shape[0],1))

        ## Concatenate positive and negative instances
        X_train_triplets = np.concatenate((X_train_all_pos, X_train_all_neg), axis=0)
        y_train_triplets = np.concatenate((y_train_all_pos, y_train_all_neg), axis=0)
        
        return X_train_triplets, y_train_triplets


    def create_attr_net_data(self):
        dict_vals_a = dict()
        for el in self.train_attri_data.values:
            attri = self.dict_att_2_idx[el[1]]
            if attri in dict_vals_a:
                l = dict_vals_a[attri]
                l.append(el[2])
                dict_vals_a[attri]=l
            else:
                dict_vals_a[attri] = [el[2]]

        # 先做一個dict存每個attri fit scaler後結果
        # 有不同標準化的方式，這裡是將最小值視為0，最大值視為1，先找到每個attri值的分布情況
        dict_scaler = dict()
        for key in dict_vals_a:
            scaler = MinMaxScaler()
            X = np.array(dict_vals_a[key]).reshape((-1,1))
            scaler.fit(X)
            dict_scaler[key] = scaler
            
        dict_e2rv = dict()
        for el in self.train_attri_data.values:
            #r = self.dict_att_2_idx[el[1]]
            attri = self.dict_att_2_idx[el[1]]
            scaler_attri = dict_scaler[attri]
            v = scaler_attri.transform(np.array(el[2]).reshape((-1,1)))[0][0]
            e = self.dict_ent_2_idx[el[0]]
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
        X_train_triples, y_train_triplets = self.create_triplets_data()
        for i, triple in enumerate(X_train_triples):
            ei = triple[0]
            rk = triple[1]
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
        
        X_train_head_attr = np.array(X_list_head_attr, dtype=int).reshape((len(X_list_head_attr), 2))
        X_train_tail_attr = np.array(X_list_tail_attr, dtype=int).reshape((len(X_list_tail_attr), 2))
        ##
        y_train_head_attr = np.array(y_list_head_attr, dtype=int).reshape((len(X_list_head_attr), 1))
        y_train_tail_attr = np.array(y_list_tail_attr, dtype=int).reshape((len(X_list_tail_attr), 1))
        

        return X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr
    
    def create_val_triple_data(self):
        dict_all_2_idx = {}
        dict_all_2_idx.update(self.dict_ent_2_idx)
        dict_all_2_idx.update(self.dict_rel_2_idx)

        ## Construct positive instances
        X_val_pos = np.empty([self.val_rel_data.shape[0], self.val_rel_data.shape[1]], dtype=int)
        for i, el in enumerate(self.val_rel_data.values):
            X_val_pos[i] = [dict_all_2_idx[el_] for el_ in el]
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
        neg_rel = self.dict_rel_2_idx[random.choice(all_val_rel)]
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


    
    def create_pytorch_data(self, X_train_triplets, y_train_triplets, X_train_head_attr, y_train_head_attr,
                           X_train_tail_attr, y_train_tail_attr,
                           batch_size):
        # Wait, is this a CPU tensor now? Why? Where is .to(device)?
        x_train_tensor_triplets = torch.from_numpy(X_train_triplets)
        y_train_tensor_triplets = torch.from_numpy(y_train_triplets)
        train_data_triplets = TensorDataset(x_train_tensor_triplets, y_train_tensor_triplets)
        train_loader_triplets = DataLoader(dataset=train_data_triplets, batch_size=batch_size, shuffle=True)
        ##
        x_train_tensor_head_attr = torch.from_numpy(X_train_head_attr)
        y_train_tensor_head_attr = torch.from_numpy(y_train_head_attr)
        train_data_head_attr = TensorDataset(x_train_tensor_head_attr, y_train_tensor_head_attr)
        train_loader_head_attr = DataLoader(dataset=train_data_head_attr, batch_size=batch_size, shuffle=True)
        ##
        x_train_tensor_tail_attr = torch.from_numpy(X_train_tail_attr)
        y_train_tensor_tail_attr = torch.from_numpy(y_train_tail_attr)
        train_data_tail_attr = TensorDataset(x_train_tensor_tail_attr, y_train_tensor_tail_attr)
        train_loader_tail_attr = DataLoader(dataset=train_data_tail_attr, batch_size=batch_size, shuffle=True)
        
        return train_loader_triplets, train_loader_head_attr, train_loader_tail_attr
    
    def eval_data():
        pass