import torch.nn as nn
import torch
from torch.nn import functional as F, Parameter
from itertools import chain

## define ER_MLP architecture
class ER_MLP(nn.Module):
    def __init__(self, tot_entity, tot_relation, emb_size, hidden_size):
        ##
        super(ER_MLP, self).__init__()
        ##
        self.ent_embeddings = nn.Embedding(tot_entity, emb_size)
        self.rel_embeddings = nn.Embedding(tot_relation, emb_size)
        torch.normal(self.ent_embeddings.weight)
        torch.normal(self.rel_embeddings.weight)
        ##
        self.M1 = nn.Linear(emb_size, hidden_size, bias = False)
        self.M2 = nn.Linear(emb_size, hidden_size, bias = False)
        self.M3 = nn.Linear(emb_size, hidden_size, bias = False)
        ##
        self.hidden_fc = nn.Linear(hidden_size, int(hidden_size/2))
        ##
        self.hidden_fc_2 = nn.Linear(int(hidden_size/2), 1)
        #
        self.dropout = nn.Dropout(0.2)
        ##        
    def forward(self, h, r, t):
        # add hidden layer, with relu activation function
        #
        x_h = self.ent_embeddings(h)
        x_r = self.rel_embeddings(r)
        x_t = self.ent_embeddings(t)
        ##
        Tanh = torch.tanh(torch.cat(self.M1(x_h), self.M2(x_r), self.M3(x_t)),1)
        # add dropout layer
        Tanh = self.dropout(Tanh)
        #
        fc1 = self.hidden_fc(Tanh)
        #
        fc1 = torch.tanh(fc1)
        #
        fc1 = self.droput(fc1)
        #
        z = self.hidden_fc_2(fc1)
        #
        return z    

config={'input_dropout':0.5}
## define KGMTL architecture
class KGMTL(nn.Module):
    def __init__(self, tot_entity, tot_relation, tot_attribute, emb_size, hidden_size):
        '''@tot_entity: total number of entities '''
        super(KGMTL, self).__init__()
        ## 
        # self.num_entities = tot_entity
        # self.num_relations = tot_relation
        # self.num_attributes = tot_attribute

        ## Initialize Embedding layers
        self.ent_embeddings = nn.Embedding(tot_entity, emb_size, padding_idx=0)
        self.rel_embeddings = nn.Embedding(tot_relation, emb_size, padding_idx=0)
        self.att_embeddings = nn.Embedding(tot_attribute, emb_size, padding_idx=0)
        
        ## Weights init
        # nn.init.normal_(self.ent_embeddings.weight)
        # nn.init.normal_(self.rel_embeddings.weight)
        # nn.init.normal_(self.att_embeddings.weight)
        ### tail, head, relation hidden layers
        ### nn.linear(input_dim, output_dim)
        self.Mh = nn.Linear(emb_size, hidden_size, bias = False)
        self.Mr = nn.Linear(emb_size, hidden_size, bias = False)
        self.Mt = nn.Linear(emb_size, hidden_size, bias = False)
        # ### hidden layer of Structnet
        self.hidden_struct_net_fc = nn.Linear(hidden_size, 1)
        ### head att, and tail att relation hidden layers
        self.ah = nn.Linear(emb_size, hidden_size, bias = False)
        self.at = nn.Linear(emb_size, hidden_size, bias = False)
        # ### hidden layer of AttrNet
        self.hidden_attr_net_fc = nn.Linear(hidden_size*2, 1)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(config['input_dropout'])

        self.tahn = nn.Tanh()
        self.relu = nn.ReLU()

        # self.loss_rel = nn.BCELoss()
        self.criterion = nn.MSELoss()

        # ### attr_net_left
        # self.lh = nn.Linear(emb_size * 2, hidden_size, bias = False)

        # ### attr_net_right
        # self.rh = nn.Linear(emb_size * 2, hidden_size, bias = False) 


    def StructNet_forward(self, h, r, t):
        ## 1st Part of KGMTL4REC -> StructNet
        # x_h, x_r and x_t are the embeddings 
        x_h = self.ent_embeddings(h)
        x_r = self.rel_embeddings(r)
        x_t = self.ent_embeddings(t)
        # # Mh, Mr, Mt are the h,r,t hidden layers 
        # ## hidden_struct_net_fc1 is the struct net hidden layer
        struct_net_fc1 = self.relu(self.hidden_struct_net_fc(self.Mh(x_h) + self.Mr(x_r) + self.Mt(x_t)))
        pred1 = self.dropout(struct_net_fc1)

        return pred1
    
    def AttrNet_h_forward(self, h, ah):
        ## 2nd part of KGMTL4REC -> AttrNet for head entity
        x_ah = self.att_embeddings(ah)
        x_h = self.ent_embeddings(h)
        ## hidden_head_att_net_fc1 is the head attribute net hidden layer
        head_att_net_fc1 = self.relu(self.hidden_attr_net_fc(torch.cat((self.ah(x_ah), self.Mh(x_h)),1)))
        pred_h = self.dropout(head_att_net_fc1) 
        ##
        return pred_h
        
    def AttrNet_t_forward(self, t, at): 
        ## 3rd part of the NN -> AttrNet for tail entity
        x_at = self.att_embeddings(at)
        x_t = self.ent_embeddings(t)
        ## hidden_head_att_net_fc1 is the head attribute net hidden layer
        tail_att_net_fc1 = self.relu(self.hidden_attr_net_fc(torch.cat((self.at(x_at), self.Mt(x_t)),1)))
        pred_t = self.dropout(tail_att_net_fc1)  
        ##
        return pred_t

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # regularization_loss = 0
        # for param in self.parameters():
        # # 使用L2正則項
        # # regularization_loss += torch.sum(abs(param))
        #     #regularization_loss += torch.sum(param ** 2)
        #     return self.criterion(pred, target) + 0.00075 * regularization_loss
        x_constraint = torch.tensor([x[i][0]*x[i][18] for i in range(len(x))])
        x_constraint = x_constraint.to(device)          
        return self.criterion(pred, target) + 0.00075 * regularization_loss

    def forward_AST(self, batch_size):
        # ramdomly choose an atrribute (and no repeated index number)
        # output would be like: tensor([139, 101,  78, 151, 161,  71,  40, 126,   8,  96])
        weights = torch.ones(tot_attribute)
        idxs_attr = torch.multinomial(weights, num_samples=batch_size, replacement=False)

        # idxs_attr = torch.randint(self.n_num_lit, size=(m,)).cuda()
        idxs_ent = torch.randint(self.num_entities, size=(m,)).cuda()

        attr_emb = self.att_embeddings(idxs_attr)
        ent_emb = self.ent_embeddings(idxs_ent)

        inputs = torch.cat([ent_emb, attr_emb], dim=1)

        # torch.nn.Linear(2*self.emb_dim, 100),
        # torch.nn.Tanh(),
        # torch.nn.Linear(100, 1))


        pred_left = torch.relu(self.lh(inputs),1)
        pred_right = torch.relu(self.rh(inputs),1)

        ### TODO 
        # target = self.numerical_literals[idxs_ent][range(m), idxs_attr]

        return pred_left, pred_right, target