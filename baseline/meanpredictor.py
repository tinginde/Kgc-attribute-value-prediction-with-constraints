''' 
Baseline model1: given one attribute, use the sample mean of the attribute specific 
training data as predictor for missing value

Baseline model2: given one attribute and one entity, use the sample mean of the attribute 
which belongs to the entities connected to this given entity
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random
import pickle

# pred = sum(a.v)/len(a)
# a is from training sample 

df = pd.read_csv('LiterallyWikidata/files_needed/numeric_literals_ver06')
df = df[['e','a','std_v']]
# get proper idx
ent2idx={v:k for k,v in enumerate(df["e"].unique())}
att2idx={v:k for k,v in enumerate(df["a"].unique())}

num_lit = np.zeros([len(ent2idx), len(att2idx)], dtype=np.float32)
count = np.zeros(len(att2idx))
for i, (e, a, lit) in enumerate(df.values):
    try:
        num_lit[ent2idx[e], att2idx[a]] = lit
        count[att2idx[a]] += 1
    except KeyError:
        continue

sum_attr = np.sum(num_lit, axis=0)
avg_attr = sum_attr / count   

def baseline1(test_path):
    
    # the mean of each attribute

    test_df = pd.read_csv(test_path)
    att_test_idx = test_df["a"].map(att2idx).tolist()
    pred = [avg_attr[idx] for idx in att_test_idx]
    target = test_df["v"].tolist()
    return target, pred

# # implement baseline2
# ent2type={}
# with open('valuepredic/entity_types.txt','r') as f:
#   for line in f.readlines():
#     id, e_type = line.strip().split('\t')
#     ent2type[id.strip()] = e_type.strip()
# idx2type={k:v for k,v in enumerate(set(ent2type.values()))}


# test_df = pd.read_csv('valuepredic/thesis_data/test.txt',sep='\t')
# test_aidx = test_df["a"].map(att2idx).tolist()
# test_type = test_df["e"].map(ent2type).tolist()

 
# with open('valuepredic/types_dic', 'rb') as f:
#     typedata = pickle.load(f)
    

# pred=np.zeros(len(test_df))
# for i in range(len(test_type)):
#     try:
#         df_type=typedata[idx2type[i]]

#         onetype_atts = df_type.groupby('a',as_index=False)
#         mean_type_atts = onetype_atts.mean()
#         #print(i, test_df.loc[i]["a"], mean_type_atts['a'].values)
#         if test_df.loc[i]["a"] in mean_type_atts['a'].values:
#             att1_index = list(mean_type_atts['a'].values).index(test_df.loc[i]["a"])
#             x = mean_type_atts.loc[att1_index]["v"]
#             pred[i]=x
#         else:
#             pred[i]=avg_attr[test_aidx[i]]
#     except KeyError: 
#         pred[i]=avg_attr[test_aidx[i]]
#         continue
            
# target = test_df["v"]

# # mse = mean_squared_error(target, pred)
# # rmse = mse **0.5


# #test_df.to_csv('valuepredic/result/b2_pred',sep='\t')

# print(f"total df: {len(df)},{len(ent2idx)},{len(att2idx)}")
t, p=baseline1("LiterallyWikidata/files_needed/numeric_literals_ver06")
print(t,p,sep='\t')
# test_df['b1_pred']= p
# test_df['b2_pred']= pred
# test_df.to_csv('valuepredic/result/b_pred',sep='\t')

# print(f'baseline1:mse {a}, rmse {b}')
# print(f'baseline2:mse {mse}, rmse {rmse}')

