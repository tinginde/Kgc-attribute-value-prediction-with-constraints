import pandas as pd
import numpy as np
import pickle

#path =  "/projekte/tcl/tclext/kgc_chu/"

# get all index
entities = pd.read_csv('LiterallyWikidata/Entities/entity_labels_en.txt', sep='\t', names=['label', 'name'])
attri_data = pd.read_csv('LiterallyWikidata/files_needed/attribute.txt', names=['label'])
relations = pd.read_csv('LiterallyWikidata/Relations/relation_labels_en.txt', sep='\t', names=['label', 'name'])

ent2idx = {v:k for k,v in enumerate(entities['label'].unique())}
att2idx = {v:k for k,v in enumerate(attri_data["label"].unique())}
rel2idx = {v:k for k,v in enumerate(relations['label'].unique())}

def numeric_literal_array(data,ent2idx,att2idx):
    #'LiterallyWikidata/LitWD48K/train_attri_data'
    df_all = pd.read_csv(data)
    df_all=df_all.loc[:,['e','a','v']]

    # Resulting file
    num_lit = np.zeros([len(ent2idx), len(att2idx)], dtype=np.float32)

# Create literal wrt vocab
    for i, (s, p, lit) in enumerate(df_all.values):
        try:
            num_lit[ent2idx[s], att2idx[p]] = lit
        except KeyError:
            continue
    return num_lit
    #np.save('{}_numerical_literals.npy'.format(data), num_lit)

# num_lit shape (47998, 291)

num_lit = numeric_literal_array('LiterallyWikidata/files_needed/numeric_literals_ver06', ent2idx, att2idx)
print(num_lit.shape[1])
np.save('LiterallyWikidata/files_needed/num_lit.npy',num_lit)
# print(num_lit[:,1])
# print(num_lit[0:4,:].shape)


# pop_idx = att2idx['P1082']
# gdp = att2idx['P4010']
# nominal_gdp = att2idx['P2131']
# nominal_gdp_per = att2idx['P2132']
# gdp_per = att2idx['P2299']

# print(pop_idx, gdp, gdp_per)