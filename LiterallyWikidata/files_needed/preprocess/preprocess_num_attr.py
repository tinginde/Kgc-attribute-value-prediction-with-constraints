import pandas as pd
import numpy as np
from tqdm import tqdm

#path =  "/projekte/tcl/tclext/kgc_chu/"

# get all index
entities = pd.read_csv('LiterallyWikidata/Entities/entity_labels_en.txt', sep='\t', names=['label', 'name'])
attri_data = pd.read_csv('LiterallyWikidata/LitWD48K/numeric_literals_final_ver04', sep='\t')
relations = pd.read_csv('LiterallyWikidata/Relations/relation_labels_en.txt', sep='\t', names=['label', 'name'])

ent2idx = {v:k for k,v in enumerate(entities['label'].unique())}
att2idx = {v:k for k,v in enumerate(attri_data["a"].unique())}
rel2idx = {v:k for k,v in enumerate(relations['label'].unique())}

def numeric_literal_array(data,ent2idx,att2idx):
    #'LiterallyWikidata/LitWD48K/train_attri_data'
    df_all = pd.read_csv(data,sep='\t')

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

num_lit = numeric_literal_array('LiterallyWikidata/LitWD48K/numeric_literals_final_ver04', ent2idx, att2idx)
print(num_lit.shape)
