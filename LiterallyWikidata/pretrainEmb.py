import pandas as pd
import numpy as np
import torch
import torch.nn

emb_ent = torch.load('LiterallyWikidata/files_needed/pretrained_kge/pretrained_complex_entemb.pt')
print(emb_ent.shape)
list_ent_ids =[]
with open('LiterallyWikidata/files_needed/list_ent_ids.txt','r') as f:
    for line in f:
        list_ent_ids.append(line.strip())
print(len(list_ent_ids))

attri_data = pd.read_csv('LiterallyWikidata/files_needed/nogeo_df48_var')

#attri_data['e'][0] --> find list idx --> emb.ent[idx]
ent2idx = {e:i for i,e in enumerate(list_ent_ids)}
attri_data['list_idx']=attri_data['e'].map(ent2idx)


weight = emb_ent
embedding = torch.nn.Embedding.from_pretrained(weight)
input = torch.LongTensor(attri_data['list_idx'].to_numpy())
print(embedding(input))