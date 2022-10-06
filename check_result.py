import pandas as pd
import numpy as np


with open('LiterallyWikidata/files_needed/attribute.txt','r') as fr:
    attributes = [line.strip() for line in fr] 
dict_att_2_idx = dict(zip(attributes, np.arange(0, len(attributes), 1)))

## constraint needed:
pop_idx = dict_att_2_idx ['P1082']
gdp = dict_att_2_idx ['P4010']
nominal_gdp = dict_att_2_idx ['P2131']
nominal_gdp_per = dict_att_2_idx ['P2132']
gdp_per = dict_att_2_idx ['P2299']
date_of_birth = dict_att_2_idx ['P569']
date_of_death = dict_att_2_idx ['P570']
area = dict_att_2_idx['P2046']
work_end = dict_att_2_idx['P2032']
work_start = dict_att_2_idx['P2031']
height= dict_att_2_idx['P2048']
Latitude = dict_att_2_idx['P625_Latitude']
Longtiude = dict_att_2_idx['P625_Longtiude']
list_var=[pop_idx,gdp,gdp_per,date_of_birth,date_of_death,area,work_start,work_end,height,Latitude,Longtiude] 
print('pop_idx,gdp,gdp_per,date_of_birth,date_of_death,area,work_start,work_end,height,Latitude,Longtiude',list_var)
def read_result (filepath):
    return pd.read_csv(filepath)

def var_r_df (df,var=list_var):
    #還可以增加
    list_var = var
    df = df[df.a.isin(list_var)].sort_values(by='a')
    df['mae']=abs(df.pred_h-df.target_h)
    return df

df=read_result('exp/predicted_result/minmax_ast_epoch50_batch200_preds_att_head.csv')
df=var_r_df(df)
print(df[:40])
df_preds=df[df['pred_h']!=0.0]
print(len(df_preds))
print(list(df_preds.groupby('a').pred_h.value_counts().index))
