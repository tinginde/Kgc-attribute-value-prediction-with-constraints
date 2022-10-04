import torch
import sys
import numpy as np

sys.path.append('./MTKGNN/KGMTL4Rec')
sys.path.append('./LiterallyWikidata')
from Data_Processing_copy_less import KGMTL_Data
from Model import KGMTL
from Evaluation import evaluation, save_result

config={'batch_size': 20, 'epochs':500}

# built init data
KGMTL_Data_local = KGMTL_Data('LiterallyWikidata/',Ns=3)
tot_entity = len(KGMTL_Data_local.entities)
tot_rel = len(KGMTL_Data_local.relations)
tot_attri = len(KGMTL_Data_local.attributes)
model = KGMTL(tot_entity, tot_rel, tot_attri , 128, 64)
print(model)
#model.load_state_dict(torch.load('MTKGNN/KGMTL4Rec/pre_saved_model/model_500_500_0.001.pt'))
model.load_state_dict(torch.load('MTKGNN/KGMTL4Rec/saved_model/ast_model_20_500_0.001.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

##****** Set Device ******
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 

device = torch.device(dev)  

model.to(device)

#preparing TEST data
# X_test_triplets, y_test_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.test_rel_data)
# print(f'test rel set: {len(X_test_triplets)}')
# X_test_head_attr, X_test_tail_attr, y_test_head_attr, y_test_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.test_rel_data)
# print(f'X_test_head_attr: {len(X_test_head_attr)}')

X_val_triplets, y_val_triplets = KGMTL_Data_local.create_triplets_data(KGMTL_Data_local.val_rel_data)
print(f'val rel set: {len(X_val_triplets)}')
X_val_head_attr, X_val_tail_attr, y_val_head_attr, y_val_tail_attr = KGMTL_Data_local.create_attr_net_data(KGMTL_Data_local.val_rel_data)
print(f'X_val_head_attr: {len(X_val_head_attr)}')

#Put data into dataloader
# test_loader_triplets, test_loader_head_attr, test_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
# X_test_triplets, y_test_triplets, 
# X_test_head_attr, y_test_head_attr, 
# X_test_tail_attr, y_test_tail_attr, config['batch_size'], mode='test')


valid_loader_triplets, valid_loader_head_attr, valid_loader_tail_attr = KGMTL_Data_local.create_pytorch_data(
X_val_triplets, y_val_triplets, 
X_val_head_attr, y_val_head_attr, 
X_val_tail_attr, y_val_tail_attr, config['batch_size'], mode='test')

# # val_loss_fn=[]
# # val_loss_mse =[]
# # model.eval()
# # for x, y in valid_loader_triplets:                         # iterate through the dataloader
# #     x, y = x.to(device), y.to(device) 
# #     with torch.no_grad(): 
# #         pred_1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
# #         loss_1 = model.loss_fn(pred_1, y)
# #     val_loss_fn.append(loss_1.detach().cpu().item()) 
# # print('Validation loss_rel {}'.format(np.mean(val_loss_fn)))

# # for x, y in valid_loader_head_attr:
# #     x, y = x.to(device), y.to(device)
# #     with torch.no_grad():
# #         pred_2 = model.AttrNet_h_forward(x[:,0], x[:,1])
# #         loss_2 = model.cal_loss(pred_2, y)
# #     val_loss_mse.append(loss_2.detach().cpu().item())
# # print('Validation loss_head {}'.format(np.mean(val_loss_mse)))



#test model
model.eval()
table = evaluation(valid_loader_triplets, valid_loader_head_attr, valid_loader_tail_attr, device , mymodel=model) 
save_result(table, 'predicted_result/valid_nonorm_epoch{}_preds_att_head.csv'.format(config['epochs']))
#save_pred(preds1, 'predicted_result/test_cons_epoch{}_preds_rel.csv'.format(config['epochs']))
#save_pred(preds2, 'predicted_result/test_cons_epoch{}_preds_att_head.csv'.format(config['epochs'])) 
#save_pred(preds3, 'predicted_result/test_cons_epoch{}_preds_att_tail.csv'.format(config['epochs']))