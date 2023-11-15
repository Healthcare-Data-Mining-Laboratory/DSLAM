import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_multivisit import PatientDataset

import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from  model_state_space import mllt
from transformers import AutoTokenizer
import copy
import json
SEED = 2019 

torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
class_3 = True
loss_ratio = [1,1e-4]
num_epochs = 100
max_length = 300
BATCH_SIZE = 8
latent_ndims = 5
visit = 'twice'
weight_dir = "weights/xxx.pth"

evaluation = False
pretrained = True
Freeze = False
SV_WEIGHTS = True
logs = True

Best_Roc = 0.7
Best_F1 = 0.6

save_dir= "xx"
save_name = f"xxx"
log_file_name = f'xxx.txt'

if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
    visit = 'twice'
    weight_dir = "xxx.pth"


device1 = "cuda:1" if torch.cuda.is_available() else "cpu"

device1 = torch.device(device1)
device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0

label_tokens = ["acute and unspecified renal failure",
        "acute cerebrovascular disease",
        "acute myocardial infarction",
        "complications of surgical procedures or medical care",
        "fluid and electrolyte disorders",
        "gastrointestinal hemorrhage",
        "other lower respiratory disease",
        "other upper respiratory disease",
        "pleurisy, pneumothorax, pulmonary collapse",
        "pneumonia except that caused by tuberculosis or sexually transmitted disease",
        "respiratory failure insufficiency arrest",
        "septicemia except in labor",
        "shock",
        "chronic kidney disease",
        "chronic obstructive pulmonary disease and bronchiectasis",
        "coronary atherosclerosis and other heart disease",
        "diabetes mellitus without complication",
        "disorders of lipid metabolism",
        "essential hypertension",
        "hypertension with complications and secondary hypertension",
        "cardiac dysrhythmias",
        "conduction disorders",
        "congestive heart failure; nonhypertensive",
        "diabetes mellitus with complications",
        "other liver diseases"]


def logistic_func(x):
    return 1 / (1 + torch.exp(-x))
def beta_func(a, b):
    return (torch.lgamma(a) + torch.lgamma(b)-torch.lgamma(a+b)).exp()
def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]
    seq_mask = attention_mask[:,[-1]]
    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - len(input_ids[0])
    padding_ids = torch.ones((1,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((1,sentence_difference), dtype = torch.long).to(device)

    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec


def collate_fn(data):    
    text_list = [d[0] for d in data]
    label_list = [d[1] for d in data]
    return text_list,label_list



def ELBO(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post):
    KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1))
    return KLD


def fit(epoch,model,cluster_loss,y_bce_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc,prior_alpha,prior_beta
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    cluster_loss.to(device)

    eopch_loss_list = []
    epoch_cls_loss_list = []
    epoch_cluster_loss_list = []
    epoch_elbo_loss_list = []

    cluster_id_list = []

    y_list = []
    pred_list_f1 = []
    pred_list_roc = []

    for i,(text_list,label_list) in enumerate(tqdm(dataloader)):
        # if i == 10: break
        optimizer.zero_grad()
        batch_cls_list = torch.zeros(len(text_list)).to(device)
        batch_tansl_list = torch.zeros(len(text_list)).to(device)
        pi_list = []

        if flag == "train":
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    for p in range(len(text_list)):
                        p_text = text_list[p]
                        p_label = label_list[p]
                        Ztd_zero = torch.randn((1, 384)).to(device)
                        Ztd_zero.requires_grad = True
                        cls_loss = torch.zeros(len(p_text)).to(device)
                        tans_loss = torch.zeros(len(p_text)).to(device)
                
                        Ztd_last = Ztd_zero
                        Ztd_list = [Ztd_zero]
                        for v in range(len(p_text)):
                            text = p_text[v]
                            label = p_label[v]

                            label_ids =  tokenizer(label_tokens, return_tensors="pt",padding=True,max_length = max_length).to(device)
                            label = torch.tensor(label).to(torch.float32).to(device)
                            text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                
                            if text['input_ids'].shape[1] > max_length:
                                text = clip_text(BATCH_SIZE,max_length,text,device)
                            elif text['input_ids'].shape[1] < max_length:
                                text = padding_text(BATCH_SIZE,max_length,text,device)
                            if v == 0:
                                Ztd_last = Ztd_zero
                            y,Ztd_mean_post,Ztd_logvar_post,Ztd_mean_priori,Ztd_logvar_priori = model(Ztd_list,Ztd_last,text,label_ids,flag)
                            Ztd_last = Ztd_mean_post
                            Ztd_list.append(Ztd_last)
                            s_cls =  y_bce_loss(y.squeeze(),label.squeeze())
                            if v == 0:
                                elbo_loss = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)
                            else:
                                elbo_loss = ELBO(Ztd_mean_priori,Ztd_logvar_priori,Ztd_mean_post, Ztd_logvar_post)
                            cls_loss[v] = s_cls
                            tans_loss[v] = elbo_loss
                            y_list.append(label.cpu().data.tolist())
                            pred_list_roc.append(y.squeeze().cpu().data.tolist())
                            pred = np.array(y.cpu().data.tolist())
                            y = np.array(label.cpu().data.tolist())
                            pred=(pred > 0.5)*1
                            pred_list_f1.append(pred)
                        cls_loss_p = cls_loss.view(-1).mean()
                        tans_loss_p = tans_loss.view(-1).mean()

                        batch_cls_list[p] = cls_loss_p
                        batch_tansl_list[p] = tans_loss_p

                    batch_cls_list = batch_cls_list.view(-1).mean()
                    batch_tansl_list = batch_tansl_list.view(-1).mean()
                    loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_tansl_list
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    eopch_loss_list.append(loss.cpu().data )  
                    epoch_cls_loss_list.append(batch_cls_list.cpu().data) 
                    epoch_elbo_loss_list.append(batch_tansl_list.cpu().data) 

        else:
            with torch.no_grad():
                    for p in range(len(text_list)):
                        p_text = text_list[p]
                        p_label = label_list[p]
                        Ztd_zero = torch.randn((1, 384)).to(device)
                        Ztd_zero.requires_grad = True
                        cls_loss = torch.zeros(len(p_text)).to(device)
                        tans_loss = torch.zeros(len(p_text)).to(device)
                
                        Ztd_last = Ztd_zero
                        Ztd_list = [Ztd_zero]
                        for v in range(len(p_text)):
                            text = p_text[v]
                            label = p_label[v]
                            label_ids =  tokenizer(label_tokens, return_tensors="pt",padding=True,max_length = max_length).to(device)
                            label = torch.tensor(label).to(torch.float32).to(device)

                            text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                
                            if text['input_ids'].shape[1] > max_length:
                                text = clip_text(BATCH_SIZE,max_length,text,device)
                            elif text['input_ids'].shape[1] < max_length:
                                text = padding_text(BATCH_SIZE,max_length,text,device)
                            if v == 0:
                                Ztd_last = Ztd_zero
                            pred,Ztd_mean_post,Ztd_logvar_post,Ztd_mean_priori,Ztd_logvar_priori = model(Ztd_list,Ztd_last,text,label_ids,flag)
                            Ztd_last = Ztd_mean_post
                            Ztd_list.append(Ztd_last)
                            s_cls =  y_bce_loss(pred.squeeze(),label.squeeze())
                            if v == 0:
                                elbo_loss = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)
                            else:
                                elbo_loss = ELBO(Ztd_mean_priori,Ztd_logvar_priori,Ztd_mean_post, Ztd_logvar_post)
                            cls_loss[v] = s_cls
                            tans_loss[v] = elbo_loss

                            y_list.append(label.cpu().data.tolist())
                            pred_list_roc.append(pred.squeeze().cpu().data.tolist())
                            label = np.array(label.cpu().data.tolist())
                            pred = np.array(pred.cpu().data.tolist())

                            pred = (pred > 0.5) 
                            pred_list_f1.append(pred)

                        cls_loss_p = cls_loss.view(-1).mean()
                        tans_loss_p = tans_loss.view(-1).mean()

                        batch_cls_list[p] = cls_loss_p
                        batch_tansl_list[p] = tans_loss_p

                    batch_cls_list = batch_cls_list.view(-1).mean()
                    batch_tansl_list = batch_tansl_list.view(-1).mean()

                    loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_tansl_list
                    eopch_loss_list.append(loss.cpu().data )  
                    epoch_cls_loss_list.append(batch_cls_list.cpu().data) 
                    epoch_elbo_loss_list.append(batch_tansl_list.cpu().data) 

    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)
    pred_list_roc = np.vstack(pred_list_roc)
    acc = metrics.accuracy_score(y_list,pred_list_f1)

    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_roc,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_roc,average="macro")
    
    total_loss = sum(eopch_loss_list) / len(eopch_loss_list)
    total_cls_loss = sum(epoch_cls_loss_list) / len(epoch_cls_loss_list)
    total_elbo_loss = sum(epoch_elbo_loss_list) / len(epoch_elbo_loss_list)

    print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC : {} | ACC: {}| CLS LOSS  : {} | ELBO LOSS : {} Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,acc,total_cls_loss,total_elbo_loss,total_loss))

    if flag == 'test':

        if logs:
            with open(f'{log_file_name}', 'a+') as log_file:
                log_file.write("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC : {} | ACC: {}| CLS LOSS  : {} | ELBO LOSS : {} Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,acc,total_cls_loss,total_elbo_loss,total_loss)+'\n')
                log_file.close()
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"xxx.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"xxx.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)

    return model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro

if __name__ == '__main__':

    train_dataset = PatientDataset(f"dataset/", class_3 = class_3, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dataset/",class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model = mllt(class_3,latent_ndims)

    if pretrained:
        print(f"loading weights: {weight_dir}")
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 0:
                for param in child.parameters():
                    param.requires_grad = False


    cluster_loss = nn.KLDivLoss(reduction='sum')
    y_bce_loss = nn.BCELoss()

    if evaluation:

        model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(1,model,cluster_loss,y_bce_loss,testloader,optimizer,flag='test')

    else:
        for epoch in range(start_epoch,num_epochs):

            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,cluster_loss,y_bce_loss,trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,cluster_loss,y_bce_loss,testloader,optimizer,flag='test')



   

 







