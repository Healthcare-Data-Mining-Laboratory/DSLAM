import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging
from transformers import AutoTokenizer, AutoModel
from numpy.testing import assert_almost_equal
import botorch


class mllt(nn.Module):
    def __init__(self,class_3,n_tokens):
        super(mllt, self).__init__()
        self.hidden_size = 768
        self.alpha = 5
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.t_tok = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.l_toq = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.t_value = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.drop_out1 = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.3)
        self.drop_out3 = nn.Dropout(0.3)
        self.drop_out4 = nn.Dropout(0.3)
        self.drop_out5 = nn.Dropout(0.3)
        self.drop_out6 = nn.Dropout(0.3)
        self.drop_out7 = nn.Dropout(0.3)
        self.drop_out8 = nn.Dropout(0.3)

        self.ff = nn.Sequential(
                    nn.Linear(self.hidden_size//2, self.hidden_size//2),
                    nn.PReLU(),
                    nn.Linear(self.hidden_size//2, self.hidden_size//2)
                    )
        
        self.cluster_layer = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.PReLU()
            )
        if class_3:
            self.MLPs = nn.Sequential(
                        nn.Linear(self.hidden_size//2, 100),
                        nn.Dropout(0.3),
                        nn.Linear(100, 3),
                        )
        else:
            self.MLPs = nn.Sequential(
                        nn.Linear(self.hidden_size//2, 100),
                        nn.Dropout(0.3),
                        nn.Linear(100, 25),
                        )
        self.transd = nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size= self.hidden_size//2, dropout = 0.5, num_layers=1, bidirectional=True)

        self.transd_mean = nn.Linear( self.hidden_size//2,  self.hidden_size//2)
        self.transd_logvar = nn.Linear( self.hidden_size//2,  self.hidden_size//2)

        self.transRNN =  nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size= self.hidden_size//2, dropout = 0.5, num_layers=1, bidirectional=True)

        self.zd_mean = nn.Linear( self.hidden_size//2,  self.hidden_size//2)
        self.zd_logvar = nn.Linear( self.hidden_size//2,  self.hidden_size//2)  
        self.Ztd_cat = nn.Linear(self.hidden_size, self.hidden_size//2)

      
        self.forget_gate =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            )

        self.sigmoid = nn.Sigmoid()

    def get_cluster_prob(self, embeddings,Center):
        # print(embeddings.unsqueeze(1)[[0],:,:] - Center)
        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center) ** 2, -1)

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        # print(numerator.shape)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()

    def sampling(self,mu,logvar,flag):
        if flag == "test":
            return mu
        std = torch.exp(0.5 * logvar).detach()        
        epsilon = torch.randn_like(std).detach()
        zt = epsilon * std + mu 
        return zt



    def cross_attention(self,v,c):
       
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        g = torch.bmm(v, c.transpose(-2, -1))
        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        return b
    

    def approximation(self,Ztd_list,Ot,label_token,flag):

        text_embedding = self.text_encoder(**Ot).last_hidden_state
    
        value_t = self.drop_out3(self.t_value(text_embedding))
        value_t = value_t.mean(1)

        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        Ztd_last =  torch.mean(Ztd_last,0)
        Ztd = torch.cat((Ztd_last,value_t),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)
        Ztd = self.drop_out4( self.Ztd_cat(gate_ratio_ztd*Ztd))
        Ztd_mean = self.zd_mean(Ztd)
        Ztd_logvar = self.zd_logvar(Ztd)
        Ztd = self.sampling(Ztd_mean,Ztd_logvar,flag)
        return Ztd,Ztd_mean,Ztd_logvar

    def trasition(self,Ztd_last):

        _,Ztd_last_last_hidden = self.transd(Ztd_last.unsqueeze(0))

        Ztd =  torch.mean(Ztd_last_last_hidden,0)

        Ztd_mean =   self.transd_mean(Ztd)
        Ztd_logvar =  self.transd_logvar(Ztd)

        return Ztd_mean,Ztd_logvar

    def forward(self,Ztd_list,Ztd_last,fuse_input,label_ids,flag):
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)
        Ztd,Ztd_mean_post,Ztd_logvar_post = self.approximation(Ztd_list,fuse_input,label_ids,flag)
        Ztd_mean_priori,Ztd_logvar_priori = self.trasition(Ztd_last)
        y =  self.sigmoid(self.MLPs(Ztd))
        return y,Ztd_mean_post,Ztd_logvar_post,Ztd_mean_priori,Ztd_logvar_priori


        

       

    


   