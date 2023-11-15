import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
import random
from datetime import datetime
from collections import defaultdict
import string
SEED = 2019
torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True,TOKENIZERS_PARALLELISM=True)
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

class PatientDataset(object):
    def __init__(self, data_dir,class_3,visit,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.text_dir = 'dataset/brief_course/'
        self.filling_file = "missing_fill.csv"
        self.stopword = list(pd.read_csv('stopwods.csv').values.squeeze())
        self.visit = visit
        self.sbj_dir = os.path.join(f'{data_dir}',flag)
        self.sbj_list = os.listdir(self.sbj_dir)

        self.max_length = 1000
        self.class_3 = class_3

    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
    def padding_text(self,vec):
        input_ids = vec['input_ids']
        attention_mask = vec['attention_mask']
        padding_input_ids = torch.ones((input_ids.shape[0],self.max_length-input_ids.shape[1]),dtype = int).to(self.device)
        padding_attention_mask = torch.zeros((attention_mask.shape[0],self.max_length-attention_mask.shape[1]),dtype = int).to(self.device)
        input_ids_pad = torch.cat([input_ids,padding_input_ids],dim=-1)
        attention_mask_pad = torch.cat([attention_mask,padding_attention_mask],dim=-1)
        vec = {'input_ids': input_ids_pad,
        'attention_mask': attention_mask_pad}
        return vec
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)

        return temp
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def __getitem__(self, idx):
    
        patient_id = self.sbj_list[idx]
        visit_list = sorted(os.listdir(os.path.join(self.data_dir,self.flag, patient_id)), key= self.sort_key)
        label_list = []
        breif_course_list = []
        all_label_list = []

        for patient_file in visit_list:
            text_df = pd.read_csv(self.text_dir+"_".join(patient_file.split("_")[:2])+".csv").values
            cheif_complaint = text_df[:,0:1].tolist()
            cheif_complaint = [n[0] for n in cheif_complaint if not pd.isnull(n)]   
            cheif_complaint = " ".join(cheif_complaint)
            cheif_complaint = self.rm_stop_words(remove_punctuation(cheif_complaint))

            cheif_complaint_list.append(cheif_complaint)
            breif_course = text_df[:,1:2].tolist()

            breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
            text = ' '.join(breif_course)
            text = self.rm_stop_words(text)
            text_length = len(tokenizer.tokenize(text))
            breif_course_list.append(text)
            lab_dic = defaultdict(list)

            lab_description = []
            for k in lab_dic.keys():
                strs =  str(lab_dic[k][0]) + " " + str(self.feature_list[k]) 
                lab_description.append(strs.lower())

            if self.visit == 'twice':
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag+"1",patient_file))[self.label_list].values[:1,:][0])

            else:
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag,patient_file))[self.label_list].values[:1,:][0])
            all_label_list.append(label)
            cluster_label = [0,0,0]
            if self.class_3:
                if sum(label[:13]) >=1:
                    cluster_label[0] = 1
                if sum(label[13:20]) >= 1:
                    cluster_label[1] = 1
                if sum(label[20:]) >= 1:
                    cluster_label[2] = 1
                label_list.append(cluster_label)
            else:
                label_list.append(label)
        return breif_course_list,label_list,cheif_complaint_list,all_label_list


    def __len__(self):
        return len(self.sbj_list)
