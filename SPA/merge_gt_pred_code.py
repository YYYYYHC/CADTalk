from pc import program_controller
import pdb
import os
import csv
from tqdm import tqdm
import fasttext
import numpy as np
import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)



def text_similarity(text1, text2):
    # 获取文本嵌入
    with torch.no_grad():
    # Your operations here
        text_inputs = torch.cat([clip.tokenize([text1]), clip.tokenize([text2])]).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似性
        similarity = (text_features @ text_features.T).detach().cpu().numpy()
        del text_features
    return similarity[0, 1]
def registrate_syn(gt_labels, pred_labels):
    pred_lt = []
    for pred_label in pred_labels:
        simscore=0
        corr_gt_label=None
        for gt_label in gt_labels:
            if text_similarity(gt_label, pred_label) > simscore:
                corr_gt_label = gt_label
                simscore = text_similarity(gt_label, pred_label)
        if simscore>0.75:
            pred_lt.append(corr_gt_label)
    del simscore
    return pred_lt

def get_labels(pc):
    gt_label_lt = []
    pred_label_lt = []
    for i in pc.blocks:
        gt_label = pcontroller.listOfLines[i-1]
        pred_label = pcontroller.listOfLines[i].split('//')[-1]
        if 'color' in pred_label:
            continue
        gt_word = gt_label.split(' ')[2].replace('\n', '')
        pred_word = pred_label.replace('\n', '')
        gt_label_lt.append(gt_word)
        pred_label_lt.append(pred_word)
    return list(set(gt_label_lt)), list(set(pred_label_lt))
    
BASE_PATH = '/home/cli7/yhc_Workspace/label_res_GPT/airplaneV1eGPT'
MERG_PATH = '/home/cli7/yhc_Workspace/data/dataset_V1/airplaneV1eM'
os.makedirs(MERG_PATH, exist_ok=True)
# from gensim.models import Word2Vec

use_syn = True
header = ['cubename','acc']

for file_path in tqdm(os.listdir(BASE_PATH)):   
    pcontroller = program_controller(os.path.join(BASE_PATH, file_path))
    pcontroller.init_blocks()
    if use_syn:
        gts, pds = get_labels(pcontroller)
        syn_lt = registrate_syn(gts, pds)
        if 'Fuel Storage' in pds:
            syn_lt[pds.index('Fuel Storage')]='wing'
        if 'backrest' in pds:
            syn_lt[pds.index('backrest')]='back'
    for i in pcontroller.blocks:
        torch.cuda.empty_cache()
        gt_label = pcontroller.listOfLines[i-1]
        pred_label = pcontroller.listOfLines[i].split('//')[-1]
        if use_syn:
            gt_word = gt_label.split(' ')[2].replace('\n', '')
            pred_word = pred_label.replace('\n', '')
            if 'color' in pred_label:
                continue
            print(syn_lt)
            print(pds)
            print(pred_word)
            pred_gtsyn = syn_lt[pds.index(pred_word)]
            # pdb.set_trace()
            
            if gt_word!=pred_gtsyn:
                pcontroller.listOfLines[i-1] =pcontroller.listOfLines[i-1].replace(gt_word, f'{gt_word} + {pred_gtsyn}')
                print(file_path)
            # pdb.set_trace()
    pcontroller.save_res(os.path.join(MERG_PATH, file_path))