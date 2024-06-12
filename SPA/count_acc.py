from pc import program_controller
import pdb
import os
import csv
import fasttext
import numpy as np
# import clip
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, transform = clip.load("ViT-B/32", device=device)

dog_syn = {"head": "head",
    "body":"body",
    "legs":"leg",
    "tail":"tail",
    "paws":"leg",
    "ears":"head",
    "eyes":"head",
    "nose":"head",
    "mouth":"head",
    "none": "none"}

airplane_syn = {
    "Body": "body",
    "Wings": "wing",
    "Tail": "tail",
    "tail":'tail',
    "Engines": "engine",
    "Wheels": "none",
    "Pilot's Area": "body",
    "Wing Extensions": "wing",
    "Wing Flaps": "wing",
    "Engine Covers": "engine",
    "Fuel Storage": "wing",
    "none": "none"
}

airplane_syn_llama2 = {
    "Wings" : "wing",
    "Fuselage" : "body",
    "Tail" : "tail",
    "Engines" : "engine",
    "Landing Gear" : None,
    "Control Surfaces" : None
}

    

chair_syn={
    "seat": "seat", 
    "backrest": "back",
    "legs" : "leg", 
    "armrests" : "arm",
    "none": "none"
}
table_syn={
    'top':'top', 
    'legs':'leg', 
    'supports': 'leg',
    'none': 'none'

}

airplane_syn_llama2 = {
    "Wings" : "wing",
    "Fuselage" : "body",
    "Tail" : "tail",
    "Engines" : "engine",
    "Landing Gear" : None,
    "Control Surfaces" : None,
    'none': 'none'
}
dog_syn_llama2 = {
    
    "Head": 'head',
    "Body": 'body',
    "Legs": 'leg',
    "Neck": 'none',
    "Tail": 'tail',
    "Ears": 'none',
    "Eyes": 'none',
    "Mouth": 'none',
    "Nose": 'none',
    "Paws": 'none',
    'none': 'none'
}
chair_syn_llama2={
    'seat': 'seat',
    'backrest': 'back',
    'armrests': 'arm',
    'legs': 'leg',
    'frame': 'none',
    'upholstery': 'none',
    'none': 'none'
}

table_syn_llama2= {
    'top surface': 'top',
    'legs': 'leg',
    'frame': 'top',
    'aprons': 'leg',
    'support braces': 'leg',
    'feet': 'leg',
    'none': 'none'
}
def calculate_label_iou(pdl_, gtl_, label):
    # 初始化交集和并集计数
    intersection = 0
    union = 0
    
    # 遍历列表，计算每个标签的交集和并集
    for pdl, gtl in zip(pdl_, gtl_):
        if pdl == gtl == label:  # 如果预测和真实标签相同且等于当前标签
            intersection += 1
            union += 1
        elif pdl == label or gtl == label:  # 如果预测或真实标签之一等于当前标签
            union += 1
    
    # 计算IoU
    iou = intersection / union if union != 0 else 0
    
    return iou

# 定义列表

def registrate_syn(pred_labels):
    pred_lt = []
    for pred_label in pred_labels:
        simscore=0
        corr_gt_label=count_syn[pred_label]
            
        pred_lt.append(corr_gt_label)
    
    return pred_lt

def get_labels(pc):
    gt_label_lt = []
    pred_label_lt = []
    for i in pc.blocks:
        gt_label = pcontroller.listOfLines[i-1]
        print(gt_label)
        pred_label = pcontroller.listOfLines[i].split('//')[-1]
        if 'color' in pred_label:
            pred_label = 'none'
            continue
        gt_word = gt_label.split(':')[1].strip().replace('\n', '')
        pred_word = pred_label.replace('\n', '')
        
        if '+' not in gt_label:
            gt_label_lt.append(gt_word)
        pred_label_lt.append(pred_word)
    return list(set(gt_label_lt)), list(set(pred_label_lt)), gt_label_lt,pred_label_lt
use_syn = True
count_syn = table_syn_llama2
BASE_PATH = '/home/cli7/yhc_Workspace/label_res_llama2/tablev1e'
output_path = '/home/cli7/yhc_Workspace/SPA/llama2exp/tablev1e.csv'

# from gensim.models import Word2Vec


csvfile = open(output_path, 'w', encoding='UTF8')
    # create the csv writer
csvwriter = csv.writer(csvfile)

header = ['cubename','acc']
csvwriter.writerow(header)
cnt = 0
full_acc=0
full_iou=0
full_blocks=0
full_progs=0
full_a_acc = 0
for file_path in os.listdir(BASE_PATH):
    
    print(file_path)
    pcontroller = program_controller(os.path.join(BASE_PATH, file_path))
    pcontroller.init_blocks()
    gts, pds,gtl,pdl = get_labels(pcontroller)
    print(gtl, pdl)
    # if 'whistle_general_allnodes' in file_path:
    
    if use_syn:
        pdl = [count_syn[item] for item in pdl]
        syn_lt = registrate_syn(pds)
    assert len(gtl) == len(pdl)
    # 获取所有唯一的标签
    all_labels = set(gtl)
    iou=0
    # 计算并打印每个标签的IoU
    for label in all_labels:
        if label != 'none' and label !='None' and 'color' not in label:  # 'none' 标签可能不需要计算IoU
            label_iou = calculate_label_iou(pdl, gtl, label)
            iou+=label_iou
    iou/=len(all_labels)
    
    acc=0
    blockcount = 0
    for i in pcontroller.blocks:
        gt_label = pcontroller.listOfLines[i-1]
        pred_label = pcontroller.listOfLines[i].split('//')[-1]
        
        if 'color([0,0,0])' in pred_label:
            pred_label = 'none'
            continue
        blockcount +=1
        if use_syn:
            # gt_word = gt_label.split(' ')[2].replace('\n', '') 
                # pdb.set_trace()
            pred_word = pred_label.replace('\n', '')
            # print(pds)
            # pdb.set_trace()
            pred_gtsyn = syn_lt[pds.index(pred_word)]
            if pred_gtsyn is None:
                continue
            if pred_gtsyn in gt_label:
                acc +=1
            # # pdb.set_trace()
            # if gt_word==pred_gtsyn:
            #     acc+=1
            # pdb.set_trace()
        else:
            # if 'tree' in file_path:
            #     pdb.set_trace()
            pred_word = pred_label.replace('\n', '')
            if pred_word in gt_label:
                acc +=1
    acc = acc
    cur_f_acc = acc/blockcount
    full_blocks += blockcount
    csvwriter.writerow([file_path, 'acc:'+str(acc/blockcount)])
    csvwriter.writerow([file_path, 'iou:'+str(iou)])
    full_acc +=acc
    full_a_acc += cur_f_acc
    full_iou +=iou
    full_progs +=1
    cnt+=1
    
# full_acc = full_acc/len(os.listdir(BASE_PATH))
csvwriter.writerow(['overall acc',full_acc/full_blocks])
csvwriter.writerow(['average acc',full_a_acc/cnt])
csvwriter.writerow(['overall iou',full_iou/full_progs])
print(full_a_acc/cnt, full_acc/full_blocks , full_iou/full_progs)