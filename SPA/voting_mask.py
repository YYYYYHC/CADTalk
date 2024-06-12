import numpy as np
import os
import pdb
from tqdm import tqdm
from pc import program_controller

def type2parts_gt(type):
    if type == 'airplane':
        return ['wing','body','engine','tail']
    elif type == 'chair':
        return ['arm','seat','back','leg']
    elif type == 'table':
        return ['top','leg']
    elif type == 'animal':
        return ['head','body','leg','tail']

BLOCK_NUM=300
WRONG_LIST=[]
IOU_BAR = 0
IOB_BAR = 0.9
CONFIDENCE_BAR = 0.2
PROMPTS=['top','leg']
LABEL_NUM=len(PROMPTS)
CUBE_ID='0001'
airplane_parts = [
    "Body",
    "Wings",
    "Tail",
    "Engines",
    "Wheels",
    "Pilot's Area",
    "Wing Extensions",
    "Wing Flaps",
    "Engine Covers",
    "Fuel Storage"
]
airplane_parts_gt = [
    'body', 'wing','tail','engine'
]
table_parts = ['top', 'legs', 'supports']
table_parts_gt = ['top','leg']

chair_parts = ["seat", "backrest", "legs", "armrests"]
chair_parts_gt = ['arm','seat','back','leg']

dog_parts = [
    "head",
    "body",
    "legs",
    "tail",
    "paws",
    "ears",
    "eyes",
    "nose",
    "mouth"
]
dog_parts_gt = ['head','body','leg','tail']
airplane_parts_llama2 = [
    "Wings" ,
    "Fuselage" ,
    "Tail",
    "Engines" ,
    "Landing Gear",
    "Control Surfaces"
]
airplane_parts = [
    "Body",
    "Wings",
    "Tail",
    "Engines",
    "Wheels",
    "Pilot's Area",
    "Wing Extensions",
    "Wing Flaps",
    "Engine Covers",
    "Fuel Storage"
]
airplane_parts_llama2 = [
    "Wings" ,
    "Fuselage" ,
    "Tail",
    "Engines" ,
    "Landing Gear",
    "Control Surfaces"
]
table_parts = ['top', 'legs', 'supports']
table_parts_llama2 = ['top surface', 'legs', 'frame', 'aprons', 'support braces', 'feet']
chair_parts_llama2 = ['seat', 'backrest', 'armrests', 'legs', 'frame', 'upholstery']


dog_parts_llama2 = [
 "Body", "Head", "Neck", "Legs", "Tail", "Ears", "Eyes", "Mouth", "Nose", "Paws"]


CODE_DIR = '/home/cli7/yhc_Workspace/data/dataset_V2/table_C'
PRED_DIR = '/mnt/yhc/prediction_res/tableV2cllama2'
MINF_DIR = '/mnt/yhc/multiview_info/table_c_V2'
SAVE_DIR = '/home/cli7/yhc_Workspace/label_res_llama2/tablev2c'
# SAVE_DIR = '/home/cli7/yhc_Workspace/label_res_ablation_singleImg/tableV2c'
part_lt = table_parts_llama2
os.makedirs(SAVE_DIR, exist_ok=True)
USE_SINGLE_IMG=False
USE_no_SAM=False

def label2name(type,label):
    if type=='airplane':
        if label ==1:
            return 'body'
        elif label ==2:
            return 'wing'
        elif label ==3:
            return 'tail'
        elif label ==4:
            return 'engine'
    if type=='chair':
        if label==1:
            return 'arm'
        if label==2:
            return 'seat'
        if label==3:
            return 'back'
        if label==4:
            return 'leg'
    if type=='table':
        if label==1:
            return 'top'
        if label==2:
            return 'leg'
        

def merge_2group(g1,gc1,g2,gc2):
    g1[np.where(gc1<gc2)] = g2[np.where(gc1<gc2)]
    gc1[np.where(gc1<gc2)] = gc2[np.where(gc1<gc2)]
    return g1, gc1

def apply_grouping_res(program_path, groups, group_confidence,cubename):
    confidence_bar = np.mean(group_confidence)
    pc = program_controller(program_path=program_path)
    
    pc.init_blocks()
    global CURRENT_blocks
    if CURRENT_blocks == -1:
        return
    with open(program_path, 'r') as f:
        all_prog = f.read()
        type = 'table'
        CLASSES_wrong=type2parts_gt('table')
        CLASSES_right = [gt for gt in CLASSES_wrong if gt in all_prog and 'one' not in gt]
        mapC = {}
        for i, right_class in enumerate(CLASSES_right):
            mapC[CLASSES_wrong[i]] = right_class
    
    for bi, gi in enumerate(groups[0:CURRENT_blocks]):
        if gi!=-1:
            np.random.seed(np.int32(gi)+1)
            color = np.random.rand(3)
            
            
            pc.change_block_color(bi, (color[0],color[1], color[2]))
            # print(mapC)
            pc.add_caption_to_block(bi, str(PROMPTS[int(gi)]))
        else:
            if bi>= len(pc.blocks):
                continue
            pc.change_block_color(bi,(0,0,0))
    pc.save_res(f'{SAVE_DIR}/{cubename}')
    
def get_group_from_score(score_matrix):
    #given a score matrix L * B, get the label for all blcoks 
    max_scores = np.max(score_matrix, axis=0)
    
    sum_scores = np.sum(score_matrix, axis=0)
    
    max_idxs = np.argmax(score_matrix, axis=0)
    
    groups = -1 * np.ones(len(max_idxs))
    groups[np.where(max_scores>0)] = max_idxs[np.where(max_scores>0)]
    
    return groups, max_scores, sum_scores

def get_score_mask(mask, score, blockarea, pixel2block, score_matrix):
    #one box is one 'grouping with label'
    #confidence of the grouping is measured by
    # 1. confidence of the box
    # 2. IoU of the covered part(block) with its whole scheme
    mask_score = np.zeros(score_matrix.shape[1])
    #get confidence of grouping in this box
    pixels_i = pixel2block[mask]
    idxs_i, areas_i = np.unique(pixels_i, return_counts=True)
    idxs_i = np.int32(idxs_i)[1:]
    areas_i = areas_i[1:]
    areas_u = np.sum(mask) + blockarea[idxs_i] - areas_i
    # pdb.set_trace()
    # IoUs = areas_i/blockarea[idxs_i]
    IoUs = areas_i/areas_u
    IoBs = areas_i/blockarea[idxs_i]
    IoUs[np.where(IoUs < IOU_BAR)] = 0
    IoBs[np.where(IoBs < IOB_BAR)] = 0
    
    # print(IoUs,'\n', IoBs)
    confidence = IoUs * score.item()
    # pdb.set_trace()
    # confidence[np.where(confidence)]
    mask_score[idxs_i] = confidence
    return mask_score

def get_score_box(box, score, blockarea, pixel2block, score_matrix):
    #one box is one 'grouping with label'
    #confidence of the grouping is measured by
    # 1. confidence of the box
    # 2. IoU of the covered part(block) with its whole scheme
    box_score = np.zeros(score_matrix.shape[1])
    
    #get confidence of grouping in this box
    x0,y0,x1,y1 = np.int32(box)
    box_area = (x1-x0) * (y1 - y0)
    
    pixels_i = pixel2block[x0:x1, y0:y1]
    idxs_i, areas_i = np.unique(pixels_i, return_counts=True)
    idxs_i = np.int32(idxs_i)[1:]
    areas_i = areas_i[1:]
    areas_u = box_area + blockarea[idxs_i] - areas_i
    
    IoUs = areas_i/blockarea[idxs_i]
    
    IoUs[np.where(IoUs < IOU_BAR)] = 0
    
    confidence = IoUs * score.item()
    box_score[idxs_i] = confidence
    return box_score

CURRENT_blocks = -1
def get_score_oneView(pixel2code, predicted_info, mask_info):
    pixel2block = pixel2code['pixel2block']
    blockarea = pixel2code['blockarea']
    blocknums = pixel2code['blocknums']
    global CURRENT_blocks 
    CURRENT_blocks= blocknums
    boxes = predicted_info['boxes']
    scores = predicted_info['scores']
    labels = predicted_info['labels']
    masks = mask_info
    # PROMPTS = ['top','leg']
    if None in labels:
        pdb.set_trace()
    labels_idxs = [PROMPTS.index(l) for l in labels if l is not None]
    # assert BLOCK_NUM==blocknums, 'block number error'
    # BLOCK_NUM = blocknums
    score_matrix = np.zeros([LABEL_NUM, BLOCK_NUM])
    #score_matrix[l,b] = confidence of block b to have the label l
    #confidence score = IoU * confidence
    
    #compute confidence wihin all boxes
    for id, score, box, mask in zip(labels_idxs, scores, boxes, masks):
        if score < CONFIDENCE_BAR:
            continue
        box_score = get_score_box(box, score, blockarea, pixel2block, score_matrix)
        
        mask_score = get_score_mask(mask, score, blockarea, pixel2block, score_matrix)
        # score_matrix[id] = box_score
        if USE_no_SAM:
            score_matrix[id] = 0*mask_score + box_score
        else:
            score_matrix[id] = mask_score + 0*box_score
    groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
    # pdb.set_trace()
    # apply_grouping_res('/root/programanalysis/OpenSCADPrograms/motor_plain2.scad',groups, grouping_confidence)
    return groups, grouping_confidence, score_matrix
    
    
def group_1stage(rz, genn,cubename):
    pixel2code_path = f'{MINF_DIR}/{cubename}/ry={rz}/pixel2block.npy'
    predicted_info_path = f'{PRED_DIR}/{cubename}/ry=f{rz}/gen{genn}.npy'
    mask_predicted_path = f'{PRED_DIR}/{cubename}/ry=f{rz}/gen{genn}_mask.npy'
    if not os.path.exists(pixel2code_path):
        return 0,0,0
    if not os.path.exists(predicted_info_path):
        return 0,0,0
    if not os.path.exists(mask_predicted_path):
        return 0,0,0
    pixel2code = np.load(pixel2code_path, allow_pickle=True)
    predicted_info = np.load(predicted_info_path, allow_pickle=True)
    mask_info = np.load(mask_predicted_path, allow_pickle=True)
    if pixel2code.item()['pixel2block'].shape[0]!=512:
        WRONG_LIST.append(cubename)
        return 0,0,0
    
    groups, grouping_confidence, score_matrix = get_score_oneView(pixel2code.item(), predicted_info.item(), mask_info)
    return groups, grouping_confidence, score_matrix

def group_2stage(rz,cubename):
    
    score_matrix = np.zeros([LABEL_NUM,BLOCK_NUM])
    for genn in range(4):
        g, gc,sm = group_1stage(rz, genn,cubename)
        
        score_matrix+=sm
        groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
        if USE_SINGLE_IMG:
            break
    return groups, grouping_confidence, score_matrix

def group_3stage(cubename):
    score_matrix = np.zeros([LABEL_NUM,BLOCK_NUM])
    for rz in [i*36 for i in range(10)]:
        g, gc,sm = group_2stage(rz,cubename) 
        score_matrix+=sm  
        groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
        
        # pdb.set_trace()
        
    apply_grouping_res(f'{CODE_DIR}/{cubename}',groups, grouping_confidence, cubename)



for cubename in tqdm(os.listdir(CODE_DIR)):
    print(cubename)
    cubename_0 = cubename.split('.')[0]
    if 'assembly' in cubename:
        cubename_0 = cubename_0.replace('predict_assembly_cube_', 'cube_1_')
    if not os.path.exists(f'{PRED_DIR}/{cubename}'):
        continue
    print(cubename_0)
    # PROMPTS=[label2name('airplane',i) for i in cubelabels]
    
    PROMPTS=[name for name in part_lt]
    # PROMPTS = ['top','leg']
    LABEL_NUM=len(PROMPTS)
    group_3stage(cubename)
        
np.save('wronglist.npy', WRONG_LIST)