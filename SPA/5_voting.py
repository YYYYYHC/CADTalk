import numpy as np
import os
import pdb
from tqdm import tqdm
from pc import program_controller
BLOCK_NUM=106
WRONG_LIST=[]
IOU_BAR = 0
IoUU = 1
IOB_BAR = 0.9
CONFIDENCE_BAR = 0.1
PROMPTS=None
LABEL_NUM=None

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
            return 'back'
        if label==2:
            return 'seat'
        if label==3:
            return 'leg'
        if label==4:
            return 'arm'
    if type=='table':
        if label==1:
            return 'top'
        if label==2:
            return 'leg'

def merge_2group(g1,gc1,g2,gc2):
    g1[np.where(gc1<gc2)] = g2[np.where(gc1<gc2)]
    gc1[np.where(gc1<gc2)] = gc2[np.where(gc1<gc2)]
    return g1, gc1

def apply_grouping_res(program_path, groups, group_confidence,instancename, save_dir):
    confidence_bar = np.mean(group_confidence)
    pc = program_controller(program_path=program_path)
    print(program_path)
    print("haha")
    pc.init_blocks()
    
    for bi, gi in enumerate(groups):
          
        if gi!=-1:
            if bi < len(pc.blocks):
                np.random.seed(np.int32(gi)+1)
                color = np.random.rand(3)
                pc.change_block_color(bi, (color[0],color[1], color[2]))
                pc.add_caption_to_block(bi, str(PROMPTS[int(gi)]))
        else:
            continue
            pc.change_block_color(bi,(0,0,0))
    pc.save_res(f'{save_dir}/{instancename}')
    
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
    confidence = (IoUs)**(IoUU) * score.item()
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

def get_score_oneView(pixel2code, predicted_info, mask_info):
    pixel2block = pixel2code['pixel2block']
    blockarea = pixel2code['blockarea']
    blocknums = pixel2code['blocknums']
    
    boxes = predicted_info['boxes']
    scores = predicted_info['scores']
    labels = predicted_info['labels']
    # pdb.set_trace()
    masks = mask_info
    labels_idxs = [PROMPTS.index(l) for l in labels]
    # BLOCK_NUM = blocknums
    # assert BLOCK_NUM==blocknums, 'block number error'
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
        score_matrix[id] = mask_score + 0*box_score
    groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
    # pdb.set_trace()
    # apply_grouping_res('/root/programanalysis/OpenSCADPrograms/motor_plain2.scad',groups, grouping_confidence)
    return groups, grouping_confidence, score_matrix
    
    
def group_1stage(rz, genn,instancename, pixel2code_dir, predicted_info_dir):
    pixel2code_path = f'{pixel2code_dir}/rz={rz}/pixel2block.npy'
    predicted_info_path = f'{predicted_info_dir}/rz=f{rz}/gen{genn}.npy'
    mask_predicted_path = f'{predicted_info_dir}/rz=f{rz}/gen{genn}_mask.npy'
    # pdb.set_trace()
    if not os.path.exists(pixel2code_path):
        print(f'{pixel2code_path} not exist')
        return 0,0,0
    # pdb.set_trace()
    pixel2code = np.load(pixel2code_path, allow_pickle=True)
    predicted_info = np.load(predicted_info_path, allow_pickle=True)
    mask_info = np.load(mask_predicted_path, allow_pickle=True)
    if pixel2code.item()['pixel2block'].shape[0]!=512:
        print(f'{pixel2code_path} shape error')
        WRONG_LIST.append(instancename)
        return 0,0,0
    
    groups, grouping_confidence, score_matrix = get_score_oneView(pixel2code.item(), predicted_info.item(), mask_info)
    
    return groups, grouping_confidence, score_matrix

def group_2stage(rz,instancename, pixel2code_dir, predicted_info_dir):
    
    score_matrix = np.zeros([LABEL_NUM,BLOCK_NUM])
    for genn in range(4):
        g, gc,sm = group_1stage(rz, genn,instancename, pixel2code_dir, predicted_info_dir)
        print(rz,instancename, genn)    
        score_matrix+=sm
        groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
    return groups, grouping_confidence, score_matrix

def group_3stage(source_code_path, pixel2code_dir, predicted_info_dir, save_dir):
    instancename = source_code_path.split('/')[-1].split('.')[0]
    score_matrix = np.zeros([LABEL_NUM,BLOCK_NUM])
    for rz in [i*36 for i in range(10)]:
        g, gc,sm = group_2stage(rz,instancename, pixel2code_dir, predicted_info_dir) 
        score_matrix+=sm  
        groups, grouping_confidence, sum_scores = get_group_from_score(score_matrix)
        
        # pdb.set_trace()
        
    apply_grouping_res(source_code_path,groups, grouping_confidence, instancename, save_dir)

def get_labels(pc):
    gt_label_lt = []
    pred_label_lt = []
    for i in pc.blocks:
        gt_label = pc.listOfLines[i-1]
        pred_label = pc.listOfLines[i].split('//')[-1]
        if 'color' in pred_label:
            pred_label = 'none'
        print(gt_label)
        if 'translate' in gt_label:
            continue
        gt_word = gt_label.split(':')[1].replace('\n', '').replace(' ',  '')
        print(gt_word)
        pred_word = pred_label.replace('\n', '')
        
        if '+' not in gt_label:
            gt_label_lt.append(gt_word)
        pred_label_lt.append(pred_word)
    return list(set(gt_label_lt))

# for instancename in tqdm(os.listdir('/home/cli7/yhc_Workspace/SPA/test/pd')):
#     # print(instancename)
#     if not 'train' in instancename:
#         continue
#     # cubeinfo = np.load(cubeinfo, allow_pickle=True).item()

    # if 'smallcar_allnodes' in instance_name or 'penholder_allnodes' in instance_name or \
    #     'fan_allnodes' in instance_name or 'candleStand_allnodes' in instance_name or\
    #         'stamp' in instance_name:
    #     continue
    # instance_name = config_data['world_info']['current_instance']
if __name__ == '__main__':  
    source_code_path = './examples/stage0/input_codes/bike_with_holder.scad'
    pixel2code_dir = './examples/stage1/working_dir/bike_with_holder.scad'
    predicted_info_dir = './examples/stage3/bike_with_holder.scad'
    save_dir = './examples/stage4/'
    pc = program_controller(source_code_path)
    pc.init_blocks()
    # cubelabels = get_labels(pc)
    # pdb.set_trace()
    PROMPTS = ["wheel", "frame", "seat", "handlebar"]
    LABEL_NUM=len(PROMPTS)
    group_3stage(source_code_path, pixel2code_dir, predicted_info_dir, save_dir)
    # pdb.set_trace()
    np.save('wronglist.npy', WRONG_LIST)