import cv2
import numpy as np
import supervision as sv
import pdb
import os
from tqdm import tqdm
import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

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

chair_parts = ["seat", "backrest", "legs", "armrests"]
chair_parts_llama2 = ['seat', 'backrest', 'armrests', 'legs', 'frame', 'upholstery']
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

dog_parts_llama2 = [
 "Body", "Head", "Neck", "Legs", "Tail", "Ears", "Eyes", "Mouth", "Nose", "Paws"]


def type2parts(type):
    if type == 'airplane':
        return airplane_parts_llama2
    elif type == 'chair':
        return chair_parts
    elif type == 'table':
        return table_parts
    elif type == 'animal':
        return dog_parts

def type2parts_gt(type):
    if type == 'airplane':
        return ['wing','body','engine','tail']
    elif type == 'chair':
        return ['arm','seat','back','leg']
    elif type == 'table':
        return ['top','leg']
    elif type == 'animal':
        return ['head','body','leg','tail']

def label2name(type,label):

    if type=='airplane':
        return airplane_parts[label]
    if type=='chair':
        if label ==1:
            return 'back'
        if label ==2:
            return 'seat'
        if label ==3:
            return 'leg'
        if label==4:
            return 'arm'
    if type=='table':
        if label==1:
            return 'top'
        if label==2:
            return 'leg'
        
    # if type=='animal'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

CLASS=[]
CLASSES_to_label = []
# Predict classes and hyper-param for GroundingDINO

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

def get_seg(source_image_path, res_path):
    # load image
    image = cv2.imread(source_image_path)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    detections_tosave = {'boxes': detections.xyxy, 'scores': detections.confidence, 'labels':[CLASSES_to_label[i] for i in detections.class_id]}
    np.save(res_path.replace('png', 'npy'), detections_tosave)
    print(f"After NMS: {len(detections.xyxy)} boxes")

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    np.save(res_path.replace('.png', '_mask.npy'), detections.mask)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(text_scale=0.3,text_padding=5)
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES_to_label[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # save the annotated grounded-sam image
    cv2.imwrite(res_path, annotated_image)

def get_seg_dir(gen_path, output_path):
    for f in os.listdir(gen_path):
        if 'gen' in f and '.png' in f:
            # print(gen_path,f)
            if 'bf9f2ecb581640bdf91663a74ccd2338.scad/ry=36' in gen_path:
                continue
            get_seg(os.path.join(gen_path, f), os.path.join(output_path, f))

USE_GT_LABEL = False
def get_prediction_with_info(data_path, res_path, code_path, type):
    
    os.makedirs(res_path, exist_ok=True)
    # get_seg(SOURCE_IMAGE_PATH, '/root/autodl-tmp/0712/rz=0/gen1_detect.png')
    for cube in tqdm(os.listdir(data_path)):
        if '.DS' in cube:
            continue
    # for cube in tqdm(os.listdir('/root/autodl-tmp/airplane0914/correct_depth_nofilter')):
        cubeid = cube.split('.')[0]
        if 'assembly' in cubeid:
            cubeid = cubeid.replace('predict_assembly_cube_', 'cube_1_')
        print(cubeid)
        global CLASSES_to_label
        CLASSES_to_label=type2parts(type)
        global CLASSES
        CLASSES=[f'{name} of {type}' for name in CLASSES_to_label]
        program_path = f'{code_path}/{cube}'
        if not os.path.exists(program_path):
            continue
        if USE_GT_LABEL:
            with open(program_path, 'r') as f:
                all_prog = f.read()
                CLASSES_to_label=type2parts_gt(type)
                CLASSES = [f'{gt} of {type}' for gt in CLASSES_to_label if gt in all_prog and 'one' not in gt]

        # pdb.set_trace()
        for ry in [i*36 for i in range(10)]:
            gen_path = f'{data_path}/{cube}/ry={ry}'
            output_path = f'{res_path}/{cube}/ry=f{ry}'
            
            if os.path.exists(os.path.join(output_path,'gen0.png')):
                continue
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(f'{res_path}/{cube}',exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            print(output_path)
            if gen_path == '/mnt/yhc/Ellip_depth/animal_c/cube_1_0022.scad/ry=144':
                continue
            if output_path == '/mnt/diska/yhc/prediction_res/animalV2egtprompt/93_normalized.scad/ry=f144':
                continue
            if output_path == '/mnt/yhc/prediction_res/animalV2eExp2/93_normalized.scad/ry=f144':
                continue
            get_seg_dir(gen_path, output_path)
import argparse       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='接受两个文件路径作为输入。')

    # 添加文件路径参数
    parser.add_argument('data_path', type=str, help='第一个文件的路径')
    parser.add_argument('res_path', type=str, help='第二个文件的路径')

    # 解析命令行参数
    args = parser.parse_args()

    # data_path = "/mnt/yhc/cuboid_depth/chairV2_c"
    # res_path = "/mnt/yhc/prediction_res/chairV2cllama2"
    data_path = args.data_path
    res_path = args.res_path
    os.makedirs(res_path, exist_ok=True)
    # get_seg(SOURCE_IMAGE_PATH, '/root/autodl-tmp/0712/rz=0/gen1_detect.png')
    for cube in tqdm(os.listdir(data_path)):
        if '.DS' in cube:
            continue
    # for cube in tqdm(os.listdir('/root/autodl-tmp/airplane0914/correct_depth_nofilter')):
        cubeid = cube.split('.')[0]
        if 'assembly' in cubeid:
            cubeid = cubeid.replace('predict_assembly_cube_', 'cube_1_')
        print(cubeid)
        CLASSES_to_label=table_parts_llama2
        
        CLASSES=[f'{name} of table' for name in CLASSES_to_label]
        # pdb.set_trace()
        for ry in [i*36 for i in range(10)]:
            gen_path = f'{data_path}/{cube}/ry={ry}'
            output_path = f'{res_path}/{cube}/ry=f{ry}'
            
            # if os.path.exists(os.path.join(output_path,'gen0.png')):
            #     continue
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(f'{res_path}/{cube}',exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            print(output_path)
            if gen_path == '/mnt/diska/yhc/Ellip_depth/animal_c/cube_1_0022.scad/ry=144':
                continue
            if output_path == '/mnt/diska/yhc/prediction_res/animalV2egtprompt/93_normalized.scad/ry=f144':
                continue
            
            get_seg_dir(gen_path, output_path)
        