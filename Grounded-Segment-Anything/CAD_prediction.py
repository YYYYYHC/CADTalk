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
table_parts = ['top', 'legs', 'supports']
chair_parts = ["seat", "backrest", "legs", "armrests"]
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

def type2parts(type):
    if type == 'airplane':
        return airplane_parts
    elif type == 'chair':
        return chair_parts
    elif type == 'table':
        return table_parts
    elif type == 'animal':
        return dog_parts

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


# Predict classes and hyper-param for GroundingDINO
CLASSES=['top of table','leg of table']
CLASSES_to_label=['top','leg']

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
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES_to_label[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    cv2.imwrite(res_path, annotated_image)

def get_seg_dir(gen_path, output_path):
    for f in os.listdir(gen_path):
        if 'CAD' in f and '.png' in f:
            get_seg(os.path.join(gen_path, f), os.path.join(output_path, f))


# def get_prediction(data_path, info_path, res_path):
    
#     # get_seg(SOURCE_IMAGE_PATH, '/root/autodl-tmp/0712/rz=0/gen1_detect.png')
#     for cube in tqdm(os.listdir(data_path)):
#     # for cube in tqdm(os.listdir('/root/autodl-tmp/airplane0914/correct_depth_nofilter')):
#         cubeid = cube.split('.')[0]
#         cubeinfo = f'{info_path}/{cubeid}.npy'
#         if not os.path.exists(cubeinfo):
#             continue
#         cubeinfo = np.load(cubeinfo, allow_pickle=True).item()
#         cubelabels = list(cubeinfo.keys())
        
#         # CLASSES_to_label=[label2name('airplane',i) for i in cubelabels]
#         CLASSES_to_label=[label2name('airplane',i) for i in cubelabels]
#         CLASSES=[f'{name} of airplane' for name in CLASSES_to_label]
#         # pdb.set_trace()
#         for ry in [i*36 for i in range(10)]:
#             gen_path = f'{data_path}/{cube}/ry={ry}'
#             output_path = f'{res_path}/{cube}/ry=f{ry}'
#             os.makedirs(gen_path, exist_ok=True)
#             os.makedirs(f'{res_path}/{cube}',exist_ok=True)
#             os.makedirs(output_path, exist_ok=True)
#             get_seg_dir(gen_path, output_path)

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
   
          
if __name__ == "__main__":
    data_path = "/mnt/diska/yhc/CAD_render/airplaneV1c"
    res_path = "/mnt/diska/yhc/prediction_res/airplaneV1cGPTCAD"
    os.makedirs(res_path, exist_ok=True)
    # get_seg(SOURCE_IMAGE_PATH, '/root/autodl-tmp/0712/rz=0/gen1_detect.png')
    for cube in tqdm(os.listdir(data_path)):
    # for cube in tqdm(os.listdir('/root/autodl-tmp/airplane0914/correct_depth_nofilter')):
        cubeid = cube.split('.')[0]
        if 'assembly' in cubeid:
            cubeid = cubeid.replace('predict_assembly_cube_', 'cube_1_')
        print(cubeid)
        CLASSES_to_label=airplane_parts
        CLASSES=[f'{name} of airplane' for name in CLASSES_to_label]
        # pdb.set_trace()
        for ry in [i*36 for i in range(10)]:
            gen_path = f'{data_path}/{cube}/ry={ry}'
            output_path = f'{res_path}/{cube}/ry=f{ry}'
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(f'{res_path}/{cube}',exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            get_seg_dir(gen_path, output_path)