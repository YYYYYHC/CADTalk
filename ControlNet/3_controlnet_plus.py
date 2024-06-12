from share import *
import config
import os
import json
from tqdm import tqdm
import cv2
import einops
import gradio as gr
import numpy as np
import torch
from PIL import Image
import random
import pdb
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from annotator.zoe import ZoeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

kernel_15 = np.zeros((9,9))
kernel_15[0,0] = 1/4
kernel_15[0,-1] = 1/4
kernel_15[-1,0] = 1/4
kernel_15[-1,-1] = 1/4
preprocessor = None

model_name = 'control_v11f1p_sd15_depth'
model = create_model(f'/mnt/yhc/controlnetModels/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('/mnt/yhc/controlnetModels/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict('/mnt/yhc/controlnetModels/control_v11f1p_sd15_depth.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(rz, det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    global preprocessor
    
    if det == 'Depth_Midas':
        if not isinstance(preprocessor, MidasDetector):
            preprocessor = MidasDetector()
    if det == 'Depth_Zoe':
        if not isinstance(preprocessor, ZoeDetector):
            preprocessor = ZoeDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # pdb.set_trace()
    
    return [detected_map] + results

def GM_filter(imgpath, use_filter=True):
    image = cv2.imread(imgpath,  cv2.IMREAD_GRAYSCALE)
    if not use_filter:
        return image
    kernel = np.ones((5, 5), np.uint8)  # 这里创建了一个5x5的矩形核
    dilation = cv2.dilate(image, kernel, iterations=5)
    if '.jpg' in imgpath:
        cv2.imwrite(imgpath.replace('.jpg', '_dilation.jpg'), dilation)
    elif '.png' in imgpath:
        cv2.imwrite(imgpath.replace('.png', '_dilation.png'), dilation)
    
    return dilation
def fileter(imgpath, use_filter = True):
    
    img_raw = cv2.imread(imgpath)
    
    if not use_filter:
        return img_raw
    noneblack_area_raw = np.where(np.sum(img_raw,axis=2)!=0)
    img = img_raw
    for i in range(40):
        # noneblack_area = np.where(np.sum(img,axis=2)!=0)
        img = cv2.filter2D(img, -1, kernel_15 )
        # img = cv2.filter2D(img, -1, kernel)
        img[noneblack_area_raw[0], noneblack_area_raw[1]] = img_raw[noneblack_area_raw[0], noneblack_area_raw[1]]
    img[noneblack_area_raw[0], noneblack_area_raw[1]] = img_raw[noneblack_area_raw[0], noneblack_area_raw[1]]
    print(imgpath)
    cv2.imwrite(imgpath.replace('.jpg', '_mask.png'), img)
    return img

USE_FILTER = False
def process_2(det, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, depth_folder_path):
    # depth_img_path = '/root/autodl-tmp/airplane0914/correct_depth_nofilter'
    depth_img_path = depth_folder_path
    
    # for f in tqdm(os.listdir(depth_img_path)):
    print('processing',depth_img_path)

    for ry in [i*36 for i in range(10)]:
        # save_path = os.path.join(depth'/root/autodl-tmp/0717car', f'rz={rz}')
        seed = random.randint(1111,111111)
        save_path = os.path.join(depth_img_path, f'rz={ry}')
        dip = os.path.join(depth_img_path, 'rz={}'.format(ry),f'{ry}.jpg')
        # di = Image.open(dip)
        # input_depth_image = np.array(di)
        input_depth_image = GM_filter(dip,USE_FILTER)
        results = process(ry, det, input_depth_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)[1:]
        for i in range(len(results)):
            timg = Image.fromarray(results[i])
            timg.save(os.path.join(save_path,'gen{}.png'.format(i)))
    return 

if __name__ == '__main__':
    det = 'None'
    config_path = '/home/cli7/yhc_Workspace/SPA/real_exp/config.json'
    with open(config_path, 'r') as file:
    # 读取文件并将JSON转换为字典
        config_data = json.load(file)
    instance_name = config_data['world_info']['current_instance']
    depth_folder_path = os.path.join(config_data["world_info"]["depth_folder_base_path"],config_data[instance_name]["cube_name"] )
    # print(depth_folder_path)
    # det = 'Depth_Zoe'
    typename = config_data[instance_name]["type"]
    prompt = f'{typename}'
    a_prompt = 'best quality'
    n_prompt = 'lowres, holes, bad anatomy, bad hands, cropped, worst quality'
    num_samples = 4
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 40
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = 111111
    eta = 1
    process_2(det, prompt, a_prompt, n_prompt, num_samples,\
    image_resolution, detect_resolution, ddim_steps, guess_mode, \
    strength, scale, seed, eta, depth_folder_path)
