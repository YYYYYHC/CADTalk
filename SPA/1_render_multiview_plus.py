from utils.openscad import openscad_controller as ocontroller
from utils.program import program_controller as pcontroller
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import pdb

RESOLUTION = 512
def get_img_diff(img1_path, img2_path):
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    w1, h1 = img1.shape
    w2, h2 = img2.shape
    img1 = cv.resize(img1, (h1, w1))
    img2 = cv.resize(img2, (h1, w1))
    
    diff = np.sum(img1-img2)
    return diff

class rendering_solver():
    def __init__(self, 
                 program_path, 
                 virtualfb_path, 
                 output_dir, 
                 camera_pose=(0,0,0,55,0,25,140),
                 init_xserver=True) -> None:
        self.program_path = program_path
        self.camera_pose = camera_pose
        self.ocontroller = ocontroller(virtualfb_path=virtualfb_path)

        self.pcontroller = pcontroller(program_path=self.program_path)
        self.output_dir = output_dir
        self.pixel2block = None
        if init_xserver:
            self.ocontroller.init_xserver()
        
    def prepare_init_figure(self, init_figure_name='init.png'):
        #record for SAM to use
        self.init_figure_path = os.path.join(self.output_dir, init_figure_name)
        self.ocontroller.render_png(self.program_path, output_path=self.init_figure_path, camera_pose=self.camera_pose)
    def register_pixel2block_2(self):
        #get block list
        self.pcontroller.init_blocks_plus()
        
        self.block_nums = len(self.pcontroller.blocks)
        
        #get block colors for tmp_init.png
        color_code = np.random.uniform(0.2, 0.7, [self.block_nums, 3])
        
        #register pixel to color code
        pixel2code=-1*np.ones([RESOLUTION,RESOLUTION]) #-1 for background
        for i in range(self.block_nums):
            self.pcontroller.uncomment_color_block(i)
            self.pcontroller.change_block_color(i, (0,0,0))
            for j in range(self.block_nums):
                if j==i:
                    continue
                # self.pcontroller.change_block_color(j,(0,0,0))
                self.pcontroller.comment_color_block(j)
            
            self.pcontroller.save_res(os.path.join(self.output_dir, "pos_encoded.scad"))
            self.ocontroller.render_png(os.path.join(self.output_dir, "pos_encoded.scad"), output_path=os.path.join(self.output_dir,"pos_encoded.png"),camera_pose=self.camera_pose,imgsize=(RESOLUTION,RESOLUTION))
            pos_encode_fig = cv.imread(os.path.join(self.output_dir,"pos_encoded.png"))
            pos_encode_fig = cv.cvtColor(pos_encode_fig, cv.COLOR_BGR2RGB)
            
            black_region = np.abs(np.sum(pos_encode_fig,axis=2)) < 1
            pixel2code[black_region] = i
        self.pixel2block = pixel2code
        self.blockarea = np.zeros([self.block_nums])
        bn, bs = np.unique(self.pixel2block, return_counts=True)
        self.blockarea[np.int64(bn)] = bs #get all block areas
        save_block_dict = {
            "pixel2block": self.pixel2block,
            "blockarea": self.blockarea,
            "blocknums": self.block_nums
        }
        save_dict_path = os.path.join(self.output_dir, "pixel2block.npy")
        np.save(save_dict_path, save_block_dict)
        
        print("registred")
    def register_pixel2block(self):
        #get block list
        self.pcontroller.init_blocks_plus()
        
        self.block_nums = len(self.pcontroller.blocks)
        
        #get block colors for tmp_init.png
        color_code = np.random.uniform(0.2, 0.7, [self.block_nums, 3])
        
        #register pixel to color code
        pixel2code=-1*np.ones([RESOLUTION,RESOLUTION]) #-1 for background
        for i in range(self.block_nums):
            self.pcontroller.change_block_color(i, (color_code[i][0], color_code[i][1], color_code[i][2]))
            for j in range(self.block_nums):
                if j==i:
                    continue
                self.pcontroller.change_block_color(j,(0,0,0))
            self.pcontroller.save_res(os.path.join(self.output_dir, "pos_encoded.scad"))
            self.ocontroller.render_png(os.path.join(self.output_dir, "pos_encoded.scad"), output_path=os.path.join(self.output_dir,"pos_encoded.png"),camera_pose=self.camera_pose,imgsize=(RESOLUTION,RESOLUTION))
            pos_encode_fig = cv.imread(os.path.join(self.output_dir,"pos_encoded.png"))
            pos_encode_fig = cv.cvtColor(pos_encode_fig, cv.COLOR_BGR2RGB)
            bg_color = pos_encode_fig[0,0] 
            bg_region = np.abs(np.sum(pos_encode_fig - bg_color,axis=2)) < 1
            black_region = np.abs(np.sum(pos_encode_fig,axis=2)) < 1
            interested_region = np.logical_and(np.logical_not(bg_region), np.logical_not(black_region))
            pixel2code[interested_region] = i
        self.pixel2block = pixel2code
        self.blockarea = np.zeros([self.block_nums])
        bn, bs = np.unique(self.pixel2block, return_counts=True)
        self.blockarea[np.int64(bn)] = bs #get all block areas
        save_block_dict = {
            "pixel2block": self.pixel2block,
            "blockarea": self.blockarea,
            "blocknums": self.block_nums
        }
        save_dict_path = os.path.join(self.output_dir, "pixel2block.npy")
        np.save(save_dict_path, save_block_dict)
        
        print("registred")
    
    def get_3d(self, path_3d):
        self.ocontroller.get_3d(self.program_path, output_path=path_3d)
    
    def render(self): #single edition
        #stage.1 prepare input data
        
        self.prepare_init_figure()
        self.register_pixel2block_2()
    def stop(self):
        self.ocontroller.stop_xserver()

def render_multi_view(working_dir='', program_dir ='',model_path='',virtualfb_path=''):
    init_xserver=True
    grouping_res_all = []
    for program_name in tqdm(os.listdir(program_dir)):
        os.system("rm -rf "+ os.path.join(working_dir,program_name))
        os.mkdir(os.path.join(working_dir,program_name))
        for rz in [i*36 for i in range(10)]:
            camera_pose=(0,0,0,65,0,rz,500)
            # camera_pose=(0,0,0,314,rz,359,140)

            
            res_path = os.path.join(working_dir,program_name,'rz={}'.format(rz))
            gs = rendering_solver(program_path=os.path.join(program_dir,program_name),
                                        output_dir=res_path,
                                        virtualfb_path = virtualfb_path,
                                        camera_pose=camera_pose,
                                        init_xserver=init_xserver)
            if rz == 0:
                os.system("rm -rf "+os.path.join(model_path,program_name))
                os.mkdir(os.path.join(model_path,program_name))
                gs.get_3d(os.path.join(model_path,program_name,'model.stl'))
                
            os.system("rm -rf "+res_path)
            os.mkdir(res_path)
            gs.render()
            
            init_xserver=False  
        gs.stop()
        # break
    return grouping_res_all

if __name__ == "__main__":
    wd = '/home/yyyyyhc/gitCADTalk/CADTalk/examples/stage1/working_dir'
    pd = '/home/yyyyyhc/gitCADTalk/CADTalk/examples/stage1/program_dir'
    md = '/home/yyyyyhc/gitCADTalk/CADTalk/examples/stage1/model_dir'
    vfb = '/home/yyyyyhc/gitCADTalk/CADTalk/virtualfb/virtualfb.sh'
    render_multi_view(working_dir=wd, program_dir=pd, model_path=md, virtualfb_path=vfb)