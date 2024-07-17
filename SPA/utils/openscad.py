import os
import time
class openscad_controller():
    def __init__(self, virtualfb_path) -> None:
        """_summary_

        Args:
            virtualfb_path (str): ***/virtualfb.sh
        """
        self.virtualfb_path = virtualfb_path
        
    def init_xserver(self):
        mycmd='sudo ' + self.virtualfb_path 
        s = "".join(os.popen(mycmd).readlines())
        if 'DISPLAY=:' not in s:
            s = "".join(os.popen(mycmd).readlines())
        DISPLAY_SENTENCE = s.split(' ')[-1]
        self.DISPLAY_ID = DISPLAY_SENTENCE.replace('DISPLAY=','').replace('\n','')
        os.system("export DISPLAY=:"+self.DISPLAY_ID)
        os.environ["DISPLAY"]=self.DISPLAY_ID
        print('initialized succesfully, self display id = %s'%(self.DISPLAY_ID))

    def stop_xserver(self):
        mycmd='sudo ' + self.virtualfb_path +' stop'
        s = "".join(os.popen(mycmd).readlines())
        
    def render_png(self, program_path, output_path, imgsize=(512,512), camera_pose=(0,0,0,55,0,25,140)):
        """_summary_

        Args:
            program_path(str): ***/xx.scad
            output_path (str): ***/xx.png
            imgsize (_type_): (width, height)
            camera_pose (_type_): (translate_x,y,z,rot_x,y,z,dist)
        """
        width,height = imgsize
        translate_x,translate_y,translate_z,rot_x,rot_y,rot_z,dist = camera_pose
        mycmd = 'openscad -q -o ' + output_path +' ' \
            + '--camera={},{},{},{},{},{},{}'.format(translate_x,translate_y,translate_z,rot_x,rot_y,rot_z,dist) +' '\
            + '--imgsize={},{}'.format(width, height)+' '\
            + program_path
        #print(mycmd)
        #self.init_xserver()
        s = "".join(os.popen(mycmd).readlines())
        
    def get_3d(self, program_path, output_path):
        mycmd = 'openscad -q -o ' + output_path +' ' + program_path
        s = "".join(os.popen(mycmd).readlines())
