# CADTalk

Hi, this is the original code for our CVPR2024 paper: CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs.

We use ControlNet for img-to-img conversion, and Grounded-SAM for semantic segmentation.

# ToDo

- [ ] Release Usage and Data (on going, hopefully done before Aug 2024)
- [ ] Provide a Notebook version
- [ ] Make codes... readable

# Usage

## stage0: SPA/0_parse_scad_code.py

This file takes as input an arbitrary .scad file, and output all the 'to be commented' locations by adding placeholders to those locations.

The core function is the 'CADTalk_parser' function. Run with 

    python SPA/0_parse_scad_code.py

for an example and usage

## stage1: SPA/1_render_multiview.py

After having all the 'to-be-commented' blocks, this file is to render multiview images of the given CAD program, and, in the meantime, register pixel-to-block correspondence. The core is the 'render_multi_view' function, which takes four input:

- working_dir: this is used as a tempory storage during the registration, and the final results will also be here.
- program_dir: this is a folder that stores all the programs that you want to register and render.
- model_dir: this is to store the 3D model for all the programs.

the output is 'pixel2block.npy', which stores a correspondence matrix named by 'pixel2block', along with the block area and number of blocks.

the registration is used to conduct the voting, we achieve this by change block color and recognize the intersed color.

## stage1: SPA/2_get_depth.py

This file is to produce depth image for the program. The input is the working_dir and model_dir specified in the previous stage. Change the two paths in 2_get_depth.py

The script requires blender to execute. Install blender and run it with


    blender ./SPA/2_get_depth.blend --background --python ./SPA/2_get_depth.py 


You will find the depth image under working_dir with the name as depth0001.png

** For 'Unable to open a display' issue, the solution is:
1. run ``sudo ./virtualfb/virtualfb.sh `` 
2. you will see output like 'DISPLAY=:567'
3. run ``export DISPLAY=:567``
4. retry the blender script
5. run ``sudo ./virtualfb/virtualfb.sh stop`` to quit the virtualfb

You can avoid using blender by implementing your own depth-rendering tool.


