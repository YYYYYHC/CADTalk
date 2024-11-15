# CADTalk

Hi, this is the original code for our CVPR2024 paper: CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs.

We use ControlNet for img-to-img conversion, and Grounded-SAM for semantic segmentation.

# ToDo

- [x] Release Usage and Data 
- [ ] Provide a Notebook version
- [ ] Make codes... readable

# Data

The dataset is avilable [here](https://drive.google.com/file/d/1NfqRnPEafU0wSYYCJejLH5tpepTGZpEO/view). Dataset_V1 and V2 are for different abstraction levels, i.e. primitive numbers. Since the dataset has not been throughly reviewed, we also release our labelling tool [here](https://drive.google.com/file/d/1NfqRnPEafU0wSYYCJejLH5tpepTGZpEO/view)

# Usage

## stage0: SPA/0_parse_scad_code.py

This file takes as input an arbitrary .scad file, and output all the 'to be commented' locations by adding placeholders to those locations.

The core function is the 'CADTalk_parser' function. Please specify the file paht and output path and run with 

    python SPA/0_parse_scad_code.py

*Known Issue* : The parser doesn't support all .scad files since it is developed only for this project.

## stage1.1: SPA/1_render_multiview.py

After having all the 'to-be-commented' blocks, this file is to render multiview images of the given CAD program, and, in the meantime, register pixel-to-block correspondence. The core is the 'render_multi_view' function, which takes four input:

- working_dir: this is used as a tempory storage during the registration, and the final results will also be here.
- program_dir: this is a folder that stores all the programs that you want to register and render.
- model_dir: this is to store the 3D model for all the programs.

the output is 'pixel2block.npy', which stores a correspondence matrix named by 'pixel2block', along with the block area and number of blocks.

the registration is used to conduct the voting, we achieve this by change block color and recognize the interested color.

## stage1.2: SPA/2_get_depth.py

This file is to produce depth image for the program. The input is the working_dir and model_dir specified in the previous stage. Change the two paths in 2_get_depth.py

The script requires blender to execute. Install [blender3.2](https://download.blender.org/release/Blender3.2/) and run it with


    [path_to_blender3.2/blender] ./SPA/2_get_depth.blend --background --python ./SPA/2_get_depth.py 

**Notice :** Blender 4.x will produce low-contrast depth maps, that is because the background depth is considered in the 'normalize' node.

You will find the depth image under working_dir with the name as depth0001.png

** For 'Unable to open a display' issue, the solution is:
1. run ``sudo ./virtualfb/virtualfb.sh `` 
2. you will see output like 'DISPLAY=:567'
3. run ``export DISPLAY=:567``
4. retry the blender script
5. run ``sudo ./virtualfb/virtualfb.sh stop`` to quit the virtualfb

You can avoid using blender by implementing your own depth-rendering tool.

## stage2: ControlNet/3_controlnet.py

This stage convert depth images to realistic images with ControlNet

1. Setup ControlNet with

        cd ControlNet
        conda env create -f environment.yaml
        conda activate control-v11

2. Download stable diffusion v1.5 checkpoint [v1-5-pruned.ckpt](https://huggingface.co/botp/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt?download=true)
 and [depth ControlNet](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth?download=true) and save under ControlNet/models.

2. Under the ControlNet folder, conduct depth-to-image with

        python ./3_controlnet.py

## stage3: Grounded-Segment-Anything/4_prediction.py

The file is to conduct open-vocabulary image segmentation based on the generated images.

1. Setup Grounded SAM by:

        cd Grounded-Segment-Anything
        make build-image
        
**Notice1 :** The dockerfile has been modified
**Notice2 :** For issue regrading installing GSA, please refer to their [official page](https://github.com/IDEA-Research/Grounded-Segment-Anything)

2. Run prediction (DINO + SAM) with:

        make run #you will enter a terminal
        cd Grounded-Segment-Anything/
        python 4_prediction.py 

You will have result under examples/stage3

## stage4: SPA/5_voting.py

With numpy installed:

        python SPA/5_voting.py

You will have the annotated code under examples/stage4

**Notice!!! :** We tried to visualize the code label in OpenSCAD, however, this only works for abstracted shape. You will see incorrect color display due to their complex program structure for instances from dataset_Real.



