# CADTalk

Hi, this is the original code for our CVPR2024 paper: CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs.

We use ControlNet for img-to-img conversion, and Grounded-SAM for semantic segmentation.

# ToDo

- [ ] Release Usage and Data
- [ ] Provide a Notebook version
- [ ] Make codes... readable

# Usage

## SPA/0_parse_scad_code.py

This file takes as input an arbitrary .scad file, and output all the 'to be commented' locations by adding placeholders to those locations.

The core function is the CADTalk_parser function. Run with 
'''
python SPA/0_parse_scad_code.py
'''
for an example and usage

##