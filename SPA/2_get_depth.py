import bpy
import math
import numpy as np
import os
# from tqdm import tqdm

DATA_PATH = "/home/cli7/yhc_Workspace/data/chairmodel400"
CUBES_PATH = "/home/cli7/yhc_Workspace/data/cuboid_dataset/cubes_refined/chair"
def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format is 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
 
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rxyz2xyz(rx, ry, rz, distance):
    initial_pos = np.array([0,-1*distance, 0])
    rm =  eulerAngles2rotationMat([rx,-1*rz,ry], format='degree')
    res_pos = np.dot(rm, initial_pos)
    return res_pos

def rd2xyz(theta, phi, distance, trans):
    r = distance
    phi_ = (phi-90)/180 * pi
    theta_ = (theta)/180 * pi
    x = r * math.sin(theta_) * math.cos(phi_) + trans[0]
    y = r * math.sin(theta_) * math.sin(phi_) + trans[1]
    z = r * math.cos(theta_) + trans[2]

    return (x,y,z)

def my_save_handler(scene):
    # now it runs after render completes
    if bpy.context.scene.my_custom_save == True:
        # assuming that you know the name of the node, or else you need a new string property
        bpy.data.images['Viewer Node.002'].save()
        # and now the bool property is reset
        bpy.context.scene.my_custom_save = False

# as for initialization having the property and handler registered
bpy.types.Scene.my_custom_save = bpy.props.BoolProperty(name="My Custom Save Property")

# prevent double register
if my_save_handler.__name__ not in [x.__name__ for x in bpy.app.handlers.render_post]:
    bpy.app.handlers.render_post.append(my_save_handler)

for f in os.listdir(CUBES_PATH):  
    # print(f)
    if 'cube' in f:
        # print(bpy.data.objects.keys())
        # break
        for key in bpy.data.objects.keys():
            if 'model' in key or 'Model' in key:
                current_model = bpy.data.objects[key]
                bpy.data.objects.remove(current_model)
        # break
    else:
        continue
    bpy.ops.import_mesh.stl(filepath=(os.path.join(CUBES_PATH, f, 'model.stl')))
    # print(bpy.data.objects.keys())
    # break
    # print([f for f in bpy.data.objects])
    current_model = bpy.context.selected_objects[0]
    bpy.data.objects["Camera"].constraints["Track To"].target = current_model
    current_model.rotation_euler[0]=math.pi/2
    
    
    for ry in [i*36 for i in range(10)]:

        bpy.data.scenes['Scene'].node_tree.nodes["File Output"].base_path = \
        os.path.join(DATA_PATH, f, f'ry={ry}/')
        pi = math.pi
        rx,ry,rz = [328, ry, 359]
        distance = 140

      
        xyz = rxyz2xyz(rx, ry, rz, distance)

        camera = bpy.data.objects["Camera"]

        for i in range(3):
            camera.location[i] =  xyz[i]
        bpy.ops.render.render()    