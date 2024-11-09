import pdb
import bpy
import math
import os

working_dir = "./examples/stage1/working_dir"
model_dir = "./examples/stage1/model_dir"

def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
 
    R_x = [
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ]
 
    R_y = [
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ]
 
    R_z = [
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ]

    # Matrix multiplication R = Rz * Ry * Rx
    R_zy = [[sum(a * b for a, b in zip(R_z_row, R_y_col)) for R_y_col in zip(*R_y)] for R_z_row in R_z]
    R = [[sum(a * b for a, b in zip(R_zy_row, R_x_col)) for R_x_col in zip(*R_x)] for R_zy_row in R_zy]

    return R

def mat_mult_vec(mat, vec):
    return [sum(m * v for m, v in zip(mat_row, vec)) for mat_row in mat]

def rxyz2xyz(rx, ry, rz, distance):
    initial_pos = [0, 0, distance]
    rm = eulerAngles2rotationMat([rx, ry, rz], format='degree')
    res_pos = mat_mult_vec(rm, initial_pos)
    return res_pos

def rd2xyz(theta, phi, distance, trans):
    r = distance
    phi_ = (phi-90) / 180 * math.pi
    theta_ = (theta) / 180 * math.pi
    x = r * math.sin(theta_) * math.cos(phi_) + trans[0]
    y = r * math.sin(theta_) * math.sin(phi_) + trans[1]
    z = r * math.cos(theta_) + trans[2]

    return (x, y, z)

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

for f in os.listdir(model_dir):
    
    for key in bpy.data.objects.keys():
        if 'model' in key or 'Model' in key:
            current_model = bpy.data.objects[key]
            bpy.data.objects.remove(current_model)

    bpy.ops.import_mesh.stl(filepath=(os.path.join(model_dir, f, 'model.stl')))
    current_model = bpy.context.selected_objects[0]
    bpy.data.objects["Camera"].constraints["Track To"].target = current_model
    # current_model.rotation_euler[0] = math.pi / 2

    for ry in [i * 36 for i in range(10)]:
        print('processing:', ry)
        bpy.data.scenes['Scene'].node_tree.nodes["File Output"].base_path = \
        os.path.join(working_dir, f, f'rz={ry}')
        bpy.data.scenes['Scene'].node_tree.nodes["File Output"].file_slots[0].path = "depth"
        rx, ry, rz = [65, 0, ry]
        distance = 500
        
        xyz = rxyz2xyz(rx, ry, rz, distance)

        camera = bpy.data.objects["Camera"]

        for i in range(3):
            camera.location[i] = xyz[i]
        
        bpy.ops.render.render()
        