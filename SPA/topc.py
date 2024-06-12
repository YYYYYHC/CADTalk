import os

import open3d as o3d
import pdb
from tqdm import tqdm
# mesh = o3d.io.read_triangle_mesh("path_to_your_obj_file.obj")

# # mesh = o3d.io.read_triangle_mesh(source_path)
# # point_cloud = mesh.sample_points_poisson_disk(number_of_points=301402)  # 这里的 number_of_points 可以根据需要调整
# # point_cloud.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for i in range(len(point_cloud.points))])

# # o3d.io.write_point_cloud("test.ply", point_cloud)

# # pdb.set_trace()
# source_path = '/home/cli7/yhc_Workspace/suppl/filteredbird'
# target_path = '/home/cli7/yhc_Workspace/suppl/filteredbirdpc'
# for source_folder in os.listdir(source_path):
    
#     # if 'chair' not in source_folder or 'tar' in source_folder:
#     #     continue

#     os.makedirs(os.path.join(target_path, source_folder), exist_ok=True)
#     for f in tqdm(os.listdir(os.path.join(source_path, source_folder))):
#         if '.DS' in f:
#             continue
#         if not 'stl' in f:
#             continue
#         print(f)
#         model_path = os.path.join(source_path, source_folder, f)
#         model_name = f.split('.')[0]
#         target_pathf =  os.path.join(target_path ,source_folder,f'{model_name}.ply')

#         mesh = o3d.io.read_triangle_mesh(model_path)
#         point_cloud = mesh.sample_points_poisson_disk(number_of_points=51402)  # 这里的 number_of_points 可以根据需要调整
#         point_cloud.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for i in range(len(point_cloud.points))])
#         o3d.io.write_point_cloud(target_pathf, point_cloud)
model_path = "/home/cli7/yhc_Workspace/data/airplane400stl/airplanev1c/cube_1_0002.scad/model.stl"
target_pathf = "/home/cli7/yhc_Workspace/suppl/baseap.ply"
mesh = o3d.io.read_triangle_mesh(model_path)
point_cloud = mesh.sample_points_poisson_disk(number_of_points=51402)  # 这里的 number_of_points 可以根据需要调整
point_cloud.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for i in range(len(point_cloud.points))])
o3d.io.write_point_cloud(target_pathf, point_cloud)

        
