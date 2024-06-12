data_path="/mnt/yhc/Ellip_depth/airplane_e_V1"
res_path="/mnt/yhc/prediction_res/airplaneV1eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/airplane_E"
type="airplane"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/Ellip_depth/airplane_e_V2"
res_path="/mnt/yhc/prediction_res/airplaneV2eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/airplane_E_raw"
type="airplane"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/Ellip_depth/chair_e_V1"
res_path="/mnt/yhc/prediction_res/chairV1eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/chair_E_raw"
type="chair"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/Ellip_depth/chair_e_V2"
res_path="/mnt/yhc/prediction_res/chairV2eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/chair_E_raw"
type="chair"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type


data_path="/mnt/yhc/cuboid_depth/animal_c_v1"
res_path="/mnt/yhc/prediction_res/animalV1cExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/animal_C_raw"
type="animal"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/cuboid_depth/animal_c_v2"
res_path="/mnt/yhc/prediction_res/animalV2cExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/animal_C_raw"
type="animal"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type


data_path="/mnt/yhc/cuboid_depth/tableV2_c"
res_path="/mnt/yhc/prediction_res/tableV2cExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/table_C"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type