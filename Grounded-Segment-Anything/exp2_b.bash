data_path="/mnt/yhc/Ellip_depth/table_e_V1"
res_path="/mnt/yhc/prediction_res/tableV1eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/table_E_raw"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/Ellip_depth/table_e_V2"
res_path="/mnt/yhc/prediction_res/tableV2eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/table_E_raw"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type


data_path="/mnt/yhc/Ellip_depth/animal_e_V1"
res_path="/mnt/yhc/prediction_res/animalV1eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/animal_E_raw"
type="animal"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/Ellip_depth/animal_e_V2" TBD
res_path="/mnt/yhc/prediction_res/animalV2eExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/animal_E_raw"
type="animal"
# 执行Python脚本

python exp2.py $data_path $res_path $code_path $type
data_path="/mnt/yhc/cuboid_depth/airplaneV2_c_s100"
res_path="/mnt/yhc/prediction_res/airplaneV2cExp2"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/airplane_C"
type="airplane"

# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

