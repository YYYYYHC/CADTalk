data_path="/mnt/yhc/CAD_render/tableV1e"
res_path="/mnt/yhc/prediction_res/tableV1eGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/table_E_raw"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/CAD_render/tableV2e"
res_path="/mnt/yhc/prediction_res/tableV2eGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/table_E_raw"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/CAD_render/animalV1e"
res_path="/mnt/yhc/prediction_res/animalV1eGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/animal_E_raw"
type="animal"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/CAD_render/animalV2e"
res_path="/mnt/yhc/prediction_res/animalV2eGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/animal_E_raw"
type="animal"
# 执行Python脚本

python exp2.py $data_path $res_path $code_path $type
data_path="/mnt/yhc/CAD_render/airplaneV2c"
res_path="/mnt/yhc/prediction_res/airplaneV2cGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/airplane_C"
type="airplane"

# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type

data_path="/mnt/yhc/CAD_render/tableV1c"
res_path="/mnt/yhc/prediction_res/tableV1cGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/table_C"
type="table"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type


data_path="/mnt/yhc/CAD_render/chairV1c"
res_path="/mnt/yhc/prediction_res/chairV1cGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V1/chair_C"
type="chair"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type


data_path="/mnt/yhc/CAD_render/chairV2c"
res_path="/mnt/yhc/prediction_res/chairV2cGPTCAD"
code_path="/home/cli7/yhc_Workspace/data/dataset_V2/chair_C_raw"
type="chair"
# 执行Python脚本
python exp2.py $data_path $res_path $code_path $type