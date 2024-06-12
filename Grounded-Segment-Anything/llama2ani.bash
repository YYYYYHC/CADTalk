data_path="/mnt/yhc/Ellip_depth/animal_e_V1"
res_path="/mnt/yhc/prediction_res/animal_e_V1llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path


data_path="/mnt/yhc/cuboid_depth/animal_c_v1"
res_path="/mnt/yhc/prediction_res/animal_c_v1llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path

data_path="/mnt/yhc/cuboid_depth/animal_c_v2"
res_path="/mnt/yhc/prediction_res/animal_c_v2llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path


data_path="/mnt/yhc/Ellip_depth/animal_e_V2"
res_path="/mnt/yhc/prediction_res/animal_e_V2llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path

