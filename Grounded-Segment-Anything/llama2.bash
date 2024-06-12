data_path="/mnt/yhc/Ellip_depth/table_e_V1"
res_path="/mnt/yhc/prediction_res/table_e_V1llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path


data_path="/mnt/yhc/cuboid_depth/table"
res_path="/mnt/yhc/prediction_res/tableV2cllama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path


data_path="/mnt/yhc/Ellip_depth/table_e_V2"
res_path="/mnt/yhc/prediction_res/table_e_V2llama2"
# 执行Python脚本
python fourth_prediction.py $data_path $res_path

