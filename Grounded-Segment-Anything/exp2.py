import argparse
from CAD_prediction import get_prediction_with_info
# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='接受两个文件路径作为输入。')

# 添加文件路径参数
parser.add_argument('data_path', type=str, help='第一个文件的路径')
parser.add_argument('res_path', type=str, help='第二个文件的路径')
parser.add_argument('code_path', type=str, help='第二个文件的路径')
parser.add_argument('type', type=str, help='第二个文件的路径')

# 解析命令行参数
args = parser.parse_args()

get_prediction_with_info(args.data_path,args.res_path,args.code_path,args.type)
