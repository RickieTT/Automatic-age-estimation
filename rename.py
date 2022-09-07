# 导入 csv 库
import csv
import os

path = "age.csv"
path_test = "age_test.csv"
dataset_path = "../UTKFace"
# dataset_path = "../part1"
# os.listdir is a out-of-order list of function
files = os.listdir(dataset_path)

# flag12 = 0

for file in files:

    split_var = file.split('.')
    if file != '.DS_Store' and split_var[2] == "chip":
        # 重新组合文件名和后缀名
        newname = split_var[0] + ".jpg"
        filenamedir = dataset_path + '/' + file
        newnamedir = dataset_path + '/' + newname
        print(newnamedir)

        os.rename(filenamedir, newnamedir)