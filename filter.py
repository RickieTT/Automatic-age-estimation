# 导入 csv 库
import csv
import os

path = "age.csv"
path_test = "age_test.csv"
dataset_path = "../UTKFace"
# dataset_path = "../part1"
# os.listdir is a out-of-order list of function
files = os.listdir(dataset_path)

count = 0

for file in files:
    filename = dataset_path + '/' + file
    # size = int(os.path.getsize(filename) / 1024)
    size = int(os.path.getsize(filename))
    if size > 4096:
        count = count + 1
        # print(filename + "->" + str(int(os.path.getsize(filename)/1024)) + " kbyte")

print(count)