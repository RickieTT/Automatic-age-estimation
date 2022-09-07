# 导入 csv 库
import csv
import os
import time

path = "age.csv"
path_test = "age_test.csv"

dataset_path = "../UTKFace"
# dataset_path = "../part1"
# os.listdir is a out-of-order list of function
files = os.listdir(dataset_path)
size = len(files)
flag = 0

def create_csv():

        with open(path,'w') as f:
            csv_write = csv.writer(f)
            csv_head = ['photo_id','age','group']
            csv_write.writerow(csv_head)
        with open(path_test,'w') as f:
            csv_write = csv.writer(f)
            csv_head = ['photo_id','age','group']
            csv_write.writerow(csv_head)

# function group by age
def age_group(age):
    if age >= 0 and age <= 3:
        # real_age = {age,'0-6'}
        return 1
    elif age >= 4 and age <= 8:
        # real_age = {age,'0-6'}
        return 2
    elif age >= 9 and age <= 15:
        return 3
    elif age >= 16 and age <= 24:
        return 4
    elif age >= 25 and age <= 31:
        return 5
    # elif age >= 40 and age <= 45:
    #     return 5
    # elif age >= 29 and age <= 34:
    #     return 6
    elif age >= 32 and age <= 40:
        return 6
    elif age >= 41 and age <= 60:
        return 7
    # elif age >= 55 and age <= 70:
    #     return 6
    elif age > 60:
        return 8


# def age_group(age):
#     if age >= 0 and age <= 6:
#         # real_age = {age,'0-6'}
#         return 1
#     if age >= 7 and age <= 15:
#         return 2
#     elif age >= 16 and age <= 25:
#         return 3
#     elif age >= 26 and age <= 35:
#         return 4
#     elif age >= 36 and age <= 50:
#         return 5
#     # elif age >= 46 and age <= 60:
#     #     return 6
#     # elif age >= 61 and age <= 70:
#     #     return 7
#     else:
#         return 6

def write_csv():
    global flag
    flag = 0
    for file in files:
        try:
            filename = dataset_path + '/' + file
            file_size = int(os.path.getsize(filename) / 1024)
            if file_size > 4:
                flag = flag + 1
                split_var = file.split('_')
                # print(split_var)
                group_num = age_group(int(split_var[0]))

                # file_size = os.path.getsize(filename)
                # print(filename + "->" + str(file_size))
                # if flag <= 8000 and file_size > 4:
                # if split_var[2] == '0':
                if flag <= 9000:
                    # print(str(flag) + filename + "->" + str(file_size))
                    # time.sleep(0.01)
                    # if flag <= 8000:
                    with open(path, 'a+') as f:
                        csv_write = csv.writer(f)
                        data_row = [file, split_var[0], group_num]
                        csv_write.writerow(data_row)
                if flag > 9000:
                    #
                    # if flag > 8000:
                    #     time.sleep(0.01)
                    with open(path_test, 'a+') as f:
                        csv_write = csv.writer(f)
                        data_row = [file, split_var[0], group_num]
                        csv_write.writerow(data_row)
        except:
            continue



if __name__ == '__main__':
    create_csv()
    write_csv()