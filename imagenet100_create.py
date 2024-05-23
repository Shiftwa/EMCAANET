import os
import shutil

def keep_matching_folders(dir_a, dir_b):
    # 获取目录A中的文件夹名列表
    folders_a = [folder for folder in os.listdir(dir_a) if os.path.isdir(os.path.join(dir_b, folder))]
    print(folders_a)
    # 遍历目录B
    for folder_b in os.listdir(dir_b):
        # 如果目录B中的文件夹名与目录A中的某个文件夹名相同，则保留该文件夹
        if folder_b in folders_a:
            # 构建文件夹的完整路径
            folder_b_path = os.path.join(dir_b, folder_b)
            # 移动或复制文件夹，这取决于你想要的操作
            # 这里采用复制操作，如果想要移动则用shutil.move()替代shutil.copytree()
            shutil.copytree(folder_b_path, os.path.join('destination3', folder_b))  # 修改'destination'为你想要存储的目录
a= "/root/train"
b = "/root/val"
# 调用函数并传入目录A和目录B的路径
keep_matching_folders(a,b)  # 修改'path_to_directory_a'和'path_to_directory_b'为实际的目录路径
