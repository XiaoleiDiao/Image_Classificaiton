# import os
# import shutil
#
# genpath = "D:\Learn_Python\Image_classification\Degree-Img-ori/"
# new_root = "D:\Learn_Python\Image_classification\Degree_Segment_dataset_new"
# datanames = os.listdir(genpath)  # 获取目录下的所有文件
#
# for i in datanames:  # 遍历
#     flag = True
#     name = i
#     print("name", name)
#     i = i.split(".")[0]  # 获取文件名，例如文件名为"qwe_asd_zxc.jpg",此时i="qwe_asd_zxc"
#     print("i1", i)
#     # i = i.split("_")[1] + "-" + i.split("_")[2]  # 将文件名按照“_"分开，例如文件名为"qwe_asd_zxc",此时i="asd_zxc"
#     i = i.split("-")[0]
#     print("i2", i)
#
#     for dirpath, dirnames, filenames in os.walk(genpath):  # 获取文件夹目录下的所有文件夹，dirnames就是获取到的文件夹
#         for filepath in dirnames:
#             if filepath == i:  # 判断目录是否存在，若存在，直接将文件移入
#                 # old_path = new_root + '/' + name
#                 old_path = genpath
#                 new_path = new_root + '/' + i
#
#                 print("old_path", old_path)
#                 print("new_path", new_path)
#
#                 file = name
#                 src = os.path.join(old_path, file)
#                 dst = os.path.join(new_path, file)
#                 print(new_path)
#                 shutil.move(old_path, new_path)  # 将文件移入文件夹
#                 flag = False  # 将标记置为False，就不执行下面的语句了
#     if flag:  # 若不存在，
#         os.makedirs(new_root + '/' + i)  # 新建文件夹
#         old_path = genpath
#         new_path = new_root + '/' + i
#         file = name
#         src = os.path.join(old_path, file)
#         dst = os.path.join(new_path, file)
#         print(new_path)
#         # shutil.move(old_path, new_path)
#         shutil.copy(old_path, new_path)


import os
import shutil

source_dir = r"D:\Learn_Python\Image_classification\Degree-Img-ori"
target_dir = r"D:\Learn_Python\Image_classification\Degree_Segment_dataset_new"

# 遍历源文件夹中的所有图片
for filename in os.listdir(source_dir):
    if filename.endswith(".png"):
        # 提取类别名
        category = filename.split("-")[0]

        # 构建目标路径
        target_path = os.path.join(target_dir, category)
        os.makedirs(target_path, exist_ok=True)

        # 复制图片到目标路径
        source_path = os.path.join(source_dir, filename)
        shutil.copy(source_path, target_path)
