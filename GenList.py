# coding=utf-8
import os
import cv2
import random
import numpy
import sys

# if __name__ == "__main__":

# dict = {'1_musical_instrument': 0, #设置每一类的名称以及对应的label，名称需要与文件夹名字一致
#         '2_stringed_instrument': 1,
#         '3_keyboard_instrument': 2,
#         '4_wind_instrument': 3,
#         '5_guitar': 4,
#         '6_dulcimer': 5,
#         '7_koto': 6,
#         '8_acoustic_guitar': 7,
#         '9_electric_guitar': 8,
#         }
dict = {'0': 0, #设置每一类的名称以及对应的label，名称需要与文件夹名字一致
        '30': 1,
        '60': 2,
        '90': 3,
        '120': 4,
        '150': 5,
        '180': 6,
        '210': 7,
        '240': 8,
        '270': 9,
        '300': 10,
        '330': 11,
        }
rate = 0.1       #随机抽取10%的样本作为验证集
# root = './data/train'
# root= r'D:/Learn_Python/Image classification/dataset6/train'
root = 'DXL/sound_Data_2'


Trainlist = []
Testlist = []
alllist = []
index = 0
# max_num = 80000

for folder in dict:
    img_list = [f for f in os.listdir(os.path.join(root, folder)) if not f.startswith('.')]
    for img in img_list:
        str0 = '%d\t%s\t%d\n' % (index, os.path.join(folder, img), dict[folder])
        index += 1
        alllist.append(str0)

random.seed(100)
random.shuffle(alllist)

num = int(len(alllist) * rate)
Testlist = alllist[0:num]
Trainlist = alllist[num:]

Trainfile = open("DXL/sound_Data_2/train.txt", "w")
for str1 in Trainlist:
    Trainfile.write(str1)
Trainfile.close()

Testfile = open("DXL/sound_Data_2/valid.txt", "w")
for str1 in Testlist:
    Testfile.write(str1)
Testfile.close()


