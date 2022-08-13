'''
change the index of ground-truth/ seg label/ oversmoothed image, 
the index depends on the order of the blending process 

'''


import os
import cv2
import numpy as np
from matplotlib.pyplot import *


f = open("map.txt","r")
name_map={}
while True:
    line = f.readline()
    if line:
        #name_map[line.split(',')[1][:-1]] = line.split(',')[0]
        name_map[line.split(',')[0]] = line.split(',')[1][:-1]
    else:
        break

output_dir = ''
file_dir = ''
for file in os.listdir(file_dir):
    key = file.split('.')[0]
    print(key)
    name = name_map[key]
    img = cv2.imread(file_dir + '/' + file)
    output_path = output_dir + name + '.png'
    cv2.imwrite(output_path, img)
    
