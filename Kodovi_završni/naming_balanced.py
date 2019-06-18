# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:23:44 2019

@author: shroo
"""

from PIL import Image 

import numpy as np
import os 
import imageio
  
labels = {'drink','food','inside','menu','outside'}
labels_set = set(labels)

counting_dict = {}
for i in labels_set:
    counting_dict[i] = 0

for label in labels_set:
    for img in os.listdir('./train'):
        imgName = img.split('-')[0] 
        if(imgName==str(label)):
            counting_dict[label] += 1
            path = os.path.join('./train', img)
            saveName = './train_balanced/' + imgName + '-' + str(counting_dict[label]) + '.jpg'
            image_data = np.array(Image.open(path))
            imageio.imwrite(saveName, image_data)
        if counting_dict[label]>=1500:
            break