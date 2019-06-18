# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:58:46 2019

@author: shroo
"""
from PIL import Image 

import pandas as pd
import numpy as np
import os 
import imageio
import json
 

kategorije=pd.read_json('photo.json', lines='true')
photo_id = np.array(kategorije['photo_id'])
labels = np.array(kategorije['label'])

naming_dict = dict()
for i in range(0,len(photo_id)):
    naming_dict[photo_id[i]] = labels[i]

type(np.array(kategorije['photo_id']))
  
labels_dict = naming_dict.values()
labels_set = set(labels)

counting_dict = {}
for i in labels_set:
    counting_dict[i] = 0
    
    
#TRAIN DATA
  
for img in os.listdir('./raw_data/photos')[0:int(len(os.listdir('./raw_data/photos'))*0.75)]:
  imgName = img.split('.')[0] 
  label = naming_dict[str(imgName)]
  counting_dict[label] += 1
  path = os.path.join('./raw_data/photos', img)
  saveName = './train/' + label + '-' + str(counting_dict[label]) + '.jpg'
  image_data = np.array(Image.open(path))
  imageio.imwrite(saveName, image_data)
  
  
#TEST DATA
  
for img in os.listdir('./raw_data/photos')[int(len(os.listdir('./raw_data/photos'))*0.75):int(len(os.listdir('./raw_data/photos')))]:
  imgName = img.split('.')[0] 
  label = naming_dict[str(imgName)]
  path = os.path.join('./raw_data/photos', img)
  saveName = './test/' + imgName + '.jpg'
  image_data = np.array(Image.open(path))
  imageio.imwrite(saveName, image_data)
  
