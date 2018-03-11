# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:01:23 2018

@author: david
"""
#import os
#
#directory = 'G:/Research1/codes/audioset/features/audioset_v1_embeddings/eval'
#files = []
#for filename in os.listdir(directory):
#    if filename.endswith(".tfrecord"): 
#        files.append(os.path.join(directory, filename))
#    else:
#        continue

#%%
import h5py 
import numpy as np 
import pandas as pd

hdf5_path = './packed_features/packed_features/bal_train.h5'
def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = list(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.
    
def bool_to_float32(y):
    return np.float32(y)

(x, y, video_id_list) = load_data(hdf5_path)
x = uint8_to_float32(x)		# features, shape: (N, 10, 128)
y = bool_to_float32(y)		# label matrix, shape: (N, 527)

#%%
# How many labels per video?
assert(sum(np.bincount([np.argmax(j) for j in y])) == 22160)
print(np.bincount([np.argmax(j) for j in y])[:100])


#%%
# How many videos per index?

index_file = pd.read_csv('./packed_features/packed_features/class_labels_indices.csv')   # index_file: Dataframe
index = index_file[['index']].values
display_name = index_file[['display_name']]
#print(display_name.shape)
#index = index.T
n_videos_per_label = np.sum(y,axis=0)
#print(n_videos_per_label)
indices_and_counts = zip(index,n_videos_per_label)
sorted_indices_and_counts = sorted(indices_and_counts,reverse=True,key=lambda pair:pair[0])
print(indices_and_counts)
for (i,count) in sorted_indices_and_counts:
    print ("index %d - %d videos" % (i,count))
    pass
print('Total number of index: %d' %(index.shape[0]))

#%% Get index for flushing videos
l =[]
for i in range (y.shape[0]):
    if y[i,374] != 0:
        l.append(i)
print(len(l))      
flushing = np.mean(x[l[0],:,:], axis = 0)
for i in range (1,len(l)):
        flushing = np.vstack((flushing, np.mean(x[l[i],:,:], axis = 0)))
        
print(flushing.shape[0])