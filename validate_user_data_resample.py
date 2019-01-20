# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 00:09:40 2019

@author: david
"""

from imblearn.over_sampling import SMOTE, RandomOverSampler
import csv
import numpy as np
import pandas as pd
import os


test_data_dir = './test_data/summer_2018_freesound/scripted study/'
resampling_path = './resampled_user_valid_data/'
classes = ['bathing',
          'flushing',
          'brushing',
          'shaver',
          'frying',
          'chopping',
          'micro',
          'boiling', 
          'blender',
          'TV', 
          'piano', 
          'vacuum',  
          'washing',
          'chatting', 
          'strolling']
#%%
'''Funcs for oversampling'''

def random(input_data, input_labels):
    rd = RandomOverSampler(ratio='auto', random_state=0)
    resampled_data, resampled_labels = rd.fit_sample(input_data, input_labels)
    print('Resampled dataset shape(random):', resampled_data.shape, '\n')
    return resampled_data, resampled_labels
    
def save_resampled_data(resampled_data):
    with open(resampling_path + 'replacement_data_15classes'+'.csv','w') as f:
        wr = csv.writer(f,lineterminator='\n')
        wr.writerows(resampled_data)
    f.close()

def save_resampled_labels(resampled_labels):
    with open(resampling_path + 'replacement_labels_15classes'+'.csv','w') as f:
        wr = csv.writer(f,lineterminator='\n')
        resampled_labels = np.reshape(resampled_labels, (resampled_labels.shape[0], 1))
        wr.writerows(resampled_labels)
    f.close()
    

#%% 
# Return target file dirs
def list_files(dir, target):                                                                                              
    r = []                                                                                                                                                                                        
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(target):
                r.append(os.path.join(root, name))
    return r

#%%
if __name__ == "__main__":
    ''' Load training data '''
    array_stack = np.zeros((1,128))
    label = []
    for k in range(len(classes)):
        target = classes[k] + '.csv'   # End name of the file
        r = list_files(test_data_dir, target)   # Return all target file dirs of class k
        print('# of target %s csv files:' %classes[k], len(r))
        # Create a test array
        pd_object = pd.read_csv(r[0], header=None)   
        class_data = pd_object.values[:,0:128]
    
        for i in range(1, len(r)):
            pd_object = pd.read_csv(r[i], header=None)
            class_data = np.vstack((class_data, pd_object.values[:,0:128]))
        print('target %s data shape:' %classes[k], class_data.shape)
        for l_count in range(class_data.shape[0]):   # Create labels for class k
            label.append(k)
        array_stack = np.vstack((array_stack, class_data))   # Stack data for each class
    array_stack = array_stack[1:]
    print('training shape:', array_stack.shape)
    print('label shape:', len(label))
    
    # Sampling
    resampled_data, resampled_labels = random(array_stack, label)
    # Save data and labels
    save_resampled_data(resampled_data)
    save_resampled_labels(resampled_labels)
    print('Done')
    
    
