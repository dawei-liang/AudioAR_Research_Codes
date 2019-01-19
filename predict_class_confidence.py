# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 01:29:18 2019

@author: david
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score

from keras.models import load_model
import matplotlib.pyplot as plt

#test_data_dir = './test_data/summer_2018_freesound/scripted study/1 single class/0/'
test_data_dir = './test_data/summer_2018_freesound/scripted study/'
classes = ['Bathing',
          'Flushing',
          'Brushing teeth',
          'Shaver',
          'Frying food',
          'Chopping',
          'Microwave oven',
          'Boiling water', 
          'Blender',
          'TV', 
          'Playing music', 
          'Vacuum',  
          'Washing',
          'Speaking', 
          'Strolling']

#%%
# Standardrization
def uint8_to_float32(x):   
    return (np.float32(x) - 128.) / 128.

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
    ''' Load evaluation/test data '''
    target = 'null.csv'   # End name of the file
    r = list_files(test_data_dir, target)   # Return target file dirs
    print('# of target csv files:', len(r))
    # Create a test array
    pd_object = pd.read_csv(r[0], header=None)   
    test_data = pd_object.values[:,0:128]
    
    for i in range(1, len(r)):
        pd_object = pd.read_csv(r[i], header=None)
        test_data = np.vstack((test_data, pd_object.values[:,0:128]))
    test_data = uint8_to_float32(test_data)
    print('test data shape:', test_data.shape)
    array_stack = test_data[0]   # Create an array to store the meaned test data
    
    i = 0
    while i < test_data.shape[0]-9:   # Segmentation for t: shape[0]-(t-1)
        try:
            temp = test_data[i]
            temp = np.vstack((temp, test_data[i+1]))
            temp = np.vstack((temp, test_data[i+2]))
            temp = np.vstack((temp, test_data[i+3]))
            temp = np.vstack((temp, test_data[i+4]))
            temp = np.vstack((temp, test_data[i+5]))
            temp = np.vstack((temp, test_data[i+6]))
            temp = np.vstack((temp, test_data[i+7]))
            temp = np.vstack((temp, test_data[i+8]))
            temp = np.vstack((temp, test_data[i+9]))
            array_stack = np.vstack((array_stack, np.mean(temp, axis = 0)))
            #array_stack = np.vstack((array_stack, temp))   # No segmentation input
            i += 10   # t
        except:
            pass
    test_data_mean = array_stack[1:,:]   # Meaned test data to predict
    print('test_data_mean.shape:', test_data_mean.shape, '\n')
    
    
#%%
    
    #clf = load_model('G:/Research1/codes/audioset/trained_models/summer_2018/raw_training_data/' 
     #                + 'raw_data_cnn_base4_15classes_20epoch.h5')   # Change name!
    clf = load_model('./trained_models/summer_2018/train_on_resampled_data/' + 'replacement_cnn_base4_15classes_20epoch.h5')
    

#%%
    
    '''Predictions'''
    test_data_mean = np.reshape(test_data_mean, (test_data_mean.shape[0], 128, 1))   # Reshape test data as (#,128,1) for CNN
    prediction_from_cnn = clf.predict(test_data_mean)
    print ('prediction_from_cnn.shape:', prediction_from_cnn.shape)
    prediction_from_cnn_1 = np.argmax(prediction_from_cnn, axis=1)
    distribution_from_cnn = np.mean(prediction_from_cnn, axis = 0)
    print('Distribution shape:', distribution_from_cnn.shape)
    print('sum of distribution:', distribution_from_cnn.sum())
    
#%%
    y_pos = np.arange(len(classes))
    plt.bar(y_pos, distribution_from_cnn, align='center', alpha=0.5)
    plt.ylim(top=1)
    plt.xticks(y_pos, classes)
    plt.ylabel('Probability')
    plt.title('Distribution overall Classes')
 
    plt.show()

    

   