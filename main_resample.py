# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:02:21 2018

@author: david
"""

'''Main file to resample and save the Audio Set data'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import loading_data_and_labels
import resampling

dataset_dir = './CSV_files/'   # Path to load raw Audio Set data
resampling_path = './replacement_data/'   # Path to save resampled data
classes = ['Bathing',
          'Flushing',
          'Brushing teeth',
          'Shavering',
          'Frying food',
          'Chopping food',
          'Microwave oven',
          'Boiling water', 
          'Squeezing juice',
          'TV', 
          'Piano music', 
          'Cleaning',  
          'Washing',
          'Chatting', 
          'Strolling']
resampling_mode = 'random'   # ['SMOTE' or 'random']   # May change the desired save file names in sampling.py 

#%%


if __name__ == "__main__":
  
   #Load raw data and labels
   training_object = loading_data_and_labels.loading_data_and_labels(dataset_dir, classes)
   training_object.loading_data()
   training_object.loading_labels()
   X = training_object.training_data_fit
   Y = training_object.training_labels
   #Resampling using SMOTE
   if resampling_mode == 'SMOTE':
        resampling_object = resampling.resampling(X, Y, resampling_path)
        resampling_object.SMOTE()
        resampling_object.save_resampled_data()
        resampling_object.save_resampled_labels()
   #Resampling using random
   elif resampling_mode == 'random':
        resampling_object = resampling.resampling(X, Y, resampling_path)
        resampling_object.random()
        resampling_object.save_resampled_data()
        resampling_object.save_resampled_labels()
        
   print('Sampling Complete')

    
    
