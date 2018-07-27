# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:24:15 2018

@author: david
"""
import os
import pandas as pd
import numpy as np


class loading_data_and_labels():
    def __init__(self, dataset_dir, classes):
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.training_data = [pd.DataFrame() for i in range(len(self.classes))]
    
    def loading_data(self):   
        '''Input raw embeddings''' 
         
        self.training_data_fit = pd.DataFrame()
        
        training_sets = [x for x in os.listdir(self.dataset_dir) if x.endswith('.csv')]
        print (training_sets)
        for file in training_sets:
            class_index = int(file.split('_')[-1].strip('.csv'))   # Get class index
            temp_data = pd.read_csv(self.dataset_dir + file, header=None)
            # Load 11 single classes, pay attention that csv file indexed 12 (lawn mower) is not used, thus skipping
            if class_index <= 11:   
                self.training_data[class_index] = self.training_data[class_index].append(temp_data)
            # Load 3 double classes
            # Washing
            elif class_index == 13 or class_index == 14:   
                self.training_data[12] = self.training_data[12].append(temp_data)
            # Chatting
            elif class_index == 15 or class_index == 16:   
                self.training_data[13] = self.training_data[13].append(temp_data)
            # Strolling
            elif class_index == 17 or class_index == 18:   
                self.training_data[14] = self.training_data[14].append(temp_data)
        for i in range(len(self.training_data)):
            print('# of %s' % self.classes[i] + ' embeddings: %d' % self.training_data[i].shape[0], '\n')
            self.training_data_fit = self.training_data_fit.append(self.training_data[i])
            
        self.training_data_fit = self.training_data_fit.values   # Training data to fit the classifiers
        print('Total training set size:', self.training_data_fit.shape, '\n')
        
#%%
    def loading_labels(self):
        '''Set training labels'''
        self.training_labels = []
        
        for i in range(len(self.training_data)):
            label_count = 0
            for row in range(self.training_data[i].shape[0]):   # Set training labels for each class
                self.training_labels.append(i)
                label_count += 1
            print('# of %s' % self.classes[i] + ' labels: %d' % label_count)
        self.training_labels = np.asarray(self.training_labels)
        self.training_labels = np.ndarray.flatten(self.training_labels)   # Training labels to fit the classifiers
        print('Total training label size:', self.training_labels.shape, '\n')

