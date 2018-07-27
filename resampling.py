# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:07:46 2018

@author: david
"""

from imblearn.over_sampling import SMOTE, RandomOverSampler
import csv
import numpy as np

class resampling():
    def __init__(self, input_data, input_labels, resampling_path):
        self.input_data = input_data
        self.input_labels = input_labels
        self.resampling_path = resampling_path
        
    def SMOTE(self):
        sm = SMOTE(ratio='auto', random_state=0, n_jobs=-1)
        self.resampled_data, self.resampled_labels = sm.fit_sample(self.input_data, self.input_labels)
        print('Resampled dataset shape(SMOTE):', self.resampled_data.shape, '\n')

    def random(self):
        rd = RandomOverSampler(ratio='auto', random_state=0)
        self.resampled_data, self.resampled_labels = rd.fit_sample(self.input_data, self.input_labels)
        print('Resampled dataset shape(random):', self.resampled_data.shape, '\n')
        
    def save_resampled_data(self):
        with open(self.resampling_path + 'replacement_data_15classes'+'.csv','w') as f:
            wr = csv.writer(f,lineterminator='\n')
            wr.writerows(self.resampled_data)
        f.close()
    
    def save_resampled_labels(self):
        with open(self.resampling_path + 'replacement_labels_15classes'+'.csv','w') as f:
            wr = csv.writer(f,lineterminator='\n')
            self.resampled_labels = np.reshape(self.resampled_labels, (self.resampled_labels.shape[0], 1))
            wr.writerows(self.resampled_labels)
        f.close()
