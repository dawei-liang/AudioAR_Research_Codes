# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:27:49 2019

@author: david
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed

resampling_path = './resampled_user_valid_data/'   # Path to load oversampled training data
save_model_dir = './resampled_user_valid_data/'   # Path to save/load CNN model

'''Fix random seed'''
seed(0)
set_random_seed(0)

#%%
def uint8_to_float32(x):   # Standardrization
    return (np.float32(x) - 128.) / 128.

#%%
def cnn_model_fn():
        model = Sequential()
        model.add(Conv1D(19, 5, strides=1, activation='linear', padding="same", input_shape=(128,1)))
        model.add(Conv1D(20, 5, strides=1, activation='linear', padding="same"))
        model.add(Conv1D(30, 5, strides=1, activation='linear', padding="same"))       
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dense(15, activation='softmax'))
            
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        loss = categorical_crossentropy
        model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
        return model

#%%
'''Save model and architecture'''
def save_model(clf, count):
    clf.save(save_model_dir + str(count) + 'replacement_cnn_base4_15classes_20epoch.h5')   # Save model
    yaml_string = clf.to_yaml()
    with open(save_model_dir + \
              str(count) + 'replacement_cnn_base4_15classes_20epoch.yaml', 'w') as f:   # Save architecture
        f.write(yaml_string)
    f.close()

#%%
def reshape(training_data_fit, training_labels):
    # Reshape training data as (#,128,1) for CNN
    training_data_fit = np.reshape(training_data_fit, (training_data_fit.shape[0], 128, 1))   
    # One-hot encoding for training labels: (#,15)
    training_labels = np_utils.to_categorical(training_labels, 15)
    return training_data_fit, training_labels

#%%
if __name__ == "__main__":
    training_data_fit = pd.read_csv(resampling_path + 'replacement_data_15classes.csv',
                                    header=None).values   # Training set
    training_labels = pd.read_csv(resampling_path + 'replacement_labels_15classes.csv',
                                    header=None).values
    print ('loaded data shape:', training_data_fit.shape)
    print ('loaded labels shape:', training_labels.shape)
    training_data_fit = uint8_to_float32(training_data_fit) # Normalization, otherwise CNN not works!!!
    
    # define 3-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cvscores = []
    count = 0
    for train, test in kfold.split(training_data_fit, training_labels):
        count += 1
        # training
        data, labels = reshape(training_data_fit[train], training_labels[train])
        print ('training_data_fit shape:', data.shape)
        print ('training_labels shape:', labels.shape)
        eval_data, eval_labels = reshape(training_data_fit[test], training_labels[test])
        print ('eval_data shape:', eval_data.shape)
        print ('eval_labels shape:', eval_labels.shape)
        model = cnn_model_fn()
        model.fit(data, labels,   
            batch_size=100,
            epochs=20,
            verbose=1,
            validation_data = (eval_data, eval_labels),
            shuffle=True,
            callbacks=[EarlyStopping(monitor='val_acc', patience=0, mode='auto')])
        save_model(model, count)
        print('Well trained and saved')
        
        # evaluate the model
        scores = model.evaluate(eval_data, eval_labels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
