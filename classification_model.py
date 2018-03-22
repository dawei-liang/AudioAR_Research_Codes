# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:03:20 2018

@author: david
"""

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
import os
import scipy.stats as stat
import tensorflow as tf
import matplotlib.pyplot as plt

#%%   
#''' Load all features and label matirx from evaluation set (optional, for quicker training)'''
#
#import h5py 
#
#hdf5_path = './packed_features/packed_features/eval.h5'
#def load_data(hdf5_path):
#    with h5py.File(hdf5_path, 'r') as hf:
#        x = hf.get('x')
#        y = hf.get('y')
#        video_id_list = hf.get('video_id_list')
#        x = np.array(x)
#        y = list(y)
#        video_id_list = list(video_id_list)
#        
#    return x, y, video_id_list
#
#def uint8_to_float32(x):
#    return (np.float32(x) - 128.) / 128.
#    
#def bool_to_float32(y):
#    return np.float32(y)
#
#(x, y, video_id_list) = load_data(hdf5_path)
#x = uint8_to_float32(x)		# x: feature matrix, shape: (N, 10, 128)
#y = bool_to_float32(y)		# y: label matrix, shape: (N, 527)

#%% 
#'''Import feature vectors from above: 1200(120 videos)+2990(299)+2470(247)+2850(285)+600(60)'''
#
### Import speech video data
#
#l1 =[]
#for i in range (y.shape[0]):
#    if (y[i,5] != 0) or (y[i,4] != 0): #5,4,21,70:Narration,Conversation,Chuckle,speech noise  
#        l1.append(i)
#print('# of speech videos', len(l1))      
##speech = np.mean(x[l1[0],:,:], axis = 0)
#speech = x[l1[0],:,:]
#for i in range (1,len(l1)):
#        #speech = np.vstack((speech, np.mean(x[l1[i],:,:], axis = 0)))
#        speech = np.vstack((speech, x[l1[i],:,:]))
#        
#print('audioset speech.shape:', speech.shape)
#
### Import washing video data
#l2 =[]
#for i in range (y.shape[0]):
#    if (y[i,370] != 0) or (y[i,374] != 0) or (y[i,288] != 0):   # Water tap, faucet index: #bathroom: 374,288; walking outside: 509, 514
#        l2.append(i)
#print('# of washing videos:',len(l2))      
##washing = np.mean(x[l2[0],:,:], axis = 0)
#washing = x[l2[0],:,:]
#for i in range (1,len(l2)):
#        #washing = np.vstack((washing, np.mean(x[l2[i],:,:], axis = 0)))
#        washing = np.vstack((washing, x[l2[i],:,:]))
#        
#print('audioset washing.shape:',washing.shape)
#
#''' Import bus video data '''
#l3 =[]
#for i in range (y.shape[0]):
#    if (y[i,321] != 0) or (y[i,343] != 0):   # taking bus: 321, 343  or light: (y[i,344] != 0)
#        l3.append(i)
#print('# of bus videos:',len(l3))      
##bus = np.mean(x[l2[0],:,:], axis = 0)
#bus = x[l3[0],:,:]
#for i in range (1,len(l3)):
#        #bus = np.vstack((bus, np.mean(x[l2[i],:,:], axis = 0)))
#        bus = np.vstack((bus, x[l3[i],:,:]))
#        
#print('audioset bus.shape:',bus.shape)
#
#''' Import outdoor video data   '''
#l4 =[]
#for i in range (y.shape[0]):
#    if (y[i,509] != 0) or (y[i,514] != 0):   # walking outside: urban or manmade:509, environmental:514, traffic noise(bad):327, toot:309
#        l4.append(i)
#print('# of outdoor videos:',len(l4))      
##outdoor = np.mean(x[l2[0],:,:], axis = 0)
#outdoor = x[l4[0],:,:]
#for i in range (1,len(l4)):
#        #outdoor = np.vstack((outdoor, np.mean(x[l2[i],:,:], axis = 0)))
#        outdoor = np.vstack((outdoor, x[l4[i],:,:]))
#
#''' Import typing video data   '''
#l5 =[]
#for i in range (y.shape[0]):
#    if  (y[i,385] != 0) :   # walking outside: typing:384, typewriter:385, computer keyboard:386
#        l5.append(i)
#print('# of typing videos:',len(l5))      
##typing = np.mean(x[l5[0],:,:], axis = 0)
#typing = x[l5[0],:,:]
#for i in range (1,len(l5)):
#        #typing = np.vstack((typing, np.mean(x[l5[i],:,:], axis = 0)))
#        typing = np.vstack((typing, x[l5[i],:,:]))
#        
#print('audioset typing.shape:',typing.shape)


#%%
''' Load feature vectors from unbalance set(top 1000 videos per activity) '''

## Get file lists ##
directory = './packed_features/packed_features/'

for filename in os.listdir(directory):
    if filename.endswith("s.csv"): 
        speech = pd.read_csv(directory + filename).values 
        speech = np.asarray(speech)         
    
    elif filename.endswith("w.csv"): 
        washing = pd.read_csv(directory + filename).values 
        washing = np.asarray(washing)         

    elif filename.endswith("b_321_only.csv"): 
        bus = pd.read_csv(directory + filename).values 
        bus = np.asarray(bus)  
        
    elif filename.endswith("o.csv"): 
        outdoor = pd.read_csv(directory + filename).values 
        outdoor = np.asarray(outdoor)         

    elif filename.endswith("t_385_384.csv"): 
        typing = pd.read_csv(directory + filename).values 
        typing = np.asarray(typing)      

print ('audioset speech.shape:',speech.shape)
print ('audioset washing.shape:',washing.shape)
print ('audioset bus.shape:',bus.shape)
print ('audioset outdoor.shape:',outdoor.shape)
print ('audioset typing.shape:',typing.shape)

#%%
#''' Clutering for video data (v1: cluster within each video)'''
## func for similarity comparison
##def check_similarity(features):
##    file_test = pd.read_csv('./test_data/street_bus_shop_kitchen.csv')
##    test_data = file_test.values
##    allfeatures = np.vstack((features, test_data))   # features + testdata
##    kmeans = KMeans(n_clusters=3, random_state=0)
##    kmeans.fit(allfeatures)
##    return(kmeans.labels_)
##clusters = check_similarity(chewing)  
#
## func for extracting similar features each video
#def cluster(features, key):
#
#    if key == 'w':
#        kmeans = KMeans(n_clusters=1, random_state=0)   
#    elif key == 'b':
#        kmeans = KMeans(n_clusters=1, random_state=0)   
#    elif key == 'o':
#        kmeans = KMeans(n_clusters=1, random_state=0)   
#    elif key == 's':
#        kmeans = KMeans(n_clusters=1, random_state=0)   
#    elif key == 't':
#        kmeans = KMeans(n_clusters=1, random_state=0)   
#
#    i = 0
#    remains = features[0,:]
#    while i != features.shape[0]:   # Each video
#        
#        temp2 = features[i,:]   # Each video
#        for j in range (i+1,i+10):   # Each frame
#            temp2 = np.vstack((temp2, features[j,:]))
#        #print(temp2.shape)
#        
#        kmeans.fit(temp2)   # Each video
#        labels = kmeans.labels_
#        mode, count = stat.mode(labels)   # Get mode label for each video
#        
#        avg = np.empty(temp2[0,:].shape)
#        count = 0
#        for k in range (0,10):   # Each frame
#            if labels[k] == mode:
#                avg = np.vstack((avg, temp2[k,:]))
#                count += 1
#        avg = np.mean(avg, axis = 0)
#        remains = np.vstack((remains, avg))
#                #print(labels[k])
#        i += 10
#        #print(i)
#    return remains
#remains_speech = cluster(speech,'s')
#print('# of clustered speech data:', remains_speech[1:,:].shape)
#remains_washing = cluster(washing,'w')
#print('# of clustered washing data:', remains_washing[1:,:].shape)
#remains_bus = cluster(bus,'b')
#print('# of clustered bus data:', remains_bus[1:,:].shape)
#remains_outdoor = cluster(outdoor,'o')
#print('# of clustered outdoor data:', remains_outdoor[1:,:].shape)
#remains_typing = cluster(typing,'t')
#print('# of clustered typing data:', remains_typing[1:,:].shape)
#      
#training_data = np.vstack((remains_washing[1:,:] , remains_bus[1:,:]))   ### Set up training set: w,b,o,s,t
##training_data = np.vstack((remains_bus[1:,:] , remains_outdoor[1:,:]))   ### Set up training set: b,o,s,t
#training_data = np.vstack((training_data, remains_outdoor[1:,:]))
#training_data = np.vstack((training_data, remains_speech[1:,:]))
#training_data = np.vstack((training_data, remains_typing[1:,:]))


#%%
''' Clutering for video data(v2: cluster across all videos)'''

# func for similarity comparison
#def check_similarity(features):
#    file_test = pd.read_csv('./test_data/street_bus_shop_kitchen.csv')
#    test_data = file_test.values
#    allfeatures = np.vstack((features, test_data))   # features + testdata
#    kmeans = KMeans(n_clusters=3, random_state=0)
#    kmeans.fit(allfeatures)
#    return(kmeans.labels_)
#clusters = check_similarity(chewing)  

# func for extracting similar features each video
def cluster(features, key):

    if key == 'w':
        kmeans = KMeans(n_clusters=3, random_state=0)   
    elif key == 'b':
        kmeans = KMeans(n_clusters=3, random_state=0)   
    elif key == 'o':
        kmeans = KMeans(n_clusters=3, random_state=0)   
    elif key == 's':
        kmeans = KMeans(n_clusters=3, random_state=0)   
    elif key == 't':
        kmeans = KMeans(n_clusters=3, random_state=0)   

    i = 0
    remains = features[0,:]
    kmeans.fit(features)   # Each video
    labels = kmeans.labels_
    mode, count = stat.mode(labels)   # Get mode label for each video
    for i in range (features.shape[0]):   # Each frame
        if labels[i] == mode:
            remains = np.vstack((remains, features[i,:]))
                #print(labels[k])
        #print(i)
    return remains

remains_speech = cluster(speech,'s')
print('# of clustered speech data:', remains_speech[1:,:].shape)
remains_washing = cluster(washing,'w')
print('# of clustered washing data:', remains_washing[1:,:].shape)
remains_bus = cluster(bus,'b')
print('# of clustered bus data:', remains_bus[1:,:].shape)
remains_outdoor = cluster(outdoor,'o')
print('# of clustered outdoor data:', remains_outdoor[1:,:].shape)
remains_typing = cluster(typing,'t')
print('# of clustered typing data:', remains_typing[1:,:].shape)
      
training_data = np.vstack((remains_washing[1:,:] , remains_bus[1:,:]))   ### Set up training set: w,b,o,s,t
training_data = np.vstack((training_data, remains_outdoor[1:,:]))
training_data = np.vstack((training_data, remains_speech[1:,:]))
training_data = np.vstack((training_data, remains_typing[1:,:]))

#%%
''' Set labels '''    

labels = []   # Order: w,b,o,s,t

for i in range(remains_washing[1:,:].shape[0]):   ### new
    labels.append('w')
print('# of washing labels:', remains_washing[1:,:].shape[0])

for i in range(remains_bus[1:,:].shape[0]):   ### new
    labels.append('b')
print('# of bus labels:', remains_bus[1:,:].shape[0])
      
for i in range(remains_outdoor[1:,:].shape[0]):   ### new
    labels.append('o')
print('# of outdoor labels:', remains_outdoor[1:,:].shape[0])
      
for i in range(remains_speech[1:,:].shape[0]):   ### new
    labels.append('s')
print('# of speech labels:', remains_speech[1:,:].shape[0])

for i in range(remains_typing[1:,:].shape[0]):   ### new
    labels.append('t')
print('# of typing labels:', remains_typing[1:,:].shape[0])      
      
#%%
''' Train and test '''

file_test = pd.read_csv('./test_data/overall.csv')
#file_test2 = pd.read_csv('./test_data/toilet.csv')
test_data = file_test.values
#test_data_washing_hands = file_test2.values
#test_data = np.mean(file_test.values,axis=0)
#test_data2 = np.mean(file_test2.values,axis=0)
#test_data = np.vstack((test_data,test_data2))
test_data_mean =test_data[0]
i=0
while i < test_data.shape[0]-1:   # Window size for test data: 2
    temp = test_data[i]
    temp = np.vstack((temp, test_data[i+1]))
    test_data_mean = np.vstack((test_data_mean, np.mean(temp, axis = 0)))
    i+=2
print('test_data_mean.shape:', test_data_mean[1:,:].shape)

#clf = RandomForestClassifier(n_estimators=500, max_features=128)  #was: 185
#clf.fit(training_data, labels)
#print('RF:', clf.predict(test_data_mean)[:10])
#RF = clf.predict(test_data_mean)

clf = LinearSVC(random_state = 0)
#clf = SVC(random_state = 0)
clf.fit(training_data, labels)
print('SVM:', clf.predict(test_data_mean)[:10])
SVC = clf.predict(test_data_mean)

prediction = []
prediction.append(SVC)
#prediction.append(RF)

#%% Classification by CNN

#tf.reset_default_graph()
#
#inputs = tf.placeholder(tf.float32, (None,128,1,1), name='input')
## Set up your label placeholders
#labelscnn = tf.placeholder(tf.int64, (None), name='labelscnn')
#train_labels_cnn = labels
#
## Step 1: define the compute graph of your CNN here
##   Use 5 conv2d layers (tf.contrib.layers.conv2d) and one pooling layer tf.contrib.layers.max_pool2d or tf.contrib.layers.avg_pool2d.
##   The output of the network should be a None x 1 x 1 x 6 tensor.
##   Make sure the last conv2d does not have a ReLU: activation_fn=None
#h = tf.contrib.layers.conv2d(inputs, 1, (5,5), stride=2, scope="conv1")
#h = tf.contrib.layers.conv2d(h, 2, (5,5), stride=2, scope="conv2")
#h = tf.contrib.layers.conv2d(h, 3, (5,5), stride=2, scope="conv3")
#h = tf.contrib.layers.conv2d(h, 5, (5,5), stride=2, scope="conv4")
#h = tf.contrib.layers.conv2d(h, 7, (3,3), stride=2, scope="conv5")
#
##h = tf.contrib.layers.max_pool2d(h, (3,3), stride=2, scope="pool")
#h = tf.contrib.layers.conv2d(h, 3, (1,1), stride=2, activation_fn=None, scope="conv6")
## The input here should be a   None x 1 x 1 x 6   tensor
#output = tf.identity(tf.contrib.layers.flatten(h), name='output')
#
## Step 2: use a classification loss function (from assignment 3)
#loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labelscnn))
#
## Step 3: create an optimizer (from assignment 3)
#optimizer = tf.train.MomentumOptimizer(0.001, 0.9)
#
## Step 4: use that optimizer on your loss function (from assignment 3)
#minimizer = optimizer.minimize(loss1)
#correct = tf.equal(tf.argmax(output, 1), labelscnn)
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
#print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]), '/', 100000 )
#
#
#
### Train model
## Batch size
#BS = 1
#
## Start a session
#sess = tf.Session()
#
## Set up training
#sess.run(tf.global_variables_initializer())
#
## This is a helper function that trains your model for several epochs un shuffled data
## train_func should take a single step in the optmimzation and return accuracy and loss
##   accuracy, loss = train_func(batch_images, batch_labels)
## HINT: train_func should call sess.run
#def train(train_func):
#    # An epoch is a single pass over the training data
#    for epoch in range(5):
#        # Let's shuffle the data every epoch
#        np.random.seed(epoch)
#        np.random.shuffle(training_data)
#        print('training_data.shape:', training_data.shape)
#        np.random.seed(epoch)
#        np.random.shuffle(train_labels_cnn)
#        # Go through the entire dataset once
#        accs, losss = [], []
#        for i in range(0, training_data.shape[0]-BS+1, BS):
#            # Train a single batch
#            batch_data, batch_labels = training_data[i:i+BS], train_labels_cnn[i:i+BS]
#            acc, loss = train_func(batch_data, batch_labels)
#            accs.append(acc)
#            losss.append(loss)
#        print('[%3d] Accuracy: %0.3f  \t  Loss: %0.3f'%(epoch, np.mean(accs), np.mean(losss)))
#
#
## Train convnet
#print('Convnet')
#for e in range(len(train_labels_cnn)):
#    if train_labels_cnn[e] == 'c':
#        train_labels_cnn[e] = 0
#    elif train_labels_cnn[e] == 'w':
#        train_labels_cnn[e] = 1
#    elif train_labels_cnn[e] == 'b':
#        train_labels_cnn[e] = 2
#    elif train_labels_cnn[e] == 'o':
#        train_labels_cnn[e] = 3
#    elif train_labels_cnn[e] == 's':
#        train_labels_cnn[e] = 4
#    elif train_labels_cnn[e] == 't':
#        train_labels_cnn[e] = 5
#
#train_labels_cnn = np.asarray(train_labels_cnn)
#print('training labels for CNN:', train_labels_cnn.shape)
#
#training_data = np.reshape(training_data, (training_data.shape[0],128,1,1))
#
#def mytrainfunc(training_data, train_labels_cnn):
#    _, acc, loss = sess.run([minimizer, accuracy, loss1], feed_dict={inputs: training_data, labelscnn: train_labels_cnn})
#    return acc, loss
#
#train(mytrainfunc)
#
#
#
#
#
##np.reshape(trainImages[:,:,0,-1], (784,1))
#test_data_cnn = np.reshape(test_data_mean,(test_data_mean.shape[0],128,1,1)) 
testLabels = np.empty(test_data_mean.shape[0])
print('testLabels.shape: ',testLabels.shape)
testLabels[0:150] = 3
testLabels[150:300] = 2
#testLabels[805:937] = 3
#testLabels[21:27] = 2
#testLabels[27:30] = 1
#testLabels[30:37] = 2
testLabels[300:419] = 1
testLabels[419:569] = 4
testLabels[569:] = 5

#print('Input test shape: ' + str(test_data_cnn.shape))
#print('Labels test shape: ' + str(testLabels.shape))
#
#val_accuracy, val_loss = sess.run([accuracy, loss1], feed_dict={inputs: test_data_cnn, labelscnn: testLabels})
## Calculate batch loss and accuracy
##x = tf.placeholder("float", shape=[None, 128,1,1])
##W = tf.Variable(tf.zeros([test_data.shape[0],128,1,1]))
##b = tf.Variable(tf.zeros([10]))
##y = tf.nn.softmax(tf.matmul(x,W) + b)
##
##sess = train(mytrainfunc)
##
##prediction=tf.argmax(y,1)
###output= sess.run(tf.argmax(prediction,1),feed_dict={x: test_data})
##print (prediction.eval(feed_dict={x: test_data}, session=sess))
#
#print("ConvNet Validation Accuracy: ", val_accuracy)


#%% Calculateing overall accuracy

c_svm=np.zeros(len(prediction[0]))   # For LSVM
w_svm=np.zeros(len(prediction[0]))
b_svm=np.zeros(len(prediction[0]))
o_svm = np.zeros(len(prediction[0]))
s_svm = np.zeros(len(prediction[0]))
t_svm = np.zeros(len(prediction[0]))
count_c=0
count_w=0
count_b=0
count_o=0
count_s=0
count_t=0
for i in range (len(prediction[0])):
    if prediction[0][i] == 'c':
        c_svm[i] = 1
        if testLabels[i] == 0:
            count_c += 1
    elif prediction[0][i] == 'w':
        w_svm[i] = 1
        if testLabels[i] == 1:
            count_w += 1
    elif prediction[0][i] == 'b':
        b_svm[i] = 1
        if testLabels[i] == 2:
            count_b += 1
    elif prediction[0][i] == 'o':
        o_svm[i] = 1
        if testLabels[i] == 3:
            count_o += 1
    elif prediction[0][i] == 's':
        s_svm[i] = 1
        if testLabels[i] == 4:
            count_s += 1
    elif prediction[0][i] == 't':
        t_svm[i] = 1
        if testLabels[i] == 5:
            count_t += 1
overall_accuracy = (count_w + count_b + count_o + count_s + count_t)/len(testLabels)
t_accuracy=(count_t)/len(testLabels[569:])
s_accuracy=(count_s)/len(testLabels[419:569])
w_accuracy=(count_w)/len(testLabels[300:419])
b_accuracy=(count_b)/len(testLabels[150:300])
o_accuracy=(count_o)/len(testLabels[0:150])

print('SVM: # of c,w,b,o,s,t: ', sum(c_svm),sum(w_svm),sum(b_svm),sum(o_svm),sum(s_svm),sum(t_svm),
      '\n accuracy: ', overall_accuracy,
      '\n w_accuracy: %.5f' % w_accuracy,
      '\n b_accuracy: %.5f' % b_accuracy,
      '\n o_accuracy: %.5f' % o_accuracy,
      '\n s_accuracy: %.5f' % s_accuracy,
      '\n t_accuracy: %.5f' % t_accuracy)

#c_rf=np.zeros(len(prediction[1]))   # For RF
#w_rf=np.zeros(len(prediction[1]))
#b_rf=np.zeros(len(prediction[1]))
#o_rf = np.zeros(len(prediction[1]))
#s_rf = np.zeros(len(prediction[1]))
#t_rf = np.zeros(len(prediction[1]))
#for i in range (len(prediction[1])):
#    if prediction[1][i] == 'c':
#        c_rf[i] = 1
#    elif prediction[1][i] == 'w':
#        w_rf[i] = 1
#    elif prediction[1][i] == 'b':
#        b_rf[i] = 1
#    elif prediction[1][i] == 'o':
#        o_rf[i] = 1
#    elif prediction[1][i] == 's':
#        s_rf[i] = 1
#    elif prediction[1][i] == 't':
#        t_rf[i] = 1
##svmn = len(prediction[0]) - svmp
#print('RF: # of c,w,b,o,s,t: ', sum(c_rf),sum(w_rf),sum(b_rf),sum(o_rf),sum(s_rf),sum(t_rf))
    
#%%
# Confusion Matrix
def Confusion_Matrix(testLabels, prediction):
    
    count_ww=0   # predicted + groundtruth
    count_wb=0
    count_wo = 0
    count_ws = 0
    count_wt = 0
    
    count_bw=0
    count_bb=0
    count_bo = 0
    count_bs = 0
    count_bt = 0
    
    count_ow=0
    count_ob=0
    count_oo = 0
    count_os = 0
    count_ot = 0
    
    count_sw=0
    count_sb=0
    count_so = 0
    count_ss = 0
    count_st = 0
    
    count_tw=0
    count_tb=0
    count_to = 0
    count_ts = 0
    count_tt = 0
    for i in range (len(prediction[0])):
        if prediction[0][i] == 'w':   # Predicted as 'w'
            if testLabels[i] == 1:
                count_ww += 1
            elif testLabels[i] == 2:
                count_wb += 1
            elif testLabels[i] == 3:
                count_wo += 1
            elif testLabels[i] == 4:
                count_ws += 1
            elif testLabels[i] == 5:
                count_wt += 1
                
        elif prediction[0][i] == 'b':   # Predicted as 'b'
            if testLabels[i] == 1:
                count_bw += 1
            elif testLabels[i] == 2:
                count_bb += 1
            elif testLabels[i] == 3:
                count_bo += 1
            elif testLabels[i] == 4:
                count_bs += 1
            elif testLabels[i] == 5:
                count_bt += 1
                
        elif prediction[0][i] == 'o':   # Predicted as 'o'
            if testLabels[i] == 1:
                count_ow += 1
            elif testLabels[i] == 2:
                count_ob += 1
            elif testLabels[i] == 3:
                count_oo += 1
            elif testLabels[i] == 4:
                count_os += 1
            elif testLabels[i] == 5:
                count_ot += 1
                
        elif prediction[0][i] == 's':   # Predicted as 's'
            if testLabels[i] == 1:
                count_sw += 1
            elif testLabels[i] == 2:
                count_sb += 1
            elif testLabels[i] == 3:
                count_so += 1
            elif testLabels[i] == 4:
                count_ss += 1
            elif testLabels[i] == 5:
                count_st += 1
                
        elif prediction[0][i] == 't':   # Predicted as 't'
            if testLabels[i] == 1:
                count_tw += 1
            elif testLabels[i] == 2:
                count_tb += 1
            elif testLabels[i] == 3:
                count_to += 1
            elif testLabels[i] == 4:
                count_ts += 1
            elif testLabels[i] == 5:
                count_tt += 1
    confusion = pd.DataFrame({'0Activities':['0Washing','1Bus','2Outdoor','3Speaking','4Typing','Recall'],
                              '0Washing':[count_ww,count_bw,count_ow,count_sw,count_tw,count_ww/(count_ww+count_bw+count_ow+count_sw+count_tw)],    
                              '1Bus':[count_wb,count_bb,count_ob,count_sb,count_tb,count_bb/(count_wb+count_bb+count_ob+count_sb+count_tb)], 
                              '2Outdoor':[count_wo,count_bo,count_oo,count_so,count_to,count_oo/(count_wo+count_bo+count_oo+count_so+count_to)],
                              '3Speaking':[count_ws,count_bs,count_os,count_ss,count_ts,count_ss/(count_ws+count_bs+count_os+count_ss+count_ts)],
                              '4Typing':[count_wt,count_bt,count_ot,count_st,count_tt,count_tt/(count_wt+count_bt+count_ot+count_st+count_tt)],
                              'Precision':[count_ww/(count_ww+count_wb+count_wo+count_ws+count_wt),
                                           count_bb/(count_bw+count_bb+count_bo+count_bs+count_bt),
                                           count_oo/(count_ow+count_ob+count_oo+count_os+count_ot),
                                           count_ss/(count_sw+count_sb+count_so+count_ss+count_st),
                                           count_tt/(count_tw+count_tb+count_to+count_ts+count_tt),
                                           None]})   # 1,2,3,4,5: w,b,o,s,t
    return confusion

confusion = Confusion_Matrix(testLabels, prediction)
print (confusion)

#%% Visulization


import scipy as sp
index_w = sp.nonzero(w_svm)[0]
index_b = sp.nonzero(b_svm)[0]
index_o = sp.nonzero(o_svm)[0]
index_s = sp.nonzero(s_svm)[0]
index_t = sp.nonzero(t_svm)[0]

truth_w = np.zeros(len(testLabels))
truth_b = np.zeros(len(testLabels))
truth_o = np.zeros(len(testLabels))
truth_s = np.zeros(len(testLabels))
truth_t = np.zeros(len(testLabels))
for i in range(len(testLabels)):
    if testLabels[i]==1:
        truth_w[i] = 1
    elif testLabels[i]==2:
        truth_b[i] = 1
    elif testLabels[i]==3:
        truth_o[i] = 1
    elif testLabels[i]==4:
        truth_s[i] = 1
    elif testLabels[i]==5:
        truth_t[i] = 1

index_truth_w = sp.nonzero(truth_w)[0]
index_truth_b = sp.nonzero(truth_b)[0]
index_truth_o = sp.nonzero(truth_o)[0]
index_truth_s = sp.nonzero(truth_s)[0]
index_truth_t = sp.nonzero(truth_t)[0]

fig = plt.figure(figsize=(10,4),facecolor='white')
ax = plt.subplot(111)
ax.plot(index_w,1.4*np.ones(int(sum(w_svm))),'|g',markersize=10,
        label='washing')
ax.plot(index_b,1.3*np.ones(int(sum(b_svm))),'|b',markersize=10,
        label='bus')
ax.plot(index_o,1.2*np.ones(int(sum(o_svm))),'|r',markersize=10,
        label='outdoor')
ax.plot(index_s,1.1*np.ones(int(sum(s_svm))),'|c',markersize=10,
        label='speech')
ax.plot(index_t,np.ones(int(sum(t_svm))),'|m',markersize=10,
        label='typing')

ax.plot(index_truth_w,0.8*np.ones(int(sum(truth_w))),'|g')
ax.plot(index_truth_b,0.8*np.ones(int(sum(truth_b))),'|b')
ax.plot(index_truth_o,0.8*np.ones(int(sum(truth_o))),'|r')
ax.plot(index_truth_s,0.8*np.ones(int(sum(truth_s))),'|c')
ax.plot(index_truth_t,0.8*np.ones(int(sum(truth_t))),'|m')

ax.set_ylim([0.5,2.2])
plt.xlabel('Overall Accuracy: %.5f' % overall_accuracy, fontsize=14)
ax.legend(loc='best')
plt.title('%s\n Predictions vs Groundtruth' % 'Linear SVM')

