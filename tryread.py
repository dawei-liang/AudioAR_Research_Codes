# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 01:11:07 2018

@author: david
"""
## Cite from https://stackoverflow.com/questions/42703849/audioset-and-tensorflow-understanding ##

import tensorflow as tf
import os
import csv
import shutil
import numpy as np
import pandas as pd

root_dir = './test_data/summer_2018_freesound/scripted study/cai/14/strolling'

audio_record = root_dir + '.tfrecords'
vid_ids = []
labels = []
start_time_seconds = [] # in secondes
end_time_seconds = []
feat_audio = []
count = 0

#if os.path.exists('./result'):
#    shutil.rmtree('./result')
#os.mkdir('./washing_hands')

for example in tf.python_io.tf_record_iterator(audio_record):
    tf_example = tf.train.Example.FromString(example)
    print(tf_example)
#    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
#    labels.append(tf_example.features.feature['labels'].int64_list.value)
#    start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value)
#    end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value)

    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)

    sess = tf.InteractiveSession()
    audio_frame = []
    
    rows = tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[0].bytes_list.value[0],tf.uint8)
                      ,tf.float32).eval()   # the first row
    # iterate through frames
    for i in range(1, n_frames):
        each_row = tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval()   # Obtain each row of features, 1*128, numpy array
        audio_frame.append(each_row)
        rows = np.vstack((rows,each_row))

    sess.close()
    feat_audio.append([])

    feat_audio[count].append(audio_frame)   #此处 feat_audio = audio_frame
    count+=1
    
print(each_row)
#print(feat_audio)

with open(root_dir + '.csv','w') as f:
    wr = csv.writer(f,lineterminator='\n')
    for i in range(n_frames):
        wr.writerow(rows[i,:])
f.close()