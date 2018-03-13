# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:58:07 2018

@author: david
"""

#%%   
''' Import videos '''

import h5py 
import numpy as np

hdf5_path = './packed_features/packed_features/unbal_train.h5'
def load_data(hdf5_path):
    z=[[],[],[],[],[]]
    with h5py.File(hdf5_path, 'r') as hf:
        for i in range(len(hf.get('y'))):   # Get label list: z[]
#            if (hf.get('y')[i,5] != 0) or (hf.get('y')[i,4] != 0):   #s
#                z[0].append(i)
#            elif (hf.get('y')[i,370] != 0) or (hf.get('y')[i,374] != 0) or (hf.get('y')[i,288] != 0):   #w
#                z[1].append(i)
#            elif (hf.get('y')[i,321] != 0) :   #b or (hf.get('y')[i,343] != 0)
#                z[2].append(i)
#            elif (hf.get('y')[i,509] != 0) or (hf.get('y')[i,514] != 0):   #o
#                z[3].append(i)
            if (hf.get('y')[i,385] != 0) or (hf.get('y')[i,384] != 0):   #t
                z[4].append(i)
                
#        z0=[hf.get('x')[e,:,:] for e in z[0][:1000]]   # Get feature vectors z[i],i:0~5
#        z1=[hf.get('x')[e,:,:] for e in z[1][:1000]]
#        z2=[hf.get('x')[e,:,:] for e in z[2][:1000]]
#        z3=[hf.get('x')[e,:,:] for e in z[3][:1000]]
        z4=[hf.get('x')[e,:,:] for e in z[4][:1000]]
        
#        z0 = np.asarray(z0)
#        z1 = np.asarray(z1)
#        z2 = np.asarray(z2)
#        z3 = np.asarray(z3)
        z4 = np.asarray(z4)
        
    return z4

def uint8_to_float32(x):   # Normalization
    return (np.float32(x) - 128.) / 128.
    
def bool_to_float32(y):
    return np.float32(y)

(z4) = load_data(hdf5_path)

#z0 = uint8_to_float32(z0)   # n*10*128
#z1 = uint8_to_float32(z1)
#z2 = uint8_to_float32(z2)
#z3 = uint8_to_float32(z3)
z4 = uint8_to_float32(z4)

#z0=np.reshape(z0,(z0.shape[0]*10,128))   # m*128
#z1=np.reshape(z1,(z1.shape[0]*10,128))
#z2=np.reshape(z2,(z2.shape[0]*10,128))
#z3=np.reshape(z3,(z3.shape[0]*10,128))
z4=np.reshape(z4,(z4.shape[0]*10,128))

#%%

import csv

#with open('./packed_features/packed_features/s.csv','w') as f:
#    wr = csv.writer(f,lineterminator='\n')
#    for i in range(z0.shape[0]):
#        wr.writerow(z0[i,:])
#f.close()
#with open('./packed_features/packed_features/w.csv','w') as f:
#    wr = csv.writer(f,lineterminator='\n')
#    for i in range(z1.shape[0]):
#        wr.writerow(z1[i,:])
#f.close()
#with open('./packed_features/packed_features/b.csv','w') as f:
#    wr = csv.writer(f,lineterminator='\n')
#    for i in range(z2.shape[0]):
#        wr.writerow(z2[i,:])
#f.close()
#with open('./packed_features/packed_features/o.csv','w') as f:
#    wr = csv.writer(f,lineterminator='\n')
#    for i in range(z3.shape[0]):
#        wr.writerow(z3[i,:])
#f.close()
with open('./packed_features/packed_features/t_385_384.csv','w') as f:
    wr = csv.writer(f,lineterminator='\n')
    for i in range(z4.shape[0]):
        wr.writerow(z4[i,:])
f.close()