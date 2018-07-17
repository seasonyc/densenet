# -*- coding: utf-8 -*-
from __future__ import print_function 
import numpy as np  


#generate random index
def generate_rand_index():
    index=np.arange(10000)  
    np.random.shuffle(index)  
    print(index[0:20])
    
    np.save("validation_index.npy",index[0:5000])
    np.save("test_index.npy",index[5000:10000])
    
def load_index():
    index_v = np.load("validation_index.npy")
    index_t = np.load("test_index.npy")
    print(index_v[0:20])
    print(index_t[0:20])
