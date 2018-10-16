# -*- coding: utf-8 -*-
from __future__ import print_function 

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Multiply, Concatenate
from keras.layers import Conv2D, SeparableConv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2




#should be -1
#concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

def conv_block(x, channels, kernel_size = 3, weight_decay=1e-4, dropout_rate = None):
    x = Conv2D(channels, (kernel_size, kernel_size), use_bias=False, padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def separable_conv_block(x, channels, kernel_size = 3, weight_decay=1e-4, dropout_rate = None):
    x = SeparableConv2D(channels, (kernel_size, kernel_size), use_bias=False, padding='same', depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

'''
After test, mc_block has not any strength

def mc_block(x, dropout_rate = None):
    channels = int(x.shape[-1])
    
    x1 = conv_block(x, channels, dropout_rate = dropout_rate)
    x2 = conv_block(x, channels, dropout_rate = dropout_rate)
    m = Multiply()([x1, x2])
    
    c = Concatenate()([x, m])

    return c
'''

def ds_block_cascade(x, dropout_rate = None):
    channels = int(x.shape[-1])
    
    x1 = conv_block(x, channels, dropout_rate = dropout_rate)
    x2 = conv_block(x1, channels, dropout_rate = dropout_rate)
    
    c = Concatenate()([x, x2])

    return c

def dc_block_cascade(x, dropout_rate = None):
    channels = int(x.shape[-1]) * 2
    
    x1 = conv_block(x, channels, dropout_rate = dropout_rate)
    x2 = conv_block(x1, channels, dropout_rate = dropout_rate)
    
    return x2

def transition_layer(x, channels, down_sampling = True, dropout_rate = None):
    x = conv_block(x, channels, dropout_rate = dropout_rate)
    if down_sampling:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x) 
    return x


def test_net(input_shape=None, func_block=ds_block_cascade, num_classes=None, dropout_rate=None, weight_decay=1e-4):
    X_input = Input(shape=input_shape)


    x = conv_block(X_input, 32, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 48, down_sampling = False, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 72, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 108, down_sampling = False, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 144, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 180, down_sampling = False, dropout_rate = dropout_rate)
    x = func_block(x, dropout_rate = dropout_rate)
    x = transition_layer(x, 216, down_sampling = False, dropout_rate = dropout_rate)
    

    x = GlobalAveragePooling2D()(x)
    
    x = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model_name = None
    if func_block == dc_block_cascade:
        model_name = 'testnet_dc'
    else:
        model_name = 'testnet'
        
    model = Model(inputs = X_input, outputs = x, name=model_name)
    
    
    return model, model_name



def vgg_like(input_shape=None, func_block=conv_block, num_classes=None, dropout_rate=None, weight_decay=1e-4):
    X_input = Input(shape=input_shape)


    x = func_block(X_input, 64)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) 
    x = func_block(x, 128)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = func_block(x, 256)
    x = func_block(x, 256)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = func_block(x, 512)
    x = func_block(x, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = func_block(x, 512)
    x = func_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model_name = None
    if func_block == separable_conv_block:
        model_name = 'vgg_separable'
    else:
        model_name = 'vgg'
    
    model = Model(inputs = X_input, outputs = x, name=model_name)
    
    return model, model_name


