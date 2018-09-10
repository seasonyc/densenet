# -*- coding: utf-8 -*-
from __future__ import print_function 
'''
# to make the memory growable
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
'''
import keras 
from keras.datasets import cifar10 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import load_model
import keras.backend as K
import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import densenet
import incremental_densenet
import testnet

 
num_classes = 10 


def load_data():
    # The data, shuffled and split between train and test sets: 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
    print('x_train shape:', x_train.shape) 
    print(x_train.shape[0], 'train samples') 
    print(x_test.shape[0], 'test samples') 
     
    # Convert class vectors to binary class matrices. 
    y_train = keras.utils.to_categorical(y_train, num_classes) 
    y_test = keras.utils.to_categorical(y_test, num_classes) 
    
    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 
    x_train /= 255 
    x_test /= 255 
    index_v = np.load("validation_index.npy")
    return (x_train, y_train), (x_test[index_v,], y_test[index_v])



#should be -1
#concat_axis = 1 if K.image_data_format() == 'channels_first' else -1


        
        
def plot_error_images(images, cls_true, cls_pred=None, smooth=True):
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    for i in range(len(cls_true)):
        if np.argmax(cls_true[i]) != np.argmax(cls_pred[i]):
            plt.imshow(images[i, :, :, :],
                          interpolation=interpolation)
            
            cls_true_name = class_names[np.argmax(cls_true[i])]
            cls_pred_name = class_names[np.argmax(cls_pred[i])]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            plt.title(xlabel)
            plt.show()



def train(model, x_train, y_train, x_validation, y_validation,
          epochs_list, name,
          batch_size = 64,  
          learning_rate = 1e-3,
          lr_decay_ratio = 0.1,
          data_augmentation = True ):
    
    #opt = keras.optimizers.Adam(lr=learning_rate, epsilon=1e-08) 
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True) 
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy']) 
               
    #to enable tensorboard
    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=False, write_grads=True, write_images=True)
    #tensorboard --logdir=C:\...\logs
 
    filepath = name + 'model-ep{epoch:04d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, period=10)

    def schedule(epoch):
        lr = learning_rate;
        for epochs in epochs_list:
            if epoch >= epochs:
                lr *= lr_decay_ratio
            else:
                break
        return lr
    
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
        
    if not data_augmentation: 
        print('Not using data augmentation.') 
        model.fit(x_train, y_train, 
                  batch_size=batch_size, 
                  epochs=epochs_list[-1], 
                  validation_data=(x_validation, y_validation),
                  callbacks=[lr_scheduler, checkpoint],
                  #, callbacks=[tensorboard]
                  shuffle=True)
    else: 
        print('Using real-time data augmentation.') 
        # This will do preprocessing and realtime data augmentation: 
    
        datagen = ImageDataGenerator( 
            featurewise_center=False,  # set input mean to 0 over the dataset 
            samplewise_center=False,  # set each sample mean to 0 
            featurewise_std_normalization=False,  # divide inputs by std of the dataset 
            samplewise_std_normalization=False,  # divide each input by its std 
            zca_whitening=False,  # apply ZCA whitening 
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180) 
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width) 
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height) 
            horizontal_flip=True,  # randomly flip images 
            vertical_flip=False)  # randomly flip images 

        datagen.fit(x_train) 
        
        # Fit the model on the batches generated by datagen.flow(). 
        model.fit_generator(datagen.flow(x_train, y_train, 
                                         batch_size=batch_size), 
                            steps_per_epoch=x_train.shape[0] // batch_size, 
                            epochs=epochs_list[-1], 
                            callbacks=[lr_scheduler, checkpoint],
                            #, callbacks=[tensorboard]) 
                            validation_data=(x_validation, y_validation)) #, callbacks=[tbCallBack]) 
           

def error_analyze(model, x, y):
    predicts = model.predict(x, verbose=1)
    print(predicts.shape)
    plot_error_images(x, y, predicts)
    
    
    
    
def load_test_data():
    # The data, shuffled and split between train and test sets: 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
    
    y_test = keras.utils.to_categorical(y_test, num_classes) 
    
    x_test = x_test.astype('float32') 
    x_test /= 255 
    index_t = np.load("test_index.npy")
    return x_test[index_t,], y_test[index_t]


def test(model = None, model_file = None, error_anal = False):
    if not model:
        model = load_model(model_file)
        print(model_file + ' loaded.')
    model.summary()  
    
    x_test, y_test = load_test_data()
    loss, acc = model.evaluate(x_test, y_test)
    print(loss)
    print(acc)

    if error_anal:
        error_analyze(model, x_test, y_test)
    

'''  
To test a trained model
test(model_file = 'denseaugmodel-ep0290-loss0.143-acc0.997-val_loss0.320-val_acc0.947.h5', error_anal = True)     
'''

def get_aug_str(aug):
    if aug:
        return '_aug'
    else:
        return '_noaug'
    
    
def main_dense(learning_rate, network, error_anal = False,
         growth_rate=12, dropout_rate=0.2, bottleneck=False, compression=1.0, aug=False):
    (x_train, y_train), (x_validation, y_validation) = load_data()
    
    model, model_name = network(input_shape=x_train.shape[1:], nb_classes=num_classes, 
                              depth=40, dense_blocks=3, growth_rate=growth_rate,
                              dropout_rate=dropout_rate, bottleneck=bottleneck, compression=compression)
    model.summary()  
    train(model, x_train, y_train, x_validation, y_validation,
          epochs_list = [150, 225, 300], name = model_name + get_aug_str(aug), learning_rate = learning_rate, data_augmentation = aug)

    if error_anal:
        error_analyze(model, x_validation, y_validation)




def main_testnet(learning_rate, network, func_block, error_anal = False, dropout_rate=0.2, aug=False):
    (x_train, y_train), (x_validation, y_validation) = load_data()
    
    model, model_name = network(input_shape=x_train.shape[1:], func_block = func_block, num_classes=num_classes, dropout_rate=dropout_rate)
    
    model.summary()  
        
    train(model, x_train, y_train, x_validation, y_validation,
          epochs_list = [150, 225, 300], name = model_name + get_aug_str(aug), learning_rate = learning_rate, data_augmentation = aug)
    
    if error_anal:
        error_analyze(model, x_validation, y_validation)
        
        

print('\n\n\n\n\n\n\ntest learning_rate = 0.1 densenet k=12 aug no dropout' )
main_dense(0.1, densenet.DenseNet, dropout_rate=None, aug=True)

'''
To train other models
print('\n\n\n\n\n\n\ntest learning_rate = 0.1 densenet k=12 no aug dropout=0.2' )
main_dense(0.1, densenet.DenseNet)
       
print('\n\n\n\n\n\n\ntest learning_rate = 0.1 wide densenet k=48 aug no dropout' )
main_dense(0.1, densenet.DenseNet, dropout_rate=None, aug=True, growth_rate=48, bottleneck=True, compression=0.5)

print('\n\n\n\n\n\n\ntest learning_rate = 0.1 incremental densenet k=12 aug no dropout' )
main_dense(0.1, incremental_densenet.DenseNet, dropout_rate=None, aug=True)
   

print('\n\n\n\n\n\n\ntest learning_rate = 0.1 testnet aug no dropout' )
main_testnet(0.1, testnet.test_net, testnet.ds_block_cascade, dropout_rate=None, aug=True)


print('\n\n\n\n\n\n\ntest learning_rate = 0.1 vgg aug no dropout' )
main_testnet(0.1, testnet.vgg_like, testnet.conv_block, dropout_rate=None, aug=True)
     
'''

#model.save_weights("mcnet.h5")
#model.load_weights(filepath, by_name=False)