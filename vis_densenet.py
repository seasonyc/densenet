from vis.utils import utils
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
from keras.models import load_model
from vis.visualization import get_num_filters
import numpy as np
import scipy


model = load_model('dense_augmodel-ep0300-loss0.112-acc0.999-val_loss0.332-val_acc0.946.h5')

model.summary()  

'''
Don't need to swap softmax with linear because we won't visualize the classification layer
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)
'''

vis_images = []
layer_names = ['activation_7', 'activation_13', 'activation_14', 'activation_15', 'activation_23', 'activation_33', 'activation_39']

'''
Can also visualize other layers
layer_names = ['conv2d_xx']
layer_names = ['batch_normalization_xx']
'''

'''
Can print layer weights
layer_idx = utils.find_layer_idx(model, 'batch_normalization_xx')
print(model.layers[layer_idx].get_weights())
'''

for layer_name in layer_names:
    layer_idx = utils.find_layer_idx(model, layer_name)
    
    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))
    
    vis_images = []
    for idx in filters:
        #if idx % 2 == 0:
        # Generate input image for each filter.
        img = visualize_activation(model, layer_idx, filter_indices=idx
                                   , act_max_weight=10, lp_norm_weight=0.01, tv_weight=0.05#, lp_norm_weight=0, tv_weight=0
                                   , max_iter=200, input_modifiers=[Jitter()])  #, verbose=True
        
        vis_images.append(img)
        print(idx)

    stitched = utils.stitch_images(vis_images, cols=24)
    scipy.misc.imsave(layer_name + '.png', stitched)
