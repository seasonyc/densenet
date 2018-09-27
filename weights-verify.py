from vis.utils import utils
from keras.models import load_model
import numpy as np



model = load_model('denseaugmodel-ep0290-loss0.143-acc0.997-val_loss0.320-val_acc0.947.h5')

model.summary()  


layer_names = ['conv2d_65', 'conv2d_66', 'conv2d_67']

for layer_name in layer_names:
    layer_idx = utils.find_layer_idx(model, layer_name)
    
    w = model.layers[layer_idx].get_weights()
    
    w = np.asarray(w)
    print(w.shape)
    print(np.average(np.fabs(w[0,:,:,0:168,:])))

print(np.average(np.fabs(w[0,:,:,169:180,:])))
