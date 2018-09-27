from vis.utils import utils
from keras.models import load_model
import numpy as np



model = load_model('dense_augmodel-ep0300-loss0.112-acc0.999-val_loss0.332-val_acc0.946.h5')

model.summary()  


layer_names = ['conv2d_14', 'conv2d_15', 'conv2d_16']

for layer_name in layer_names:
    layer_idx = utils.find_layer_idx(model, layer_name)
    
    w = model.layers[layer_idx].get_weights()
    
    w = np.asarray(w)
    print(w.shape)
    print(np.average(np.fabs(w[0,:,:,0:168,:])))

print(np.average(np.fabs(w[0,:,:,169:180,:])))