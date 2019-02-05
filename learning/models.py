from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.losses import mean_squared_error
from keras.activations import relu
from keras.optimizers import Optimizer
from numpy.linalg import norm
import tensorflow as tf
import keras 
from util.type_convenience import list_broadcast
import json
import os
import random

class SGDcustom_early(Optimizer):
    """Custom stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, random, lr=0.0, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGDcustom_early, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.random = random

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        # random step length
        random_step_size = self.random.gammavariate(1,3)
        #print(grads.eval())
        #grads = grads/norm(grads)*random_step_size

        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g * random_step_size  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGDcustom_early, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SGDcustom(Optimizer):
    """Custom modified stochastic gradient descent version 2,
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
	random: a random object. For generating random step lengths.
	max_norm: float >= 0. Max magnitude of the random step length.
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, random, max_norm, lr=0.0, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGDcustom, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.max_norm = max_norm
        self.initial_decay = decay
        self.nesterov = nesterov
        self.random = random

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        # random step length
        random_coefficient = self.random.gammavariate(1,3)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        #if tf.cond(norm > self.max_norm) is true_fn:
        #    max_norm = norm

        # How to do the above: https://stackoverflow.com/questions/37049411/tensorflow-how-to-convert-scalar-tensor-to-scalar-variable-in-python
        #with tf.Session() as sess:
            #scalar_norm = norm.eval()

        
        max_coefficient = self.max_norm/norm

        random_step_size = max_coefficient*random_coefficient # which has a maxstep of the largest gradient calculated so far (DOES NOT HAVE THIS YET!)
        
        
        #print(grads.eval())
        #grads = grads/norm(grads)*random_step_size

        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g * random_step_size  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGDcustom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
import keras.optimizers
keras.optimizers.custom_optimizers = [SGDcustom_early, SGDcustom]

def get_model_latest_checkpoint_number(path, modelname):    
    latest_checkpoint_ID = -1
    for filename in os.listdir(path.rstrip('/')):
            checkpoint_filename = ""
            if modelname not in filename:
                continue
            if not filename.startswith(modelname + '_checkpoint_') or not filename.endswith('.h5'):
                continue
            else:
                checkpoint_filename = filename

            checkpoint_ID = int(checkpoint_filename.replace(modelname + '_checkpoint_','').replace('.h5',''))

            if checkpoint_ID > latest_checkpoint_ID:
                latest_checkpoint_ID = checkpoint_ID
    return latest_checkpoint_ID

def thesis_load_model(random, path, modelname, optimizer=None):
    if not path.endswith('/'):
        path = path + '/'
    model = None

    if optimizer is None:
        try:
            model = load_model(path + modelname + '.h5')
        except:
            model = load_model(path + modelname + '.h5', compile=False)
            model.compile(optimizer=SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5),
                          loss=mean_squared_error,
                          metrics=['accuracy'])
    else:
        model = load_model(path + modelname + '.h5', compile=False)
        model.compile(optimizer=optimizer,
                      loss=mean_squared_error,
                      metrics=['accuracy'])

    return model

def load_model_and_history(random, path, modelname, optimizer=None):
    # remove the "model_and" part, and just use this for the history and nothing else
    if not path.endswith('/'):
        path = path + '/'

    model = thesis_load_model(random, path, modelname, optimizer=optimizer)
        
    history = []
    try:
        with open(path + modelname + '.json', 'r') as hist_json:
            history = json.load(hist_json)
    except:
        print('Could not load history of ' + modelname + '.')

    return (model, history)

def load_model_and_history_from_latest_checkpoint(random, path, modelname):

    # Implement the use of get_model_latest_checkpoint_number
    # Add handling for when the model is not found, instead of raising crashing excepts

    latest_checkpoint_filename = modelname + '_checkpoint_not_found'
    latest_checkpoint_ID = -1
    for filename in os.listdir(path.rstrip('/')):
            checkpoint_filename = ""
            if not filename.endswith('.h5') or not filename.startswith(modelname + '_checkpoint_'):
                continue
            else:
                checkpoint_filename = filename

            checkpoint_ID = int(checkpoint_filename.replace(modelname + '_checkpoint_','').replace('.h5',''))

            if checkpoint_ID > latest_checkpoint_ID:
                latest_checkpoint_ID = checkpoint_ID
                latest_checkpoint_filename = checkpoint_filename

    if latest_checkpoint_filename is not modelname + '_checkpoint_not_found':
        print('Found latest checkpoint for model: ' + modelname + '. Loading checkpoint ' + latest_checkpoint_filename + ' into memory.')
        return load_model_and_history(random, path, modelname + '_checkpoint_' + str(latest_checkpoint_ID))
    
    return load_model_and_history(path, latest_checkpoint_filename.rstrip('.h5'))

def full_dense_model(input_shape, output_shape, hidden_layers_num_neurons, hidden_layers_activation_functions, optimizer, seed):

    # Allow activation functions to be non-lists for convenience
    if not isinstance(hidden_layers_activation_functions, list):
        hidden_layers_activation_functions = [hidden_layers_activation_functions]
    if len(hidden_layers_activation_functions) < len(hidden_layers_num_neurons):
        list_broadcast(hidden_layers_activation_functions, len(hidden_layers_num_neurons))
    
    model = Sequential()
    not_on_first_layer = False
    for layer_neurons, layer_activation in zip(hidden_layers_num_neurons, hidden_layers_activation_functions):
        if not_on_first_layer:
            model.add(Dense(layer_neurons, kernel_initializer=keras.initializers.glorot_uniform(seed=seed)))
        else:
            model.add(Dense(layer_neurons, input_shape=input_shape, kernel_initializer=keras.initializers.glorot_uniform(seed=seed)))
            not_on_first_layer = True
        model.add(normalization.BatchNormalization())
        model.add(layer_activation)
    
    model.add(Dense(output_shape, activation='linear', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))) # output layer
    
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def full_dense_model_no_batchnorm(input_shape, output_shape, hidden_layers_num_neurons, hidden_layers_activation_functions, optimizer, seed):

    # Allow activation functions to be non-lists for convenience
    if not isinstance(hidden_layers_activation_functions, list):
        hidden_layers_activation_functions = [hidden_layers_activation_functions]
    if len(hidden_layers_activation_functions) < len(hidden_layers_num_neurons):
        list_broadcast(hidden_layers_activation_functions, len(hidden_layers_num_neurons))
    
    model = Sequential()
    not_on_first_layer = False
    for layer_neurons, layer_activation in zip(hidden_layers_num_neurons, hidden_layers_activation_functions):
        if not_on_first_layer:
            model.add(Dense(layer_neurons, kernel_initializer=keras.initializers.glorot_uniform(seed=seed)))
        else:
            model.add(Dense(layer_neurons, input_shape=input_shape, kernel_initializer=keras.initializers.glorot_uniform(seed=seed)))
            not_on_first_layer = True
        model.add(layer_activation)
    
    model.add(Dense(output_shape, activation='linear', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))) # output layer
    
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_old_683_percent_val_acc(input_shape, output_shape, layer_activation, optimizer, seed):
    # Early attempts at this revealed that increasing size a bit increased training set accuracy, but did literally nothing for validation set accuracy.

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(192, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(96, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(48, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(48, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
               
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_larger_but_still_683_percent_val_acc(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(256, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
               
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(256, kernel_initializer=initializer)(inp_block1)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block2)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block3)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
               
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_giant(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(256, kernel_initializer=initializer)(inp_block1)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(256, kernel_initializer=initializer)(inp_block2)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(256, kernel_initializer=initializer)(inp_block3)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
               
    model.compile(optimizer=optimizer, # mini-batch SGD with momentum-like modification to the step
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_SGDcustom_early(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(256, kernel_initializer=initializer)(inp_block1)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block2)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block3)
    x = normalization.BatchNormalization()(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
    
    model.compile(optimizer=optimizer,
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_SGDcustom_early_no_batch_normalization(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(256, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(96, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
    
    model.compile(optimizer=optimizer,
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def test_model_bigger_no_batch_normalization(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(512, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([p, x])
    x = Dense(128, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([f1, x])
    x = Dense(128, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    x = Dense(64, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
    
    model.compile(optimizer=optimizer,
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def SGDcustom_no_batch_normalization_well_connected(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(512, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([inp_block1, p, x])
    x = Dense(512, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([inp_block1, p, f1, x])
    x = Dense(512, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
    
    model.compile(optimizer=optimizer,
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model

def SGDcustom_no_batch_normalization_well_connected_giant(input_shape, output_shape, layer_activation, optimizer, seed):

    #https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
    #https://stackoverflow.com/questions/42445275/merge-outputs-of-different-models-with-different-input-shapes

    #And if necessary this:
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models

    #Alternatively: https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    # and https://github.com/keras-team/keras/issues/369
    
    initializer = kernel_initializer=keras.initializers.glorot_uniform(seed=seed)
    
        
    inp_block1 = Input(shape=input_shape)

    x = Dense(512, kernel_initializer=initializer)(inp_block1)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    p = Dense(3, activation='linear', kernel_initializer=initializer)(x)

    #inp_block2 = merge([p, x], mode='concat', concat_axis=0)
    inp_block2 = concatenate([inp_block1, p, x])
    x = Dense(512, kernel_initializer=initializer)(inp_block2)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    f1 = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    #inp_block3 = merge([f1, x], mode='concat', concat_axis=0)
    inp_block3 = concatenate([inp_block1, p, f1, x])
    x = Dense(512, kernel_initializer=initializer)(inp_block3)
    x = layer_activation(x)
    x = Dense(256, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    x = Dense(128, kernel_initializer=initializer)(x)
    x = layer_activation(x)
    q_dot_ref = Dense(6, activation='linear', kernel_initializer=initializer)(x)

    y = concatenate([p, f1, q_dot_ref])
    model = Model(inputs=inp_block1, outputs=y)
    
    model.compile(optimizer=optimizer,
                   loss=mean_squared_error,
                   metrics=['accuracy'])
    
    return model
