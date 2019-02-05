from session import max_obstacles
import json
from learning.plot import *
from learning.dataset import *
from learning.models import load_model_and_history
from util.file_operations import make_path
from param_debug import debug
from learning.rawdata import make_obstacles_buffer
import numpy as np
import keras
import os
import math

def save_random_indexes(length, path, name):
    indexes = np.arange(length)
    np.shuffle(indexes)
    save_numpy(path, name, indexes)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    # Assumes that the input is not normalized, and that the output __is normalized__.
    def __init__(self, inputs, outputs, input_parse, mean, std, batch_size, dim, random):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self.input_parse = input_parse
        self.mean = mean
        self.std = std
        #self.random_indexes_path = random_indexes_path
        self.random = random
        self.epoch_index = 0
        self.on_epoch_end()

        #https://github.com/numpy/numpy/issues/9981

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = load_numpy(self.random_indexes_path, str(self.epoch_index))
        #self.epoch_index += 1
        print('epoch_index:', self.epoch_index)
        self.epoch_index +=1
        
        self.indexes = list(range(len(self.inputs)))
        self.indexes = self.random.sample(self.indexes, len(self.inputs))

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        if not isinstance(self.dim, tuple):
            self.dim = (self.dim,)
        
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size, self.outputs.shape[1]))

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i,:] = self.input_parse(self.inputs[index], max_obstacles, self.mean, self.std)

            # Store class
            y[i] = self.outputs[index]

        return X, y

def train(name, model, history, checkpointpath, dataset, metaparameters):
    print('Now starting the training of: ' + name)
    if debug:
        print('')
        model.summary()
        print('')

    (training_input, training_output, validation_input, validation_output, _, _) = dataset
    (epochs, batch_size, checkpoint_period, callbacks, initial_epoch) = metaparameters

    make_path(checkpointpath)
    
    training_input = normalize(training_input)
    training_output = normalize(training_output)
    validation_input = normalize(validation_input)
    validation_output = normalize(validation_output)

    if np.isnan(training_input).any():
        print('training_input contains NaN')
    if np.isnan(training_output).any():
        print('training_output contains NaN')
    if np.isnan(validation_input).any():
        print('validation_input contains NaN')
    if np.isnan(validation_output).any():
        print('validation_output contains NaN')
    if np.isinf(training_input).any():
        print('training_input contains np.inf')
    if np.isinf(training_output).any():
        print('training_output contains np.inf')
    if np.isinf(validation_input).any():
        print('validation_input contains np.inf')
    if np.isinf(validation_output).any():
        print('validation_output contains np.inf')
    

    
    if not checkpointpath.endswith('/'):
        checkpointpath = checkpointpath + '/'
    
    #try:
    #    latest_checkpoint_num = get_model_latest_checkpoint_number(checkpointpath, name)
    #except:
    #    latest_checkpoint_num = -1
    #    if debug:
    #        print('No previous checkpoint found for ' + name + ', starting at checkpoint 0...')

    if checkpoint_period is None or checkpoint_period > epochs or checkpoint_period < 1:
        checkpoint_period = epochs

    num_checkpoints = int(epochs/checkpoint_period)
    if epochs/checkpoint_period % 1 > 0.0000001:
        num_checkpoints += 1
    
    latest_checkpoint_num = int(math.ceil(initial_epoch/checkpoint_period))  - 1

    epochs += initial_epoch
    epochs_so_far = initial_epoch
    while epochs_so_far < epochs:

        epochs_this_checkpoint = checkpoint_period
        if epochs_so_far + checkpoint_period > epochs:
            epochs_this_checkpoint = epochs - epochs_so_far

        # Keras model.fit has a shuffle=True optional arg, i.e. it shuffles the training data by default
        recent_history = model.fit(training_input, training_output, epochs=epochs_this_checkpoint + epochs_so_far, initial_epoch=epochs_so_far, batch_size=batch_size, callbacks=callbacks, validation_data=(validation_input, validation_output), shuffle=True)
        recent_history = recent_history.history

        for (key, ls) in recent_history.items():
            recent_history[key] = [ float(e) for e in ls ]

        epochs_so_far += epochs_this_checkpoint
        
        if history: # is not empty, then concatenate
            history['val_loss'] += recent_history['val_loss']
            history['val_acc'] += recent_history['val_acc']
            history['loss'] += recent_history['loss']
            history['acc'] += recent_history['acc']
            try:
                history['lr'] += recent_history['lr']
            except:
                pass
        else: # recent history is all history there is
            history = recent_history                   
                

        latest_checkpoint_num += 1
        model.save(checkpointpath + name + '_checkpoint_' + str(latest_checkpoint_num) + '.h5')
        print('')
        print('Saving trained ' + name + ' checkpoint', latest_checkpoint_num, 'epochs this step:', epochs_this_checkpoint)
        print('Epochs completed: ' + str(epochs_so_far) + '/' + str(epochs))
        with open(checkpointpath + name + '_checkpoint_' + str(latest_checkpoint_num) + '.json', 'w') as hist_json:
            json.dump(history, hist_json)
    
    return (model, history)

def train_generator(name, model, history, checkpointpath, training_generator, validation_generator,  metaparameters):
    print('Now starting the training of: ' + name)
    if debug:
        print('')
        model.summary()
        print('')

    (epochs, batch_size, checkpoint_period, callbacks, initial_epoch) = metaparameters

    make_path(checkpointpath)

    if not checkpointpath.endswith('/'):
        checkpointpath = checkpointpath + '/'
    
    #try:
    #    latest_checkpoint_num = get_model_latest_checkpoint_number(checkpointpath, name)
    #except:
    #    latest_checkpoint_num = -1
    #    if debug:
    #        print('No previous checkpoint found for ' + name + ', starting at checkpoint 0...')

    if checkpoint_period is None or checkpoint_period > epochs or checkpoint_period < 1:
        checkpoint_period = epochs

    num_checkpoints = int(epochs/checkpoint_period)
    if epochs/checkpoint_period % 1 > 0.0000001:
        num_checkpoints += 1

    latest_checkpoint_num = int(math.ceil(initial_epoch/checkpoint_period))  - 1

    epochs += initial_epoch
    epochs_so_far = initial_epoch
    while epochs_so_far < epochs:

        epochs_this_checkpoint = checkpoint_period
        if epochs_so_far + checkpoint_period > epochs:
            epochs_this_checkpoint = epochs - epochs_so_far

        # Keras model.fit has a shuffle=True optional arg, i.e. it shuffles the training data by default
        recent_history = model.fit_generator(training_generator, epochs=epochs_this_checkpoint + epochs_so_far, initial_epoch=epochs_so_far, callbacks=callbacks, validation_data=validation_generator, use_multiprocessing=False, workers=0, max_queue_size=10, shuffle=False) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array, https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/ <- this one did not actually include the answer but lead to the answer! The problem is that as long as it has workers at all, it will use threading of some kind... This was discovered when keras and the generator tried to print out at the same time, leading to keras overwriting the data in the buffer of the generator up until the point where keras' string was shorter than the string output by the generator. Setting workers=0 and max_queue_size=10 worked, though I do not have time to test if only one of them would suffice.
        recent_history = recent_history.history

        for (key, ls) in recent_history.items():
            recent_history[key] = [ float(e) for e in ls ]

        epochs_so_far += epochs_this_checkpoint

        if history: # is not empty, then concatenate
            history['val_loss'] += recent_history['val_loss']
            history['val_acc'] += recent_history['val_acc']
            history['loss'] += recent_history['loss']
            history['acc'] += recent_history['acc']
            try:
                history['lr'] += recent_history['lr']
            except:
                pass
        else: # recent history is all history there is
            history = recent_history

        latest_checkpoint_num += 1
        model.save(checkpointpath + name + '_checkpoint_' + str(latest_checkpoint_num) + '.h5')
        print('')
        print('Saving trained ' + name + ' checkpoint', latest_checkpoint_num, 'epochs this step:', epochs_this_checkpoint)
        print('Epochs completed: ' + str(epochs_so_far) + '/' + str(epochs))
        with open(checkpointpath + name + '_checkpoint_' + str(latest_checkpoint_num) + '.json', 'w') as hist_json:
            json.dump(history, hist_json)
    
    return (model, history)
