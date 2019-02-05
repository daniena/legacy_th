import os
import numpy as np
from util import file_operations
from param_debug import debug

def select_all(random, rawdatapath, *selectionargs):
    return os.listdir(rawdatapath + '/episodes')

def select_random_proportion(random, rawdatapath, proportion, *selectionargs):
    filenames = os.listdir(rawdatapath + '/episodes')
    random.shuffle(filenames)

    num_files = int(len(filenames)*proportion)
    return filenames[0:num_files]

def select_same_as_other_dataset(random, datasetpath):
    pass

def shuffle_in_unison(a, b):
    # From https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def from_raw(random, rawdatapath, max_obstacles, module_parser, selection_method, *selectionargs):
    
    raw_data_filenames = selection_method(random, rawdatapath, *selectionargs)

    return (module_parser(random, rawdatapath, raw_data_filenames, max_obstacles), raw_data_filenames)

def split_random(random, inputs, outputs, training_set_proportion, validation_set_proportion, test_set_proportion):
    assert len(inputs) == len(outputs)

    inputs, outputs = shuffle_in_unison(inputs, outputs)# bad, should instead shuffle in unison the test, validation, and training sets after they are made so that they are based off of different episodes, instead of all of the episodes scrambled over all three sets!
    
    #paired = list(zip(inputs,outputs))
    #random.shuffle(paired)
    #inputs, outputs = zip(*paired)

    #inputs = list(inputs)
    #outputs = list(outputs)
    
    num_samples = len(inputs)

    test_end_index = int(num_samples*test_set_proportion)
    validation_end_index = int(num_samples*validation_set_proportion) + test_end_index
    
    test_inputs = inputs[0:test_end_index]
    test_outputs = outputs[0:test_end_index]

    validation_inputs = inputs[test_end_index:validation_end_index]
    validation_outputs = outputs[test_end_index:validation_end_index]

    training_inputs = inputs[validation_end_index: num_samples]
    training_outputs = outputs[validation_end_index: num_samples]

    return (training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs)

def denormalize_assuming_given_descriptors(dataset, mean, std):

    dataset_buffer = dataset*std
    dataset_buffer += mean

    return dataset_buffer

def normalize_assuming_given_descriptors(dataset, mean, std):
    
    dataset_buffer = dataset - mean

    for index, value in enumerate(std):
        if value == 0:
            std[index] = 1
    
    dataset_buffer = dataset_buffer/std

    return dataset_buffer

def normalize_parameters(dataset):
    dataset_mean = np.mean(dataset, axis=0)

    if debug:
        print('dataset_mean')
        print(dataset_mean)

    dataset_temp = dataset - dataset_mean
    
    dataset_std = np.std(dataset_temp, axis=0)

    if debug:
        print('dataset_std')
        print(dataset_std)
    
    return dataset_mean, dataset_std

def normalize(dataset):
    mean, std = normalize_parameters(dataset)
    return normalize_assuming_given_descriptors(dataset, mean, std)

def construct(random, rawdatapaths, max_obstacles, module_parser, selection_methods, args_per_selection_method, *split_proportion):
    inputs, outputs, files = [], [], []
    for i, rawdatapath in enumerate(rawdatapaths):
        print(rawdatapath)
        ((inputs_from_datapath, outputs_from_datapath), filenames_from_datapath) = from_raw(random, rawdatapath, max_obstacles, module_parser, selection_methods[i], *args_per_selection_method[i])
        filenames_from_datapath = [ rawdatapath + '/episodes/' + filename for filename in filenames_from_datapath ] # Low priority, but  done in a way that depends too much on current file structure
        if i == 0:
            files = filenames_from_datapath
            inputs = inputs_from_datapath
            outputs = outputs_from_datapath
        else:
            files = np.vstack((files, filenames_from_datapath))
            inputs = np.vstack((inputs, inputs_from_datapath))
            outputs = np.vstack((outputs, outputs_from_datapath))
    
    return (split_random(random, inputs, outputs, *split_proportion), files)
