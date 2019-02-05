from session import CAI_args, CAI_random
from learning.rawdata import VIK_IO_from_rawdata, CAVIKee_slots_IO_from_rawdata, CAVIKAUGee_slots_IO_from_rawdata, CAVIKee_sphere_IO_from_rawdata
from learning.dataset import *
from util.file_operations import *
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from util.file_operations import *

def VIK_dataset(datapath, seed=2, iterate_seed=False):

    # Construct a dataset using all data to train velocity inverse kinematic for tracking assuming zero obstacles

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((1,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0.7, 0.15, 0.15)
    max_obstacles = 0

    (dataset, filenames) = construct(random, (rawdatapaths[0],), max_obstacles, VIK_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    save_numpy(datasetpaths[0], datasetnames[0], dataset)
    save_numpy(datasetpaths[0], datasetnames[0] + '_filenames', filenames)

def CAVIKee_slot_dataset(datapath, seed=3, iterate_seed=False):

    # Construct a dataset using collision avoidance position tracking data only to train collision avoidance

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((0.55,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0.7, 0.15, 0.15)
    max_obstacles = 5

    (dataset, filenames) = construct(random, (rawdatapaths[1],), max_obstacles, CAVIKee_slots_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    save_numpy(datasetpaths[1], datasetnames[1], dataset)
    save_numpy(datasetpaths[1], datasetnames[1] + '_filenames', filenames)

def CAVIKAUGee_slot_dataset(datapath, seed=3, iterate_seed=False):

    # Construct a dataset using collision avoidance position tracking data only to train collision avoidance

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((0.55,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0.7, 0.15, 0.15)
    max_obstacles = 5

    ((training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs), filenames) = construct(random, (rawdatapaths[1],), max_obstacles, CAVIKAUGee_slots_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    path = datasetpaths[2]
    name = datasetnames[2]
    make_path(path)
    
    numpy.save(path + '/' + name + '_filenames', filenames)
    numpy.save(path + '/' + name + '_training_inputs', training_inputs)
    numpy.save(path + '/' + name + '_training_outputs', training_outputs)
    numpy.save(path + '/' + name + '_validation_inputs', validation_inputs)
    numpy.save(path + '/' + name + '_validation_outputs', validation_outputs)
    numpy.save(path + '/' + name + '_test_inputs', test_inputs)
    numpy.save(path + '/' + name + '_test_outputs', test_outputs)

def CAVIKee_sphere_dataset(datapath, seed=4, iterate_seed=False):

    # Construct a dataset using collision avoidance position tracking data only to train collision avoidance

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((0.55,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0.7, 0.15, 0.15)
    max_obstacles = 5

    (dataset, filenames) = construct(random, (rawdatapaths[1],), max_obstacles, CAVIKee_sphere_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    save_numpy(datasetpaths[3], datasetnames[3], dataset)
    save_numpy(datasetpaths[3], datasetnames[3] + '_filenames', filenames)

if __name__ == '__main__':

    datapath = os.getcwd() + '/data'

    VIK_dataset(datapath)
    CAVIKee_slot_dataset(datapath) # ee = end to end
    CAVIKAUGee_slot_dataset(datapath)
    CAVIKee_sphere_dataset(datapath)
