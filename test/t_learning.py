from .test_utilities import *
from simulation.workspace import *
from simulation.simulation import *
from learning.datagen import *
from util.file_operations import *
from random import Random
from numpy import *
from param_debug import debug
import os

from simulation.simulation import generate_and_simulate, exit_criteria_at_i_max_only, exit_criteria_at_end_waypoint_or_i_max
from learning.datagen import generate_data, pandas_episode_trajectory_initialize
from learning.models import full_dense_model
from learning.IKBC.parser import parse_dataset as VIK_parser
from learning.dataset import *
from learning.training import train
from util.file_operations import *
from keras.models import load_model
import os
from shutil import copyfile
from math import isnan

import pandas as pd

seed = 185735567
import numpy as np
import random as rand
from tensorflow import set_random_seed
from util.file_operations import remove_files_with_ending

np.random.seed(seed)
rand.seed(seed)
set_random_seed(seed)

def test_pandas_episode_summary():

    print('Warning: test_pandas_episode_summary currently not implemented correctly')
    return
    
    (ca_tasks, waypoints, rtpwrapped) = simple_world_init()

    q_waypoints = tuple((vector([0, 0, 0, 0, 0, 0]) for _ in waypoints))
    
    workspace_dim = []

    #print(pandas_episode_summary('testepisode.txt', q_waypoints, waypoints, ca_tasks, [1, 1]))
    
def test_generate_data(actuators, max_obstacles=5, n_episodes=100):

    # Not a test function?
    
    random = Random()
    random.seed(seed)
    
    test_csv_data_folders = ['/home/daniena/xWorkspace/project/test/data', '/home/daniena/xWorkspace/project/test/data/episodes', '/home/daniena/xWorkspace/project/test/data/episodes_backup']
    for folder in test_csv_data_folders:
        remove_filetype_in_folder(folder, '.csv')
    
    episode_numbers = generate_data(random, 'test/', n_episodes, 10000, actuators, max_obstacles)

    master_summary = pd.read_csv(test_csv_data_folders[0] + '/summary_backup.csv', index_col=[0,1])
    print(max(master_summary.xs('num_timesteps', level='datatype').loc[:,'extra']))
    print(max(master_summary.xs('highest_num_obstacles_avoided_at_same_time', level='datatype').loc[:,'extra']))

#Source: https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
# Seed value
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def test_VIK1_session_self_equality():

    random = rand.Random()
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    session_name = 'VIK1'
    datasetname = session_name + '_dataset'
    datapath = os.getcwd() + '/test/replicate_data'
    sessionpath = datapath + '/sessions/' + session_name
    
    rawdatapaths = [datapath + '/rawdata/ranpos', datapath + '/rawdata/trackpos']
    remove_files_with_ending(rawdatapaths[0], '.csv')
    remove_files_with_ending(rawdatapaths[0] + '/episodes', '.csv')
    remove_files_with_ending(rawdatapaths[0] + '/episodes_backup', '.csv')
    remove_files_with_ending(rawdatapaths[1], '.csv')
    remove_files_with_ending(rawdatapaths[1] + '/episodes', '.csv')
    remove_files_with_ending(rawdatapaths[1] + '/episodes_backup', '.csv')

    datasetpath = datapath + '/datasets/' + datasetname + '/'
    remove_files_with_ending(datasetpath, '.json')
    
    checkpointpath = sessionpath + '/checkpoints'
    remove_files_with_ending(checkpointpath, '.h5')
    remove_files_with_ending(checkpointpath, '.json')
    
    modelspath = sessionpath + '/models'
    remove_files_with_ending(modelspath, '.h5')

    historiespath = sessionpath + '/histories'
    remove_files_with_ending(historiespath, '.json')

    # ----------------------------------------------------------------------------------------------------------------------------
    # Generate data from random configurations each timestep, with one single desired end effector position for 200 such timesteps
    
    numsamples = 20
    i_max = 5 #200
    rawdatapath1 = datapath + '/rawdata/ranpos'
    generate_forced_bias_data(random, seed, rawdatapath1, numsamples, i_max, 'random_position', 0, exit_criteria=exit_criteria_at_i_max_only)
    print('Generated ' + str(numsamples) + '*200 random_position VIK data in ' + datapath + '.')

    numsamples = 20
    i_max = 5 #5000
    rawdatapath2 = datapath + '/rawdata/trackpos'
    generate_forced_bias_data(random, seed, rawdatapath2, numsamples, i_max, 'position', 0, exit_criteria=exit_criteria_at_end_waypoint_or_i_max)
    print('Generated ' + str(numsamples) + ' position tracking VIK data in ' + datapath + '.' )

    # ------------------------------------------------------------------
    # Construct a dataset based on the different types of generated data

    datasetname = session_name + '_dataset'
    datasetpath = datapath + '/datasets/' + datasetname
    rawdatapaths = (rawdatapath1, rawdatapath2)
    selection_methods = [ select_random_proportion for _ in rawdatapaths ]
    args_per_selection_method = [(1, random) for _ in rawdatapaths ]
    split_proportion = (0.7, 0.15, 0.15)

    (dataset, filenames) = construct(random, rawdatapaths, selection_methods, args_per_selection_method, VIK_parser, *split_proportion, )
    save_json(datasetpath, datasetname, dataset)
    save_json(datasetpath, datasetname + '_filenames', filenames)

    # ----------------------------
    # Train the models using keras

    epochs = 1
    batch_size=512
    checkpoint_period=25

    models = [full_dense_model((12,), 6, [24, 24, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [24, 24, 24, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [48, 48, 48], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [48, 48, 48, 48], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96, 96, 96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96, 96, 96, 96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [72, 48, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16, 16, 16], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16, 16, 16, 16, 16, 16], hidden_layer_activation='tanh')]
    modelnames = ['VIK_stack3_small',
                  'VIK_stack4_small',
                  'VIK_stack3_big',
                  'VIK_stack4_big',
                  'VIK_stack3_giant',
                  'VIK_stack4_giant',
                  'VIK_pyramid',
                  'VIK_flat96',
                  'VIK_flat16',
                  'VIK_stack3_mini',
                  'VIK_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = train(modelnames[i], models[i], history, sessionpath + '/checkpoints', dataset, (epochs, batch_size, checkpoint_period))

        (test_loss, test_acc) = model.evaluate(np.asarray(test_input), np.asarray(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(sessionpath + '/models')
        model.save(sessionpath + '/models/' + modelnames[i] + '.h5')
        
        save_json(sessionpath + '/histories/', modelnames[i] + '_history', history)
    
    # ToDo: function that saves the session documentation

    # ------
    # test_run_VIK check if generates reproducible data:

    for _ in range(2):
        this_summary = pd.read_csv(rawdatapaths[0] + '/master_summary.csv')
        other_summary = pd.read_csv(rawdatapaths[0].replace('/data/', '/replicate_data/') + '/master_summary.csv')

        dissimilarity_found = False
        for col_this, col_that in zip(this_summary, other_summary):
            for element_this, element_that in zip(this_summary[col_this], other_summary[col_that]):
                if isinstance(element_this, float) and isinstance(element_that, float):
                    if isnan(element_this) and isnan(element_that):
                        continue
                
                if debug:
                    print('element_this:', element_this, 'element_that:', element_that)
                    
                assert (element_this is element_that) or (element_this == element_that)

                
    for modelname in modelnames:
        print('Checking equality of results in model: ' + modelname)
        print(sessionpath + '/models/' + modelname + '.h5')
        print(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5')
        
        this_model = load_model(sessionpath + '/models/' + modelname + '.h5')
        other_model = load_model(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5')
        
        weights_this = this_model.get_weights()
        weights_other = other_model.get_weights()
        
        # Check that they are the same
        for list_this, list_that in zip(weights_this, weights_other):
            if debug:
                print('list_this')
                print(list_this)
                print('list_that')
                print(list_that)
            assert (list_this == list_that).all()
            
            # Check that the check works
            weights_this[0][0][0] = 1337.13371337
            all_equals = True
            for list_this, list_that in zip(weights_this, weights_other):
                assert (list_this == list_that).any()

                if not (list_this == list_that).all():
                    all_equals = False
                break
                
            assert not all_equals
    
    print('')
    print('Test: test_VIK_session_self_equality, generates rawdata, constructs dataset, trains models, and results all equal to itself, assert TRUE:', True)
    print('')

def test_VIK1_session():

    random = rand.Random()
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    session_name = 'VIK1'
    datasetname = session_name + '_dataset'
    datapath = os.getcwd() + '/test/data'
    sessionpath = datapath + '/sessions/' + session_name
    
    rawdatapaths = [datapath + '/rawdata/ranpos', datapath + '/rawdata/trackpos']
    remove_files_with_ending(rawdatapaths[0], '.csv')
    remove_files_with_ending(rawdatapaths[0] + '/episodes', '.csv')
    remove_files_with_ending(rawdatapaths[0] + '/episodes_backup', '.csv')
    remove_files_with_ending(rawdatapaths[1], '.csv')
    remove_files_with_ending(rawdatapaths[1] + '/episodes', '.csv')
    remove_files_with_ending(rawdatapaths[1] + '/episodes_backup', '.csv')

    datasetpath = datapath + '/datasets/' + datasetname + '/'
    remove_files_with_ending(datasetpath, '.json')
    
    checkpointpath = sessionpath + '/checkpoints'
    remove_files_with_ending(checkpointpath, '.h5')
    remove_files_with_ending(checkpointpath, '.json')
    
    modelspath = sessionpath + '/models'
    remove_files_with_ending(modelspath, '.h5')

    historiespath = sessionpath + '/histories'
    remove_files_with_ending(historiespath, '.json')

    # ----------------------------------------------------------------------------------------------------------------------------
    # Generate data from random configurations each timestep, with one single desired end effector position for 200 such timesteps
    
    numsamples = 20
    i_max = 5 #200
    rawdatapath1 = datapath + '/rawdata/ranpos'
    generate_forced_bias_data(random, seed, rawdatapath1, numsamples, i_max, 'random_position', 0, exit_criteria=exit_criteria_at_i_max_only)
    print('Generated ' + str(numsamples) + '*200 random_position VIK data in ' + datapath + '.')

    numsamples = 20
    i_max = 5 #5000
    rawdatapath2 = datapath + '/rawdata/trackpos'
    generate_forced_bias_data(random, seed, rawdatapath2, numsamples, i_max, 'position', 0, exit_criteria=exit_criteria_at_end_waypoint_or_i_max)
    print('Generated ' + str(numsamples) + ' position tracking VIK data in ' + datapath + '.' )

    # ------------------------------------------------------------------
    # Construct a dataset based on the different types of generated data

    datasetname = session_name + '_dataset'
    datasetpath = datapath + '/datasets/' + datasetname
    rawdatapaths = (rawdatapath1, rawdatapath2)
    selection_methods = [ select_random_proportion for _ in rawdatapaths ]
    args_per_selection_method = [(1, random) for _ in rawdatapaths ]
    split_proportion = (0.7, 0.15, 0.15)

    (dataset, filenames) = construct(random, rawdatapaths, selection_methods, args_per_selection_method, VIK_parser, *split_proportion)
    save_json(datasetpath, datasetname, dataset)
    save_json(datasetpath, datasetname + '_filenames', filenames)

    # ----------------------------
    # Train the models using keras

    epochs = 1
    batch_size=512
    checkpoint_period=25

    models = [full_dense_model((12,), 6, [24, 24, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [24, 24, 24, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [48, 48, 48], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [48, 48, 48, 48], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96, 96, 96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96, 96, 96, 96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [72, 48, 24], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [96], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16, 16, 16], hidden_layer_activation='tanh'),
                  full_dense_model((12,), 6, [16, 16, 16, 16, 16, 16], hidden_layer_activation='tanh')]
    modelnames = ['VIK_stack3_small',
                  'VIK_stack4_small',
                  'VIK_stack3_big',
                  'VIK_stack4_big',
                  'VIK_stack3_giant',
                  'VIK_stack4_giant',
                  'VIK_pyramid',
                  'VIK_flat96',
                  'VIK_flat16',
                  'VIK_stack3_mini',
                  'VIK_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = train(modelnames[i], models[i], history, sessionpath + '/checkpoints', dataset, (epochs, batch_size, checkpoint_period))

        (test_loss, test_acc) = model.evaluate(np.asarray(test_input), np.asarray(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(sessionpath + '/models')
        model.save(sessionpath + '/models/' + modelnames[i] + '.h5')
        
        save_json(sessionpath + '/histories/', modelnames[i] + '_history', history)
    
    # ToDo: function that saves the session documentation to a file so session.py is not cluttered

    # ------
    # test_run_VIK check if generates reproducible data:

    for _ in range(2):
        this_summary = pd.read_csv(rawdatapaths[0] + '/master_summary.csv')
        other_summary = pd.read_csv(rawdatapaths[0].replace('/data/', '/replicate_data/') + '/master_summary.csv')

        dissimilarity_found = False
        for col_this, col_that in zip(this_summary, other_summary):
            for element_this, element_that in zip(this_summary[col_this], other_summary[col_that]):
                if isinstance(element_this, float) and isinstance(element_that, float):
                    if isnan(element_this) and isnan(element_that):
                        continue
                
                if debug:
                    print('element_this:', element_this, 'element_that:', element_that)
                    
                assert (element_this is element_that) or (element_this == element_that)

                
    for modelname in modelnames:
        print('Checking equality of results in model: ' + modelname)
        print(sessionpath + '/models/' + modelname + '.h5')
        print(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5')
        
        this_model = load_model(sessionpath + '/models/' + modelname + '.h5')
        other_model = load_model(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5')
        
        weights_this = this_model.get_weights()
        weights_other = other_model.get_weights()
        
        # Check that they are the same
        for list_this, list_that in zip(weights_this, weights_other):
            if debug:
                print('list_this')
                print(list_this)
                print('list_that')
                print(list_that)
            assert (list_this == list_that).all()
            
            # Check that the check works
            weights_this[0][0][0] = 1337.13371337
            all_equals = True
            for list_this, list_that in zip(weights_this, weights_other):
                assert (list_this == list_that).any()

                if not (list_this == list_that).all():
                    all_equals = False
                break
                
            assert not all_equals
    
    print('')
    print('Test: test_VIK_session, results are all replicated with the same seed, assert TRUE:', True)
    print('')
