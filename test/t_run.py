from session import *
from param_debug import debug, view, exhaustive
import os
from run_generate import *
from run_dataset import *
from run_train import *
import pandas as pd
from math import isnan
from keras.models import load_model
from learning.models import SGDcustom_early

def _test_CAI_session(datapath):

    numsamples = 20 # More for a bit more thoroughness..
    samples_checkpoint_period = int(numsamples/2)
    i_max = 20
    epochs = 5
    batch_size = 2
    checkpoint_period = 2
    
    # RUN CAI
    if debug:
        print('generate trackpos')
    generate_ca_trackpos(datapath, numsamples, samples_checkpoint_period, i_max)
    if debug:
        print('generate ranpos')
    generate_ranpos(datapath, numsamples, samples_checkpoint_period, i_max)

    if debug:
        print('construct VIK_dataset')
    VIK_dataset(datapath)
    if debug:
        print('construct CAVIKee_slot_dataset')
    CAVIKee_slot_dataset(datapath) # ee = end to end
    if debug:
        print('construct CAVIKAUGee_slot_dataset')
    CAVIKAUGee_slot_dataset(datapath)
    if debug:
        print('construct CAVIKee_sphere_dataset')
    CAVIKee_sphere_dataset(datapath)
    
    if debug:
        print('IK train structure search')
    CAVIKAUGee_sphere_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    CAVIKAUGee_slot_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    CAVIKAUGee_no_obst_input_control_experiment_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    

def _CAI_new_session(datapath):

    
    (sessionname, rawdatanames, datasetnames, sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    session_clear(sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath)
    
    _test_CAI_session(datapath)

    for i in range(2):
        print('Checking equality of results in: ' + rawdatapaths[i] + '/episodes_summaries.csv' + ' vs ' + rawdatapaths[i].replace('/data/', '/replicate_data/') + '/episodes_summaries.csv')
        this_summary = pd.read_csv(rawdatapaths[i] + '/episodes_summaries.csv')
        other_summary = pd.read_csv(rawdatapaths[i].replace('/data/', '/replicate_data/') + '/episodes_summaries.csv')

        for col_this, col_that in zip(this_summary, other_summary):
            for element_this, element_that in zip(this_summary[col_this], other_summary[col_that]):
                if isinstance(element_this, float) and isinstance(element_that, float):
                    if isnan(element_this) and isnan(element_that):
                        continue
                
                if debug:
                    print('element_this:', element_this, 'element_that:', element_that)
                    
                assert (element_this is element_that) or (element_this == element_that)

    modelnames = os.listdir(modelspath)
    modelnames = [ re.findall('[^/]*[^.h5]', modelname)[0] for modelname in modelnames ]
    
    for modelname in modelnames:
        print('Checking equality of results in model: ' + modelname)
        print(sessionpath + '/models/' + modelname + '.h5')
        print(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5')
        
        this_model = load_model(sessionpath + '/models/' + modelname + '.h5', compile=False)
        other_model = load_model(sessionpath.replace('/data/', '/replicate_data/') + '/models/' + modelname + '.h5', compile=False)
        
        weights_this = this_model.get_weights()
        weights_other = other_model.get_weights()
        
        # Check that they are the same
        for list_this, list_that in zip(weights_this, weights_other):
            if debug:
                for row_this, row_that in zip(list_this, list_that):
                    print('this: ', row_this)
                    print('that: ', row_that)
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
    
def test_CAI_session():

    datapath = os.getcwd() + '/test/replicate_data'
    _CAI_new_session(datapath)

    print('')
    print('Test: test_CAI_session, runs + generated data and trained models equal themselves, assert TRUE:', True)
    print('')

    datapath = os.getcwd() + '/test/data'
    _CAI_new_session(datapath)

    print('')
    print('Test: test_CAI_session, results are all replicated with the same seed, assert TRUE:', True)
    print('')
