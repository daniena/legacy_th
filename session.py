import os
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
from util.file_operations import *

max_obstacles = 5
timestep = 0.02

def session_clear(sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath):

    for rawdatapath in rawdatapaths:
        remove_files_with_ending(rawdatapath, '.csv')
        remove_files_with_ending(rawdatapath + '/episodes', '.csv')
        remove_files_with_ending(rawdatapath + '/episodes_backup', '.csv')

    for datasetpath in datasetpaths:
        remove_files_with_ending(datasetpath, '.json')
        
    remove_files_with_ending(datasetpath, '.json')
    remove_files_with_ending(checkpointpath, '.h5')
    remove_files_with_ending(checkpointpath, '.json')
    remove_files_with_ending(modelspath, '.h5')
    remove_files_with_ending(historiespath, '.json')

def CAI_argnames():
    sessionname = 'CAI' # Collision Avoidance Inference
    rawdatanames = ('ranpos','ca_trackpos')
    datasetnames = (sessionname + '_VIK_dataset', sessionname + '_CAVIKee_slots_dataset', sessionname + '_CAVIKAUGee_slots_dataset', sessionname + '_CAVIKee_sphere_dataset')

    #VIK = Velocity Inverse Kinematics
    #CAVIK = Collision Avoidance Velocity Inverse Kinematics
    
    # IK_modelnames = [ ...
    # IKCA_modelnames = [ ...

    return (sessionname, rawdatanames, datasetnames)

def CAI_pathnames(sessionname, rawdatanames, datasetnames, datapath):

    sessionpath = datapath + '/sessions/' + sessionname
    rawdatapaths = [datapath + '/rawdata/' + rawdataname for rawdataname in rawdatanames]
    datasetpaths = [datapath + '/datasets/' + datasetname + '/' for datasetname in datasetnames]
    checkpointpath = sessionpath + '/checkpoints'
    modelspath = sessionpath + '/models'
    historiespath = sessionpath + '/histories'

    return (sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath)

def CAI_args(datapath):    
    (sessionname, rawdatanames, datasetnames) = CAI_argnames()
    (sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_pathnames(sessionname, rawdatanames, datasetnames, datapath)

    return (sessionname, rawdatanames, datasetnames, sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath)

def CAI_random(seed, iterate_seed=False):
    if iterate_seed:
        seed += 1
    
    #Source: https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
    # Seed value
    seed_value= seed

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    randomstate = np.random.RandomState(seed_value)
    
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)
    
    # 5. Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) # Force tensorflow to use a single thread. numpy.random is not thread safe. Also, GPU operations have nonguaranteed order of operations and therefore will give different results each training session.
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    return (random, randomstate, seed) # https://github.com/keras-team/keras/issues/2280

# Should probably be in its own experiments.py:
#class Experiment():
#    def __init__(self, experiment_name, inference_model, HL_activation, optimizer, LR_scheme, gradient_clipping, epochs, batch_size, checkpoint_period):
        

from run_generate import *
from run_dataset import *
from run_train import *
from run_telemetry import *
from run_extrasets import *
from run_plots import *

if __name__ == '__main__':
    
    datapath = os.getcwd() + '/data'
    
    numsamples = 40000
    sample_checkpoint_period = int(numsamples/2)
    i_max = 400
    #generate_ranpos(datapath, numsamples, sample_checkpoint_period, i_max)
    
    numsamples = 30000
    sample_checkpoint_period = int(numsamples/2)
    i_max = 1500
    generate_ca_trackpos(datapath, numsamples, sample_checkpoint_period, i_max)

    #VIK_dataset(datapath)
    #CAVIKee_slot_dataset(datapath) # ee = end to end
    CAVIKAUGee_slot_dataset(datapath)

    epochs = 1
    batch_size = 32
    checkpoint_period = 1
    #VIK_grid_search(datapath, epochs, batch_size, checkpoint_period)

    #VIK_train_structure_search_RMSprop(datapath, epochs, batch_size, checkpoint_period)
    #VIK_train_structure_search_RMSprop_minibatch(datapath, epochs, batch_size, checkpoint_period)

    #VIK_train_structure_search_RMSprop_tanh_expand_contract(datapath, epochs, batch_size, checkpoint_period)
    #VIK_train_structure_search_RMSprop_ReLU_expand_contract(datapath, epochs, batch_size, checkpoint_period)

    #VIK_train_structure_search_deeper(datapath, epochs, batch_size, checkpoint_period)
    
 
    #CAVIKee_slot_train_structure_search(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_slot_train_structure_search(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKee_sphere_train(datapath, epochs, batch_size, checkpoint_period)

    #CAVIKAUGee_slot_train_SGDcustom_linear(datapath, epochs, batch_size, checkpoint_period)


    epochs = 3
    batch_size = 64
    checkpoint_period = 3
    #VIK_train_structure_search_deeper_clipping_no_batch_normalization(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKee_slot_train_structure_search_deeper_clipping_no_batch_normalization(datapath, epochs, batch_size, checkpoint_period)
    
    epochs = 400
    batch_size = 32
    checkpoint_period = 5
    
    # Simulated Annealing with SGD:
    #CAVIKAUGee_slot_train__giant(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_slot_train_SGDcustom_early(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_slot_train_SGDcustom(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKee_slot_train_aug_control_model(datapath, epochs, batch_size, checkpoint_period)
    
    checkpoint_period = 1
    #CAVIKAUGee_slot_train_SGDcustom_relu_greater_step_bigger_model(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_slot_train_SGDcustom_well_connected_giant(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_slot_train_SGDcustom_well_connected_experiment(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_sphere_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    #CAVIKAUGee_sphere_train_SGDcustom_well_connected_continue(datapath, epochs, batch_size, checkpoint_period)
    
    # vThese were the three included in the thesis v
    CAVIKAUGee_no_obst_input_control_experiment_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    CAVIKAUGee_slot_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period)
    CAVIKAUGee_sphere_train_SGDcustom_well_connected_continue(datapath, epochs, batch_size, checkpoint_period)
    # ^These were the three included in the thesis^
    
    #CAVIKAUGee_sphere_train_SGDnormal_well_connected(datapath, epochs, batch_size, checkpoint_period) #SGDcustom wins over SGD it seems

    make_telemetry() # telemetry_summary is made human readable in telemetry_summary.ipynb
    make_extrasets()
    make_plots()
    
