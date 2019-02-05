from session import CAI_args, CAI_random
from simulation.simulation import exit_criteria_at_i_max_only, exit_criteria_at_end_waypoint_or_i_max
from learning.datagen import generate_forced_bias_data, pandas_episode_trajectory_initialize
from learning.rawdata import *
from learning.dataset import *
from learning.training import *
from learning.models import *
from util.file_operations import *
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from util.file_operations import *

def generate_ca_trackpos_val(datapath, numsamples, sample_checkpoint_period, i_max, seed=1, iterate_seed=False):
    
    # Generate data from tracking desired end effector position, with obstacles and collision avoidance (assuming perfect position control)

    (_, _, _, _, rawdatapaths, _, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    generate_forced_bias_data(random, seed, rawdatapaths[1]+'_val', numsamples, sample_checkpoint_period, i_max, 'perfect_position', 5, exit_criteria=exit_criteria_at_end_waypoint_or_i_max)
    print('Generated ' + str(numsamples) + ' position tracking VIK data in ' + datapath + '.' )

def generate_ca_trackpos_test(datapath, numsamples, sample_checkpoint_period, i_max, seed=1, iterate_seed=False):
    
    # Generate data from tracking desired end effector position, with obstacles and collision avoidance (assuming perfect position control)

    (_, _, _, _, rawdatapaths, _, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    generate_forced_bias_data(random, seed, rawdatapaths[1]+'_test', numsamples, sample_checkpoint_period, i_max, 'perfect_position', 5, exit_criteria=exit_criteria_at_end_waypoint_or_i_max)
    print('Generated ' + str(numsamples) + ' position tracking VIK data in ' + datapath + '.' )

def CAVIKAUGee_slot_dataset_val(datapath, seed=4, iterate_seed=False):

    # Construct a dataset using collision avoidance position tracking data only to train collision avoidance

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((0.10,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0, 1, 0)
    max_obstacles = 5

    ((training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs), filenames) = construct(random, (rawdatapaths[1],), max_obstacles, CAVIKAUGee_slots_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    path = datasetpaths[2] + '_val'
    name = datasetnames[2]
    make_path(path)
    
    numpy.save(path + '/' + name + '_filenames', filenames)
    numpy.save(path + '/' + name + '_validation_inputs', validation_inputs)
    numpy.save(path + '/' + name + '_validation_outputs', validation_outputs)

def CAVIKAUGee_slot_dataset_test(datapath, seed=5, iterate_seed=False):

    # Construct a dataset using collision avoidance position tracking data only to train collision avoidance

    (_, rawdatanames, datasetnames, _, rawdatapaths, datasetpaths, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    args_per_selection_method = ((0.20,),)
    selection_methods = (select_random_proportion,)
    split_proportion = (0, 0, 1)
    max_obstacles = 5

    ((training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs), filenames) = construct(random, (rawdatapaths[1],), max_obstacles, CAVIKAUGee_slots_IO_from_rawdata, selection_methods, args_per_selection_method, *split_proportion)
    path = datasetpaths[2] + '_test'
    name = datasetnames[2]
    make_path(path)
    
    numpy.save(path + '/' + name + '_filenames', filenames)
    numpy.save(path + '/' + name + '_test_inputs', test_inputs)
    numpy.save(path + '/' + name + '_test_outputs', test_outputs)

def iterate_model_checkpoints(modelname, index):
    random, _, _ = CAI_random(5)
    path = os.getcwd() + '/data/sessions/CAI/checkpoints/'
    model = thesis_load_model(random, path, modelname + '_checkpoint_' + str(index))
    index += 1
    return model, index
    
def test_dataset(model, dataset, steps):
    if isinstance(dataset, DataGenerator):
        print('notstuck')
        return model.evaluate_generator(dataset, use_multiprocessing=True, workers=2, steps=steps)
    else:
        (test_input, test_output) = dataset
        return model.evaluate(normalize(test_input), normalize(test_output))

def history_dataset(modelname, dataset, steps):
    index = 0
    history = {'acc':[], 'loss':[]}
    while True:
        try:
            model, index = iterate_model_checkpoints(modelname, index)
            acc, loss = test_dataset(model, dataset, steps)
            print(loss, acc)
            history['acc'] += [acc]
            history['loss'] += [loss]
            if index == 26:
                return history
        except Exception as e:
            print(e)
            return history

def make_extrasets():
    datapath = os.getcwd() + '/data'
    random, _, _ = CAI_random(6)

    numsamples = 5000
    sample_checkpoint_period = int(numsamples/2)
    i_max = 1500
    generate_ca_trackpos_val(datapath, numsamples, sample_checkpoint_period, i_max)
    generate_ca_trackpos_test(datapath, numsamples, sample_checkpoint_period, i_max)

    CAVIKAUGee_slot_dataset_val(datapath, seed=4, iterate_seed=False)
    CAVIKAUGee_slot_dataset_test(datapath, seed=5, iterate_seed=False)

    modelnames = ['CAVIKAUGee_sphere_correct_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',#'CAVIKAUGee_sphere_correct_activation12_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',
                  'CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',
                  'CAVIKAUGee_no_obst_input_control_experiment_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    path = datapath + '/datasets/CAI_CAVIKAUGee_slots_dataset/_val/'
    name = 'CAI_CAVIKAUGee_slots_dataset'
    batch_size = 512
    random, _, _ = CAI_random(5)
    modelpath = os.getcwd() + '/data/sessions/CAI/checkpoints/'

    #Val acc and loss:
    validation_input  = numpy.load(path + name + '_validation_inputs.npy')
    validation_output = numpy.load(path + name + '_validation_outputs.npy')

    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (CAVIKAUGee_sphere_num_inputs,), random)

    validation_history = history_dataset(modelnames[0], validation_generator, steps=int(validation_input.shape[0]/batch_size))
    save_json(datapath + '/temp', 'sphere_val_history', validation_history)

    #validation_history = history_dataset(modelnames[1], (validation_input, validation_output), 0)
    validation_history = load_json(datapath + '/temp', 'slot_val_history')
    model = thesis_load_model(random, modelpath, modelnames[1] + '_checkpoint_' + str(25))
    val_acc, val_loss = test_dataset(model, (validation_input, validation_output), 0)
    validation_history['loss'] += [val_acc]
    validation_history['acc'] += [val_loss]
    save_json(datapath + '/temp', 'slot_val_history_updated', validation_history)

    validation_generator = DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (CAVIKAUGee_no_obst_control_num_inputs,), random)
    #validation_history = history_dataset(modelnames[2], validation_generator, steps=int(validation_input.shape[0]/batch_size))
    validation_history = load_json(datapath + '/temp', 'control_val_history')
    model = thesis_load_model(random, modelpath, modelnames[2] + '_checkpoint_' + str(25))
    val_acc, val_loss = test_dataset(model, validation_generator, steps=int(validation_input.shape[0]/batch_size))
    validation_history['loss'] += [val_acc]
    validation_history['acc'] += [val_loss]
    save_json(datapath + '/temp', 'control_val_history_updated', validation_history)

    # Test acc and loss:
    path = datapath + '/datasets/CAI_CAVIKAUGee_slots_dataset/_test/'
    test_input  = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    batch_size = 32
    test_input_mean, test_input_std = normalize_parameters(test_input)
    sphere_test_generator = DataGenerator(test_input, normalize(test_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (CAVIKAUGee_sphere_num_inputs,), random)
    control_test_generator = DataGenerator(test_input, normalize(test_output), CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (CAVIKAUGee_no_obst_control_num_inputs,), random)
    
    datasetwrapper = (sphere_test_generator, (test_input, test_output), control_test_generator)
    modelinitials = ['sphere', 'slot', 'control']
    for modelname, dataset, modelinitial in zip(modelnames, datasetwrapper, modelinitials):
        model = thesis_load_model(random, datapath + '/sessions/CAI/checkpoints', modelname + '_checkpoint_' + str(25))
        test_acc, test_loss = test_dataset(model, dataset, int(test_input.shape[0]/batch_size))
        test_results = {'test_acc': test_acc, 'test_loss': test_loss}
        save_json(datapath + '/temp', modelinitial + '_test_results', test_results)    

if __name__ == '__main__':
    make_extrasets()
    
