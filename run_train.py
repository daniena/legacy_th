from session import CAI_args, CAI_random
from learning.IKBC.trainer import train as VIK_train
from learning.models import *
from learning.dataset import *
import learning.training
from util.file_operations import *
from keras.models import load_model
from keras.layers import LeakyReLU, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler
import os
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
from util.file_operations import *
from learning.rawdata import VIK_num_inputs, CAVIKee_slots_num_inputs, CAVIKAUGee_slots_num_inputs, CAVIKAUGee_slots_num_outputs, CAVIKee_sphere_num_inputs, CAVIKee_sphere_input_from_CAVIKee_slots_IO, CAVIKAUGee_sphere_num_inputs, CAVIKAUGee_sphere_num_outputs, CAVIKAUGee_no_obst_control_num_inputs, CAVIKAUGee_no_obst_control_num_outputs, CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO 
import math

def VIK_grid_search(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])
    _, _, _, _, test_input, test_output = dataset
    
    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras
    
    activations = [Activation('tanh'), LeakyReLU(alpha=0.05)]
    optimizers = [RMSprop(lr=0.02, clipvalue=0.5), SGD(lr=0.1, decay=0, clipvalue=0.5), SGDcustom(random, 0.5, lr=0.1, decay=0, clipvalue=0.5), Adam(clipvalue=0.5)]

    activation_names = ['tanh', 'relu']
    optimizer_names = ['rms', 'SGD', 'SGDcustom', 'Adam']
    
    modelnames = ['VIK_1flat16',
                  'VIK_3stack16',
                  'VIK_3stack96',
                  'VIK_pyramid_72_48_24',
                  'VIK_4stack96',
                  'VIK_6stack16',
                  'VIK_6stack32',
                  'VIK_6stack64',
                  'VIK_6stack128',
                  'VIK_7stack128']
    orig_checkpointpath = checkpointpath
    orig_historiespath = historiespath
    orig_modelspath = modelspath

    for optimizer, optimizer_name in zip(optimizers, optimizer_names):
        for activation, activation_name in zip(activations, activation_names):
            
            models = [full_dense_model((i_size,), o_size, [16], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [16, 16, 16], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [96, 96, 96], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [72, 48, 24], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [96, 96, 96, 96], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [16, 16, 16, 16, 16, 16], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [128, 128, 128, 128, 128, 128], activation, optimizer, seed),
                      full_dense_model((i_size,), o_size, [128, 128, 128, 128, 128, 128, 128], activation, optimizer, seed)]

            modelbatch_name = 'VIK_' + optimizer_name + activation_name

            checkpointpath = orig_checkpointpath + '/' + modelbatch_name
            historiespath = orig_historiespath + '/' + modelbatch_name
            modelspath = orig_modelspath + '/' + modelbatch_name

            for i in range(len(models)):
                history = {}
                model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

                (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
                print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
                
                make_path(modelspath)
                make_path(historiespath)
                model.save(modelspath + '/' + modelnames[i] + '.h5')
                save_json(historiespath + '/', modelnames[i] + '_history', history)
                save_json(historiespath + '/', modelnames[i] + '_test_loss_and_acc', [test_loss, test_acc])

def VIK_train_structure_search_RMSprop(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('tanh')
    optimizer = RMSprop(lr=0.001)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [24, 24, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [24, 24, 24, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [96, 96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [72, 48, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16, 16, 16, 16], activation, optimizer, seed)]
    modelnames = ['VIKrms_stack3_small',
                  'VIKrms_stack4_small',
                  'VIKrms_stack3_giant',
                  'VIKrms_stack4_giant',
                  'VIKrms_pyramid',
                  'VIKrms_flat16',
                  'VIKrms_stack3_mini',
                  'VIKrms_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def VIK_train_structure_search_RMSprop_minibatch(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):

    epochs = 5
    checkpoint_period = 1
    batch_size = 4
    
    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('tanh')
    optimizer = RMSprop(lr=0.001)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [24, 24, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [24, 24, 24, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [96, 96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [72, 48, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16, 16, 16, 16], activation, optimizer, seed)]
    modelnames = ['VIKrmsminib_stack3_small',
                  'VIKrmsminib_stack4_small',
                  'VIKrmsminib_stack3_giant',
                  'VIKrmsminib_stack4_giant',
                  'VIKrmsminib_pyramid',
                  'VIKrmsminib_flat16',
                  'VIKrmsminib_stack3_mini',
                  'VIKrmsminib_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def VIK_train_structure_search_RMSprop_tanh_expand_contract(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):
    
    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('tanh')
    optimizer = RMSprop(lr=0.002)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [192, 24], activation, optimizer, seed)]
    modelnames = ['VIKrmstanh_expand_contract']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def VIK_train_structure_search_RMSprop_ReLU_expand_contract(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):
    
    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    optimizer = RMSprop(lr=0.002)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [192, 24], activation, optimizer, seed)]
    modelnames = ['VIKrmsrelu_expand_contract']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def VIK_train_structure_search_deeper(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    relu_activation = LeakyReLU(alpha=0.05)
    tanh_activation = Activation('tanh')
    rms_optimizer = RMSprop(lr=0.001)
    sgd_optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], tanh_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], tanh_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], tanh_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], tanh_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], tanh_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], tanh_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], tanh_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], tanh_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], relu_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], relu_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], relu_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], relu_activation, rms_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], relu_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], relu_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], relu_activation, sgd_optimizer, seed),
              full_dense_model((i_size,), o_size, [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], relu_activation, sgd_optimizer, seed),]
    modelnames = ['VIKrms_relu_deep_12stack_narrow',
                  'VIKrms_relu_deep_18stack_narrow',
                  'VIKrms_relu_deep_12stack_medium',
                  'VIKrms_relu_deep_12stack_medium',
                  'VIKsgd_relu_deep_12stack_narrow',
                  'VIKsgd_relu_deep_18stack_narrow',
                  'VIKsgd_relu_deep_12stack_medium',
                  'VIKsgd_relu_deep_12stack_medium']
    

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def VIK_train_structure_search_deeper_clipping_no_batch_normalization(datapath, epochs, batch_size, checkpoint_period, seed=6, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[0], datasetnames[0])

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    i_size = 6 + 3
    o_size = 6

    # ----------------------------
    # Train the models using keras

    clipnorm = 0.7
    tanh_activation = Activation('tanh')
    relu_activation = LeakyReLU(alpha=0.05)
    linear_activation = Activation('linear')
    rms_optimizer = RMSprop(lr=0.003, clipvalue=clipnorm)
    sgd_optimizer = SGD(lr=0, clipvalue=clipnorm)
    adam_optimizer = Adam(clipvalue=clipnorm)
    SGDcustom_early_optimizer = SGDcustom_early(random, clipnorm)
    SGDcustom_optimizer = SGDcustom(random, clipnorm)
    models = [full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, SGDcustom_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, SGDcustom_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, SGDcustom_optimizer, seed),]
    modelnames = ['VIKclip_nobatchnorm_deep_pyramid_tanh_adam',
                  'VIKclip_nobatchnorm_deep_pyramid_relu_adam',
                  'VIKclip_nobatchnorm_deep_pyramid_linear_adam',
                  'VIKclip_nobatchnorm_deep_pyramid_tanh_sgd',
                  'VIKclip_nobatchnorm_deep_pyramid_relu_sgd',
                  'VIKclip_nobatchnorm_deep_pyramid_linear_sgd',
                  'VIKclip_nobatchnorm_deep_pyramid_tanh_SGDcustom_early',
                  'VIKclip_nobatchnorm_deep_pyramid_relu_SGDcustom_early',
                  'VIKclip_nobatchnorm_deep_pyramid_linear_SGDcustom_early',
                  'VIKclip_nobatchnorm_deep_pyramid_tanh_SGDcustom',
                  'VIKclip_nobatchnorm_deep_pyramid_relu_SGDcustom',
                  'VIKclip_nobatchnorm_deep_pyramid_linear_SGDcustom']
    

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        callbacks = [learningrate_schedule]
        if i < 3:
            callbacks = [] # removing learningrate scheduler from adam, since it is adaptive, and rms, since it requires a smaller step size.
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, callbacks, 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)
        
def CAVIKee_slot_train_structure_search(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[1], datasetnames[1])

    max_obstacles = 5
    i_size = 6 + 3 + 4*max_obstacles
    o_size = 6

    # ----------------------------
    # Train the models using keras

    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('tanh')
    optimizer = RMSprop(lr=0.001)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [24, 24, 24], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [96, 96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16, 16, 16, 16], activation, optimizer, seed)]
    modelnames = ['CAVIKee_slot_stack3_small',
                  'CAVIKee_slot_stack4_giant',
                  'CAVIKee_slot_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKee_slot_train_aug_control_model(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)    
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)
    
    dataset = load_numpy(datasetpaths[1], datasetnames[1])

    max_obstacles = 5
    i_size = 6 + 3 + 4*max_obstacles
    o_size = 6

    # ----------------------------
    # Train the models using keras

    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.001)
    optimizer = SGD(lr=0, decay=0)
    models = [full_dense_model((i_size,), o_size, [256, 256, 256, 256], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [256, 128, 96, 96], activation, optimizer, seed)]
    modelnames = ['CAVIKee_slot_SGD_simulated_annealing_giant_aug_control_model',
                  'CAVIKee_slot_SGD_simulated_annealing_aug_control_model']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKee_slot_train_structure_search_deeper_clipping_no_batch_normalization(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[1], datasetnames[1])

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    max_obstacles = 5
    i_size = 6 + 3 + 4*max_obstacles
    o_size = 6

    # ----------------------------
    # Train the models using keras

    clipnorm = 0.7
    tanh_activation = Activation('tanh')
    relu_activation = LeakyReLU(alpha=0.05)
    linear_activation = Activation('linear')
    rms_optimizer = RMSprop(lr=0.003, clipvalue=clipnorm)
    sgd_optimizer = SGD(lr=0, clipvalue=clipnorm)
    adam_optimizer = Adam(clipvalue=clipnorm)
    SGDcustom_early_optimizer = SGDcustom_early(random, clipnorm)
    SGDcustom_optimizer = SGDcustom(random, clipnorm)
    models = [full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, adam_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, sgd_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, SGDcustom_early_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], tanh_activation, SGDcustom_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], relu_activation, SGDcustom_optimizer, seed),
              full_dense_model_no_batchnorm((i_size,), o_size, [512, 256, 256, 256, 128, 128, 128, 128, 128, 128], linear_activation, SGDcustom_optimizer, seed),]
    modelnames = ['CAVIK_slot_clip_nobatchnorm_deep_pyramid_tanh_adam',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_relu_adam',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_linear_adam',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_tanh_sgd',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_relu_sgd',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_linear_sgd',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_tanh_SGDcustom_early',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_relu_SGDcustom_early',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_linear_SGDcustom_early',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_tanh_SGDcustom',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_relu_SGDcustom',
                  'CAVIK_slot_clip_nobatchnorm_deep_pyramid_linear_SGDcustom']
    

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    _, _, _, _, test_input, test_output = dataset

    for i in range(len(models)):
        history = {}
        callbacks = [learningrate_schedule]
        if i < 3:
            callbacks = [] # removing learningrate scheduler from adam, since it is adaptive, and rms, since it requires a smaller step size.
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, dataset, (epochs, batch_size, checkpoint_period, callbacks, 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_giant(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    print(training_outputs)

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002)
    optimizer = SGD(lr=0, decay=0) 
    
    models = [test_model_giant((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGD_simulated_annealing_giant']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_early(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom_early(random, lr=0, decay=0, clipvalue=0.5)
    
    models = [test_model_SGDcustom_early_no_batch_normalization((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGDcustom_early_clipping_no_batchnormalization_simulated_annealing']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [test_model_SGDcustom_early_no_batch_normalization((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_well_connected_experiment(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_relu_greater_step_bigger_model(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 4.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('linear')
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.7, lr=0, decay=0, clipvalue=0.7)
    
    models = [test_model_bigger_no_batch_normalization((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_relu_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_well_connected_giant(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.5
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected_giant((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_giant_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_slot_train_SGDcustom_linear(datapath, epochs, batch_size, checkpoint_period, seed=7, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.03
        drop = 0.97
        epochs_drop = 4.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_inputs = numpy.load(path + name + '_training_inputs.npy')
    training_outputs = numpy.load(path + name + '_training_outputs.npy')
    validation_inputs  = numpy.load(path + name + '_validation_inputs.npy')
    validation_outputs = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_slots_num_inputs
    o_size = CAVIKAUGee_slots_num_outputs

    # ----------------------------
    # Train the models using keras
    
    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('linear')
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002) 
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [test_model_SGDcustom_early_no_batch_normalization((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_slot_linear_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train(modelnames[i], models[i], history, checkpointpath, (training_inputs, training_outputs, validation_inputs, validation_outputs, test_input, test_output), (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate(normalize(test_input), normalize(test_output))
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)
        
def CAVIKee_sphere_train(datapath, epochs, batch_size, checkpoint_period, seed=8, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    dataset = load_numpy(datasetpaths[1], datasetnames[1])

    max_obstacles = 5
    i_size = CAVIKee_sphere_num_inputs
    o_size = 6

    # ----------------------------
    # Train the models using keras

    #activation = LeakyReLU(alpha=0.05)
    activation = Activation('tanh')
    optimizer = RMSprop(lr=0.001)
    #optimizer = SGD(lr=0.01)
    models = [full_dense_model((i_size,), o_size, [96, 96, 96, 96], activation, optimizer, seed),
              full_dense_model((i_size,), o_size, [16, 16, 16, 16, 16, 16], activation, optimizer, seed)]
    modelnames = ['CAVIKee_sphere_stack4_giant',
                  'CAVIKee_sphere_stack6_mini']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]
    training_input, training_output, validation_input, validation_output, test_input, test_output = dataset

    training_input_mean, training_input_std = normalize_parameters(training_input)
    training_generator = learning.training.DataGenerator(training_input, normalize(training_output), CAVIKee_sphere_input_from_CAVIKee_slots_IO, training_input_mean[0:9], training_input_std[0:9], batch_size, (i_size,), random)
    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = learning.training.DataGenerator(validation_input, normalize(validation_output), CAVIKee_sphere_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (i_size,), random)
    test_input_mean, test_input_std = normalize_parameters(test_input)
    test_generator = learning.training.DataGenerator(test_input, normalize(test_output), CAVIKee_sphere_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (i_size,), random)

    for i in range(len(models)):
        history = {}
        
        model, history = learning.training.train_generator(modelnames[i], models[i], history, checkpointpath, training_generator, validation_generator, (epochs, batch_size, checkpoint_period, [], 0))

        (test_loss, test_acc) = model.evaluate_generator(test_generator, use_multiprocessing=False, workers=0) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array
        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_sphere_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period, seed=8, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_input = numpy.load(path + name + '_training_inputs.npy')
    training_output = numpy.load(path + name + '_training_outputs.npy')
    validation_input  = numpy.load(path + name + '_validation_inputs.npy')
    validation_output = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_sphere_num_inputs
    o_size = CAVIKAUGee_sphere_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002)
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_sphere_correct_activation12_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    training_input_mean, training_input_std = normalize_parameters(training_input)
    training_generator = learning.training.DataGenerator(training_input, normalize(training_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, training_input_mean[0:9], training_input_std[0:9], batch_size, (i_size,), random)
    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = learning.training.DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (i_size,), random)
    test_input_mean, test_input_std = normalize_parameters(test_input)
    test_generator = learning.training.DataGenerator(test_input, normalize(test_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (i_size,), random)

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train_generator(modelnames[i], models[i], history, checkpointpath, training_generator, validation_generator, (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate_generator(test_generator, use_multiprocessing=False, workers=0) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array

        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_sphere_train_SGDcustom_well_connected_continue(datapath, epochs, batch_size, checkpoint_period, seed=100, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_input = numpy.load(path + name + '_training_inputs.npy')
    training_output = numpy.load(path + name + '_training_outputs.npy')
    validation_input  = numpy.load(path + name + '_validation_inputs.npy')
    validation_output = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_sphere_num_inputs
    o_size = CAVIKAUGee_sphere_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002)
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)

    model, history = load_model_and_history(random, checkpointpath, 'CAVIKAUGee_sphere_correct_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected_checkpoint_10', optimizer=optimizer)
    
    #models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    models = [model]
    modelnames = ['CAVIKAUGee_sphere_correct_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    training_input_mean, training_input_std = normalize_parameters(training_input)
    training_generator = learning.training.DataGenerator(training_input, normalize(training_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, training_input_mean[0:9], training_input_std[0:9], batch_size, (i_size,), random)
    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = learning.training.DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (i_size,), random)
    test_input_mean, test_input_std = normalize_parameters(test_input)
    test_generator = learning.training.DataGenerator(test_input, normalize(test_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (i_size,), random)

    for i in range(len(models)):
        #history = {}
        model, history = learning.training.train_generator(modelnames[i], models[i], history, checkpointpath, training_generator, validation_generator, (epochs, batch_size, checkpoint_period, [learningrate_schedule], len(history['acc'])))

        (test_loss, test_acc) = model.evaluate_generator(test_generator, use_multiprocessing=False, workers=0) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array

        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_no_obst_input_control_experiment_train_SGDcustom_well_connected(datapath, epochs, batch_size, checkpoint_period, seed=8, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_input = numpy.load(path + name + '_training_inputs.npy')
    training_output = numpy.load(path + name + '_training_outputs.npy')
    validation_input  = numpy.load(path + name + '_validation_inputs.npy')
    validation_output = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_no_obst_control_num_inputs
    o_size = CAVIKAUGee_no_obst_control_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002)
    #optimizer = SGD(lr=0, decay=0)
    optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_no_obst_input_control_experiment_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    training_input_mean, training_input_std = normalize_parameters(training_input)
    training_generator = learning.training.DataGenerator(training_input, normalize(training_output), CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO, training_input_mean[0:9], training_input_std[0:9], batch_size, (i_size,), random)
    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = learning.training.DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (i_size,), random)
    test_input_mean, test_input_std = normalize_parameters(test_input)
    test_generator = learning.training.DataGenerator(test_input, normalize(test_output), CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (i_size,), random)

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train_generator(modelnames[i], models[i], history, checkpointpath, training_generator, validation_generator, (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate_generator(test_generator, use_multiprocessing=False, workers=0) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array

        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)

def CAVIKAUGee_sphere_train_SGDnormal_well_connected(datapath, epochs, batch_size, checkpoint_period, seed=8, iterate_seed=False):

    (sessionname, _, datasetnames, sessionpath, _, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)

    # Source: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.97
        epochs_drop = 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    learningrate_schedule = LearningRateScheduler(step_decay)

    path = datasetpaths[2]
    name = datasetnames[2]
    training_input = numpy.load(path + name + '_training_inputs.npy')
    training_output = numpy.load(path + name + '_training_outputs.npy')
    validation_input  = numpy.load(path + name + '_validation_inputs.npy')
    validation_output = numpy.load(path + name + '_validation_outputs.npy')
    test_input = numpy.load(path + name + '_test_inputs.npy')
    test_output = numpy.load(path + name + '_test_outputs.npy')

    max_obstacles = 5
    i_size = CAVIKAUGee_sphere_num_inputs
    o_size = CAVIKAUGee_sphere_num_outputs

    # ----------------------------
    # Train the models using keras
    
    activation = LeakyReLU(alpha=0.05)
    #activation = Activation('tanh')
    #optimizer = RMSprop(lr=0.002)
    optimizer = SGD(lr=0, decay=0, clipvalue=0.35)
    #optimizer = SGDcustom(random, 0.5, lr=0, decay=0, clipvalue=0.5)
    
    models = [SGDcustom_no_batch_normalization_well_connected((i_size,), o_size, activation, optimizer, seed)]
    modelnames = ['CAVIKAUGee_sphere_correct_activation12_ReLU_SGDnormal_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']

    test_losses = [ 0.0 for _ in range(len(models)) ]
    test_accuracies = [ 0.0 for _ in range(len(models)) ]

    training_input_mean, training_input_std = normalize_parameters(training_input)
    training_generator = learning.training.DataGenerator(training_input, normalize(training_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, training_input_mean[0:9], training_input_std[0:9], batch_size, (i_size,), random)
    validation_input_mean, validation_input_std = normalize_parameters(validation_input)
    validation_generator = learning.training.DataGenerator(validation_input, normalize(validation_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, validation_input_mean[0:9], validation_input_std[0:9], batch_size, (i_size,), random)
    test_input_mean, test_input_std = normalize_parameters(test_input)
    test_generator = learning.training.DataGenerator(test_input, normalize(test_output), CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO, test_input_mean[0:9], test_input_std[0:9], batch_size, (i_size,), random)

    for i in range(len(models)):
        history = {}
        model, history = learning.training.train_generator(modelnames[i], models[i], history, checkpointpath, training_generator, validation_generator, (epochs, batch_size, checkpoint_period, [learningrate_schedule], 0))

        (test_loss, test_acc) = model.evaluate_generator(test_generator, use_multiprocessing=False, workers=0) # multithreading makes seeded numpy.random results irreproducable unless numpy.RandomState is passed, https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array

        print('Model ' + modelnames[i] + ' finished training with test_loss, test_acc:', test_loss, test_acc)
        test_losses[i] = test_loss
        test_accuracies[i] = test_acc

        make_path(modelspath)
        make_path(historiespath)
        model.save(modelspath + '/' + modelnames[i] + '.h5')
        save_json(historiespath + '/', modelnames[i] + '_history', history)
    
if __name__ == '__main__':

    datapath = os.getcwd() + '/data'
    
    epochs = 5
    batch_size = 32
    checkpoint_period = 25
    
    VIK_train_structure_search_RMSprop(datapath, epochs, batch_size, checkpoint_period)
    VIK_train_structure_search_SGD(datapath, epochs, batch_size, checkpoint_period)
    VIK_train_structure_search_RMSprop_minibatch(datapath, epochs, batch_size, checkpoint_period)
    CAVIKee_slot_train_structure_search(datapath, epochs, batch_size, checkpoint_period)
    CAVIKAUGee_slot_train_structure_search(datapath, epochs, batch_size, checkpoint_period)
    CAVIKee_sphere_train(datapath, epochs, batch_size, checkpoint_period)
