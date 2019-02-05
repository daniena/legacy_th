import json
from learning.plot import *
from learning.dataset import *
from learning.models import load_model_and_history
import numpy as np

def trainer(name, model, history, latest_checkpoint_num, checkpoint_path, training_input, training_output, validation_input, validation_output, test_input, test_output, *metaparameters):
    print('')
    model.summary()
    print('')

    training_input = np.array(training_input)
    training_output = np.array(training_output)
    validation_input = np.array(validation_input)
    validation_output = np.array(validation_output)
    test_input = np.array(test_input)
    test_output = np.array(test_output)

    (epochs, batch_size, checkpoint_period) = metaparameters

    if checkpoint_period is None:
        checkpoint_period = epochs
    if not checkpoint_path.endswith('/'):
        checkpoint_path = checkpoint_path + '/'
    
    num_checkpoints = int(epochs/checkpoint_period)
    if num_checkpoints < 1:
        num_checkpoints = 1
        checkpoint_period = epochs

    epochs_so_far = 0
    for checkpoint_index in range(latest_checkpoint_num+1, num_checkpoints + latest_checkpoint_num+1):
        print(training_input)

        epochs_this_step = checkpoint_period
        if epochs_so_far + checkpoint_period > epochs:
            epochs_this_step = epochs - epochs_so_far

        latest_history = model.fit(training_input, training_output, epochs=epochs_this_step, batch_size=batch_size, validation_data=(validation_input, validation_output))
        latest_history = latest_history.history

        if history: # is not empty
            history['val_loss'] += latest_history['val_loss']
            history['val_acc'] += latest_history['val_acc']
            history['loss'] += latest_history['loss']
            history['acc'] += latest_history['acc']
        else:
            history = latest_history
        
        model.save(checkpoint_path + name + '_checkpoint_' + str(checkpoint_index) + '.h5')
        print('')
        print('Saving trained ' + name + ' checkpoint', checkpoint_index, 'epochs:', epochs)
        with open(checkpoint_path + name + '_checkpoint_' + str(checkpoint_index) + '.json', 'w') as hist_json:
            json.dump(history, hist_json)
    
    (test_loss, test_acc) = model.evaluate(test_input, test_output)
    
    return (model, history, test_loss, test_acc)

def train(modelname, model, datasetname, dataset, datapath, random, *metaparameters, load_dataset_from_file=True, continue_from_checkpoint=True, create_if_none_found=False):
    
    test_inputs = []
    test_outputs = []
    validation_inputs = []
    validation_outputs = []
    training_inputs = []
    training_outputs = []
    split_dataset = []
    history = {}

    latest_checkpoint_num = -1

    if not dataset:
        if load_dataset_from_file:
            try:
                print('Loading in dataset ' + datasetname + '...')
                (training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs) = load(datapath, datasetname)
                split_dataset = (training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs)
                print('Loaded.')
            except:
                print('Found no dataset ' + datasetname + '.json')
                if create_if_none_found:
                    load_dataset_from_file = False
                else:
                    print('Set to exit if loading fails.')
                    exit(1)
        if not load_dataset_from_file:
            #((inputs, outputs), filenames) = from_raw(datapath, IKBC_parser, select_random_proportion, *(0.1, random))
            ((inputs, outputs), filenames) = from_raw(datapath, IKBC_parser, select_all, *())

            (training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs) = split_random(random, inputs, outputs, 0.7, 0.15, 0.15)
            split_dataset = (training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs)
            save(datapath, datasetname, split_dataset)
            save(datapath, datasetname + '_based_on_filenames', filenames)
    else:
        split_dataset = dataset
    
    if continue_from_checkpoint:
        for filename in os.listdir(data_subpath.strip('/')):
            checkpoint_file = ""
            if not filename.endswith('.h5') or not filename.startswith(modelname + '_checkpoint_'):
                continue
            else:
                checkpoint_file = filename
            
            checkpoint_num = int(checkpoint_file.replace(modelname + '_checkpoint_','').replace('.h5',''))
            if checkpoint_num > latest_checkpoint_num:
                latest_checkpoint_num = checkpoint_num
        
        if latest_checkpoint_num == -1:
            print('Found no model ' + modelname + '.')
            if create_if_none_found:
                continue_from_checkpoint = False
            else:
                print('Set to exit if loading fails.')
                return
        else:
            model, history = load_model_and_history(datapath, modelname + "_checkpoint_" + str(latest_checkpoint_num))
            print('Continuing ' + modelname + ' from checkpoint ' + str(latest_checkpoint_num) +'.')
        
    print('Starting the training of ' + modelname +':')
    (model, history, test_loss, test_acc) = trainer(modelname, model, history, latest_checkpoint_num, datapath, *split_dataset, *metaparameters)

    print(modelname + ' completed training with test_acc: ' + str(test_acc))
    
    return (model, history, split_dataset, test_loss, test_acc)
