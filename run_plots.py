import session
from session import CAI_random, CAI_args
from learning.plot import *
from util.file_operations import *
import copy
import os

def training_history(modelname):
    random, _, _ = CAI_random(0)
    _, history = load_model_and_history_from_latest_checkpoint(random, os.getcwd() + '/data/sessions/CAI/checkpoints', modelname)
    return history

def validation_history(historyname):
    return load_json(os.getcwd() + '/data/temp/', historyname)

def plot_histories_of_models(modelnames, plotname):
    num_ticks = 12
    path = os.getcwd() + '/data/temp'
    training_histories = [ training_history(modelname) for modelname in modelnames ]
    
    legends = ['sphere', 'slot', 'control']

    def foo(denom, ylabel, num_ticks, legends, path, plotfilename):
        train_histories = [ training_history[denom] for training_history in training_histories ]
        val_histories = [ training_history['val_' + denom] for training_history in training_histories ]
        plot_training_scores(train_histories, val_histories, ylabel, num_ticks, legends, show=False, save=True, plotpath=path, plotfilename=plotfilename)

    foo('acc', 'Accuracy', num_ticks, legends, path, plotname + '_acc')
    foo('loss', 'Loss', num_ticks, legends, path, plotname + '_loss')

def plot_histories_of_models_valfix(modelnames, plotname):
    num_ticks = 12
    path = os.getcwd() + '/data/temp'
    training_histories = [ training_history(modelname) for modelname in modelnames ]
    validation_histories = [validation_history('sphere_val_history'), validation_history('slot_val_history_updated'), validation_history('control_val_history_updated')]

    #for index, val_histories in enumerate(validation_histories):
    #    temp = copy.deepcopy(val_histories['acc'])
    #    validation_histories[index]['acc'] = copy.deepcopy(val_histories['loss'])
    #    validation_histories[index]['loss'] = temp
    
    legends = ['sphere', 'slot', 'control']

    def foo(denom, ylabel, num_ticks, legends, path, plotfilename):
        train_histories = [ training_history[denom] for training_history in training_histories ]
        val_histories = [val_history[denom] for val_history in validation_histories]
        for l in val_histories:
            print(len(l))
        plot_training_scores(train_histories, val_histories, ylabel, num_ticks, legends, show=False, save=True, plotpath=path, plotfilename=plotfilename)

    foo('acc', 'Accuracy', num_ticks, legends, path, plotname + '_acc')
    foo('loss', 'Loss', num_ticks, legends, path, plotname + '_loss')


def make_plots():
    seed = 100

    datapath = os.getcwd() + '/data'
    (random, randomstate, seed) = CAI_random(seed)
    (_, _, _, _, _, _, checkpointpath, modelspath, _) = CAI_args(datapath)

    plotpath = datapath + '/sessions/CAI/plots/training_history_plots'
    plot_all_most_recent_checkpoints(random, checkpointpath, performance_threshold_to_show=0.0, containing=None, save=True, plotpath=plotpath)

    modelnames = ['CAVIKAUGee_sphere_correct_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',#'CAVIKAUGee_sphere_correct_activation12_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',
                  'CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',
                  'CAVIKAUGee_no_obst_input_control_experiment_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']
    plot_histories_of_models_valfix(modelnames, 'training_history_comparison')
    plot_histories_of_models(('VIK_pyramid',), 'training_history_trials')
    
if __name__ == '__main__':
    make_plots()
