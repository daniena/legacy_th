from session import CAI_args
from learning.datagen import pandas_episode_trajectory_initialize
from learning.infer import *
from keras.models import load_model
import os

def make_telemetry():
    datapath = os.getcwd() + '/data'
    
    seed = 100 # sphere actually manages to avoid the 7th, where the solution algorithm unfortunately goes singular
    seed = 10 # avoids the second one by dumb luck
    seed = 23 # for slot well connected, beautifully shows how it gets real close to the desired position and then diverges, and eventually comes back

    #seed = 108 # extra

    modelnames = ['VIK_pyramid',#'CAVIKAUGee_sphere_correct_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',#'CAVIKAUGee_sphere_correct_activation12_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected', #'VIK_pyramid',
                  'CAVIKAUGee_slot_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected',
                  'CAVIKAUGee_no_obst_input_control_experiment_ReLU_SGDcustom_clipping_no_batchnormalization_simulated_annealing_greater_step_bigger_model_well_connected']
    modelinitials = ['CAVIKAUGee_sphere', 'CAVIKAUGee_slot', 'CAVIKAUGee_no_obst']
    
    
    (_, _, _, _, _, _, checkpointpath, modelspath, _) = CAI_args(datapath)
    
    # Each observe has to be run separately unfortunately. It has not been identified why setting seeds do not work here, and at the same time the test module when generating data, constructing datasets, and training, two different times, yields identical data both times.
    
    seed = 201 # 101
    max_timesteps = 30000
    max_obstacles = 5
    num_episodes = 10
    
    #observe(seed, datapath, checkpointpath, modelnames, modelinitials, num_episodes, max_timesteps, max_obstacles, from_episode_num=None, from_latest_checkpoint=False, from_checkpoint_num=25, actuators='perfect_position', verbose=False)

    seed = 200
    max_timesteps = 5000
    max_obstacles = 5
    num_episodes = 5

    observe(seed, datapath, checkpointpath, modelnames, modelinitials, num_episodes, max_timesteps, max_obstacles, from_episode_num=None, from_latest_checkpoint=True, from_checkpoint_num=None, actuators='perfect_position', verbose=False) #VIK

    seed = 101
    max_timesteps = 1000
    max_obstacles = 5
    num_episodes = 30

    #observe(seed, datapath, checkpointpath, modelnames, modelinitials, num_episodes, max_timesteps, max_obstacles, from_episode_num=None, from_latest_checkpoint=False, from_checkpoint_num=25, actuators='perfect_position', verbose=False)

    seed = 1#02 #9001
    max_timesteps = 1000
    max_obstacles = 5
    num_episodes = 30

    #observe(seed, datapath, checkpointpath, modelnames, modelinitials, num_episodes, max_timesteps, max_obstacles, from_episode_num=3, from_latest_checkpoint=False, from_checkpoint_num=25, actuators='perfect_position', verbose=False)

if __name__ == '__main__':
    make_telemetry()
    
