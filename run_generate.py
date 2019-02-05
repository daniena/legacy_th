from session import CAI_args, CAI_random
from simulation.simulation import exit_criteria_at_i_max_only, exit_criteria_at_end_waypoint_or_i_max
from learning.datagen import generate_forced_bias_data, pandas_episode_trajectory_initialize
from learning.dataset import *
from learning.training import *
from util.file_operations import *
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from util.file_operations import *

def generate_ranpos(datapath, numsamples, sample_checkpoint_period, i_max, seed=0, iterate_seed=False):
    
    # Generate data from random configurations each timestep, with one single desired end effector position for 200 such timesteps

    (_, _, _, _, rawdatapaths, _, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    generate_forced_bias_data(random, seed, rawdatapaths[0], numsamples, sample_checkpoint_period, i_max, 'random_position', 0, exit_criteria=exit_criteria_at_i_max_only)
    print('Generated ' + str(numsamples) + '*' + str(i_max) + ' random_position VIK data in ' + datapath + '.')

def generate_ca_trackpos(datapath, numsamples, sample_checkpoint_period, i_max, seed=1, iterate_seed=False):
    
    # Generate data from tracking desired end effector position, with obstacles and collision avoidance (assuming perfect position control)

    (_, _, _, _, rawdatapaths, _, _, _, _) = CAI_args(datapath)
    (random, randomstate, seed) = CAI_random(seed, iterate_seed=iterate_seed)
    
    generate_forced_bias_data(random, seed, rawdatapaths[1], numsamples, sample_checkpoint_period, i_max, 'perfect_position', 5, exit_criteria=exit_criteria_at_end_waypoint_or_i_max)
    print('Generated ' + str(numsamples) + ' position tracking VIK data in ' + datapath + '.' )

if __name__ == '__main__':

    datapath = os.getcwd() + '/data'

    numsamples = 40000
    sample_checkpoint_period = int(numsamples/2)
    i_max = 400
    generate_ranpos(datapath, numsamples, sample_checkpoint_period, i_max)
    
    numsamples = 30000
    sample_checkpoint_period = int(numsamples/2)
    i_max = 1500
    generate_ca_trackpos(datapath, numsamples, sample_checkpoint_period, i_max)
