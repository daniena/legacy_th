
from simulation.simulation import generate_and_simulate, generate_and_simulate_forced_bias_pattern_near_trajectory, exit_criteria_at_end_waypoint_or_i_max
from learning.datagen import pandas_episode_trajectory_initialize
from learning.infer import *
from keras.models import load_model
from session import CAI_random
import os

if __name__ == '__main__':

    seed = 0
    (random, randomstate, seed) = CAI_random(seed)

    max_timesteps = 1500 #16000 for position kp="3"
    max_obstacles = 5
    num_episodes = 10
    history = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
    for _ in range(num_episodes):
        #generate_and_simulate(random, history, max_timesteps, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, max_obstacles=max_obstacles, actuators='perfect_position', record=False, inference_model=None)
        generate_and_simulate_forced_bias_pattern_near_trajectory(random, history, max_timesteps, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, max_obstacles=max_obstacles, actuators='perfect_position', record=False, inference_model=None)
