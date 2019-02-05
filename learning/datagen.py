import pandas as pd
import numpy as np
from simulation.workspace import *
from simulation.simulation import *
from util.file_operations import make_path
import os

# Some tutorials and tips on pandas:
# https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python
# https://stackoverflow.com/questions/18674064/how-do-i-insert-a-column-at-a-specific-column-index-in-pandas
# https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index
# https://pandas.pydata.org/pandas-docs/stable/advanced.html

def pandas_episode_summary(filename, seed, q_waypoints, p_waypoints, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result):

    element_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z', 'radius', 'steps_actively_avoided', 'highest_percentage_to_center_penetrated', 'extra']

    n_waypoints = len(q_waypoints)
    n_obstacles = len(cartesian_obstacles)
    
    datanames_waypoints = ["waypoint_" + str(i) for i in range(0, n_waypoints)]
    datanames_obstacles = ["obstacle_" + str(i) for i in range(0, n_obstacles)]
    datanames_extra = ["singular_joints", "actuator_type", "highest_num_obstacles_avoided_at_same_time", "obstacles_avoided", "num_timesteps", "end_result", "seed"]
    
    index_2 = datanames_waypoints + datanames_obstacles + datanames_extra

    episodename = [filename for dataname in index_2]
    
    index_1 = episodename
    
    episodeframe = pd.DataFrame(index=[index_1, index_2], columns=element_names)
    episodeframe.index.names = ["filename", "datatype"]

    for i, q_w in enumerate(q_waypoints):
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q1'] = np.asscalar(q_w[0])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q2'] = np.asscalar(q_w[1])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q3'] = np.asscalar(q_w[2])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q4'] = np.asscalar(q_w[3])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q5'] = np.asscalar(q_w[4])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q6'] = np.asscalar(q_w[5])
    for i, p_w in enumerate(p_waypoints):
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'x'] = np.asscalar(p_w[0])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'y'] = np.asscalar(p_w[1])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'z'] = np.asscalar(p_w[2])
    
    for i, obstacle in enumerate(cartesian_obstacles):
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'x'] = obstacle.center_x
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'y'] = obstacle.center_y
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'z'] = obstacle.center_z
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'radius'] = obstacle.radius
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'steps_actively_avoided'] = num_timesteps_obstacle_actively_avoided[i]
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'highest_percentage_to_center_penetrated'] = obstacle_highest_percentage_to_center_penetrated[i]

    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q1'] = union_set_of_singular_joints[0]
    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q2'] = union_set_of_singular_joints[1]
    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q3'] = union_set_of_singular_joints[2]
    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q4'] = union_set_of_singular_joints[3]
    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q5'] = union_set_of_singular_joints[4]
    episodeframe.loc[(filename, "singular_joints" + str(i)), 'q6'] = union_set_of_singular_joints[5]

    highest_num_obstacles_avoided_at_same_time = 0
    for activation_mask in set_of_actively_avoided_obstacles_at_same_timestep:
        if sum(activation_mask) > highest_num_obstacles_avoided_at_same_time:
            highest_num_obstacles_avoided_at_same_time = sum(activation_mask)
        
    episodeframe.loc[(filename, "actuator_type"), "extra"] = actuators
    episodeframe.loc[(filename, "highest_num_obstacles_avoided_at_same_time"), "extra"] = highest_num_obstacles_avoided_at_same_time
    episodeframe.loc[(filename, "num_timesteps"), "extra"] = num_timesteps
    episodeframe.loc[(filename, "end_result"), "extra"] = end_result
    episodeframe.loc[(filename, "seed"), "extra"] = seed

    return episodeframe

def pandas_forced_bias_episode_summary(filename, seed, q_waypoints, p_waypoints, cartesian_obstacles, n_obst_on_line, n_obst_near_line, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result):

    n_obst_free = len(cartesian_obstacles) - n_obst_on_line - n_obst_near_line
    assert n_obst_free >= 0

    element_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z', 'radius', 'steps_actively_avoided', 'highest_percentage_to_center_penetrated', 'extra']

    n_waypoints = len(q_waypoints)
    n_obstacles = len(cartesian_obstacles)
    
    datanames_waypoints = ["waypoint_" + str(i) for i in range(0, n_waypoints)]
    datanames_obstacles = ["obstacle_" + str(i) for i in range(0, n_obstacles)]
    datanames_extra = ["singular_joints", "actuator_type", "n_obst_free", "n_obst_on_line", "n_obst_near_line", "highest_num_obstacles_avoided_at_same_time", "num_timesteps", "end_result", "seed"]
    
    index_2 = datanames_waypoints + datanames_obstacles + datanames_extra

    episodename = [filename for dataname in index_2]
    
    index_1 = episodename
    
    episodeframe = pd.DataFrame(index=[index_1, index_2], columns=element_names)
    episodeframe.index.names = ["filename", "datatype"]

    for i, q_w in enumerate(q_waypoints):
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q1'] = np.asscalar(q_w[0])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q2'] = np.asscalar(q_w[1])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q3'] = np.asscalar(q_w[2])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q4'] = np.asscalar(q_w[3])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q5'] = np.asscalar(q_w[4])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'q6'] = np.asscalar(q_w[5])
    for i, p_w in enumerate(p_waypoints):
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'x'] = np.asscalar(p_w[0])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'y'] = np.asscalar(p_w[1])
        episodeframe.loc[(filename, "waypoint_" + str(i)), 'z'] = np.asscalar(p_w[2])
    
    for i, obstacle in enumerate(cartesian_obstacles):
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'x'] = obstacle.center_x
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'y'] = obstacle.center_y
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'z'] = obstacle.center_z
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'radius'] = obstacle.radius
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'steps_actively_avoided'] = num_timesteps_obstacle_actively_avoided[i]
        episodeframe.loc[(filename, "obstacle_" + str(i)), 'highest_percentage_to_center_penetrated'] = obstacle_highest_percentage_to_center_penetrated[i]

    episodeframe.loc[(filename, "singular_joints"), 'q1'] = union_set_of_singular_joints[0]
    episodeframe.loc[(filename, "singular_joints"), 'q2'] = union_set_of_singular_joints[1]
    episodeframe.loc[(filename, "singular_joints"), 'q3'] = union_set_of_singular_joints[2]
    episodeframe.loc[(filename, "singular_joints"), 'q4'] = union_set_of_singular_joints[3]
    episodeframe.loc[(filename, "singular_joints"), 'q5'] = union_set_of_singular_joints[4]
    episodeframe.loc[(filename, "singular_joints"), 'q6'] = union_set_of_singular_joints[5]
        
    highest_num_obstacles_avoided_at_same_time = 0
    for activation_mask in set_of_actively_avoided_obstacles_at_same_timestep:
        if sum(activation_mask) > highest_num_obstacles_avoided_at_same_time:
            highest_num_obstacles_avoided_at_same_time = sum(activation_mask)
            
    episodeframe.loc[(filename, "actuator_type"), "extra"] = actuators
    episodeframe.loc[(filename, "n_obst_free"), "extra"] = n_obst_free
    episodeframe.loc[(filename, "n_obst_on_line"), "extra"] = n_obst_on_line
    episodeframe.loc[(filename, "n_obst_near_line"), "extra"] = n_obst_near_line
    episodeframe.loc[(filename, "highest_num_obstacles_avoided_at_same_time"), "extra"] = highest_num_obstacles_avoided_at_same_time
    episodeframe.loc[(filename, "num_timesteps"), "extra"] = num_timesteps
    episodeframe.loc[(filename, "end_result"), "extra"] = end_result
    episodeframe.loc[(filename, "seed"), "extra"] = seed

    return episodeframe

def pandas_episode_trajectory_initialize(max_timesteps, max_obstacles):

    element_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'q_dot_ref1', 'q_dot_ref2', 'q_dot_ref3', 'q_dot_ref4', 'q_dot_ref5', 'q_dot_ref6'] + ['activation_mask' + str(i) for i in range(1,max_obstacles+1)] + ['q1_inv', 'q2_inv', 'q3_inv', 'q4_inv', 'q5_inv', 'q6_inv']

    return pd.DataFrame(index=range(max_timesteps), columns=element_names)

def pandas_episode_to_csv(rawdatapath, seed, filename, history, q_waypoints, p_waypoints, obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result, episodes_summaries, save_summaries):
    episodes_path = rawdatapath + 'episodes/'

    make_path(episodes_path)
    
    episode_summary = pandas_episode_summary(filename, seed, q_waypoints, p_waypoints, obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result)

    if episodes_summaries is None:
        try:
            episodes_summaries = pd.read_csv(rawdatapath + 'episodes_summaries.csv', index_col=[0,1])
        except:
            episodes_summaries = episode_summary
    episodes_summaries = episodes_summaries.append(episode_summary)
    
    if save_summaries:
        episodes_summaries.to_csv(rawdatapath + 'episodes_summaries.csv')
    
    history = history.dropna(axis='index', how='all')
    history.to_csv(episodes_path + filename)

    return episodes_summaries

def pandas_forced_bias_episode_to_csv(rawdatapath, seed, filename, history, q_waypoints, p_waypoints, obstacles, n_obst_on_line, n_obst_near_line, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result, episodes_summaries, save_summaries):
    episodes_path = rawdatapath + 'episodes/'

    make_path(episodes_path)
    
    episode_summary  = pandas_forced_bias_episode_summary(filename, seed, q_waypoints, p_waypoints, obstacles, n_obst_on_line, n_obst_near_line, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result)

    if episodes_summaries is None:
        try:
            episodes_summaries = pd.read_csv(rawdatapath + 'episodes_summaries.csv', index_col=[0,1])
            episodes_summaries = episodes_summaries.append(episode_summary)
        except:
            episodes_summaries = episode_summary
    else:
        episodes_summaries = episodes_summaries.append(episode_summary)
    
    if save_summaries:
        episodes_summaries.to_csv(rawdatapath + 'episodes_summaries.csv')
    
    history = history.dropna(axis='index', how='all')
    history.to_csv(episodes_path + filename)

    return episodes_summaries

def generate_data(random, seed, rawdatapath, n_episodes, save_summaries_episodes_period, max_timesteps, actuators, max_obstacles, exit_criteria=exit_criteria_at_end_waypoint_or_i_max):

    episodes_summaries = None
    if save_summaries_episodes_period < 0 or save_summaries_episodes_period >= n_episodes:
        save_summaries_episodes_period = n_episodes

    if not rawdatapath.endswith('/'):
        rawdatapath += '/'
 
    start_number = 0
    if len(os.listdir(rawdatapath + "episodes")) > 0:
           start_number = max([int(episode.strip("episode").strip(".csv")) for episode in os.listdir(rawdatapath + "episodes")]) + 1
    
    for episode_number in range(start_number, start_number + n_episodes):
        save_summaries = False
        if episode_number >= start_number + n_episodes - 1:
            save_summaries = True
        elif episode_number % save_summaries_episodes_period == 0 and episode_number > 0:
            save_summaries = True
        
        history = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
    
        (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result) = generate_and_simulate(random, history, max_timesteps, exit_criteria=exit_criteria, max_obstacles=max_obstacles, actuators=actuators, record=True)

        filename = "episode"+str(episode_number)+".csv"
        episodes_summaries = pandas_episode_to_csv(rawdatapath, seed, filename, history, waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result, episodes_summaries, save_summaries)

    return range(start_number, start_number + n_episodes)

def generate_biased_data(random, seed, rawdatapath, n_episodes, save_summaries_episodes_period, max_timesteps, actuators, max_obstacles, exit_criteria=exit_criteria_at_end_waypoint_or_i_max):

    episodes_summaries = None
    if save_summaries_episodes_period < 0 or save_summaries_episodes_period >= n_episodes:
        save_summaries_episodes_period = n_episodes

    if not rawdatapath.endswith('/'):
        rawdatapath += '/'

    make_path(rawdatapath + "episodes")
    start_number = 0
    if len(os.listdir(rawdatapath + "episodes")) > 0:
           start_number = max([int(episode.strip("episode").strip(".csv")) for episode in os.listdir(rawdatapath + "episodes")]) + 1
    
    for episode_number in range(start_number, start_number + n_episodes):
        save_summaries = False
        if episode_number >= start_number + n_episodes - 1:
            save_summaries = True
        elif episode_number % save_summaries_episodes_period == 0 and episode_number > 0:
            save_summaries = True
        
        history = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
    
        (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result) = generate_and_simulate_biased_near_trajectory(random, history, max_timesteps, exit_criteria=exit_criteria, max_obstacles=max_obstacles, actuators=actuators, record=True)

        filename = "episode"+str(episode_number)+".csv"
        episodes_summaries = pandas_episode_to_csv(rawdatapath, seed, filename, history, waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result, episodes_summaries, save_summaries)

    return range(start_number, start_number + n_episodes)

def generate_forced_bias_data(random, seed, rawdatapath, n_episodes, save_summaries_episodes_period, max_timesteps, actuators, max_obstacles, exit_criteria=exit_criteria_at_end_waypoint_or_i_max):

    episodes_summaries = None
    if save_summaries_episodes_period < 0 or save_summaries_episodes_period >= n_episodes:
        save_summaries_episodes_period = n_episodes

    if not rawdatapath.endswith('/'):
        rawdatapath += '/'

    make_path(rawdatapath + "episodes")
    start_number = 0
    if len(os.listdir(rawdatapath + "episodes")) > 0:
           start_number = max([int(episode.strip("episode").strip(".csv")) for episode in os.listdir(rawdatapath + "episodes")]) + 1

    period = 5
    for episode_number in range(start_number, start_number + n_episodes):
        print(episode_number)

        save_summaries = False
        if episode_number >= start_number + n_episodes - 1:
            save_summaries = True
        elif episode_number % save_summaries_episodes_period == 0 and episode_number > 0:
            save_summaries = True
        
        history = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
    
        n_obstacles = 0
        n_obst_on_line = 0
        n_obst_near_line = 0

        if max_obstacles == 0:
            pass
        elif episode_number % period == 0:
            n_obstacles = random.randint(0, max_obstacles)
        elif episode_number % period < 3 :
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles - n_obst_on_line)
        else:
            # Switch the bias to let n_obst_near_line also have a higher chance of rolling a higher amount of obstacles over n_obst_on_line, so n_obst_on_line has the same average as n_obst_near_line
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles - n_obst_near_line)
        
    
        (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result) = generate_and_simulate_forced_bias_near_trajectory(random, history, max_timesteps, exit_criteria, n_obstacles, n_obst_on_line, n_obst_near_line, actuators=actuators, record=True)

        filename = "episode"+str(episode_number)+".csv"
        episodes_summaries = pandas_forced_bias_episode_to_csv(rawdatapath, seed, filename, history, waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, n_obst_on_line, n_obst_near_line, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, actuators, num_timesteps, end_result, episodes_summaries, save_summaries)

    return range(start_number, start_number + n_episodes)
