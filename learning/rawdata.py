import pandas as pd
import numpy as np
from param_debug import debug
from simulation.workspace import dim_range, border_epsilon
from robotics.rtpwrapper import *
from robotics.kinematics import jacobian_linear_velocity, forward_kinematic_position
from util.matrix import vector, norm
from math import *
from session import max_obstacles

def broadcast_waypoints(summaries, filename, numsteps):
    return np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps, axis=0)

_slot_range = (-dim_range[1] + border_epsilon, dim_range[1] - border_epsilon)
def empty_obstacle_slot(random):
    
    x = random.uniform(*_slot_range)
    y = random.uniform(*_slot_range)
    z = random.uniform(*_slot_range)
    r = 0
    
    return [x, y, z, r]

def make_obstacles_buffer(obstacles, max_obstacles):
    obstacles_buffer = []
    if isinstance(obstacles, list) or isinstance(obstacles, np.ndarray): # Assumes obstacle_slot format constructed in CAVIKee_slots_IO_from_rawdata
        obstacles_buffer = [ obstacles[4*i:4*(i+1)] for i in range(max_obstacles) if obstacles[4*i + 3] > 0 ] # Include obstacle only if its radius is larger than zero
    elif isinstance(obstacles, pd.DataFrame):
        obstacles_buffer = [ [ obstacles.loc[obstacle_name]['x'],
                               obstacles.loc[obstacle_name]['y'],
                               obstacles.loc[obstacle_name]['z'],
                               obstacles.loc[obstacle_name]['radius'] ] for obstacle_name in obstacles.index ]
    return obstacles_buffer

def make_obstacle_slots(obstacles_buffer, max_obstacles):
    obstacle_slots = [ 0 for _ in range(max_obstacles*4) ]
    n_obst = len(obstacles_buffer)
            
    slot_positions = [ slot_i if slot_i < n_obst else -1 for slot_i in range(max_obstacles) ]
    random.shuffle(slot_positions)
            
    for slot_i in range(max_obstacles):
        if slot_positions[slot_i] < 0:
            obstacle_slots[slot_i*4:(slot_i+1)*4] = empty_obstacle_slot(random)[:]
        else:
            obstacle_slots[slot_i*4:(slot_i+1)*4] = obstacles_buffer[slot_positions[slot_i]][:]
    return obstacle_slots

def spherical_coordinates(v):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system

    x = v[0]
    y = v[1]
    z = v[2]
    
    r = sqrt(x**2+y**2+z**2)
    elevation = acos(z/r)
    azimuth = atan2(y,x)

    return (r, elevation, azimuth)

_min_distance_activation_threshold = 0.0000 # Activate ONLY when it is ON the border of an obstacle! (used 0.00001 epsilon earlier, bad idea because solution algorithm uses an extra room of 0.0 once circle_of_acceptance was added to it)
_circle_of_acceptance = 0.02 # ToDo: This should rather be imported from simulation.parameters
def obstacle_matrix_spherical_base_frame(obstacles_buffer, ee, resolution, min_distance_activation_threshold):

    spherical_pixel_matrix = np.zeros(resolution)
    
    elevation_pixel_size = pi/resolution[0]
    azimuth_pixel_size = 2*pi/resolution[1]
    
    for obstacle in obstacles_buffer:

        center = obstacle[0:3]
        radius = obstacle[3]
        if radius < 0.00001:
            continue
        diff_vector = vector(center) - vector(ee) # in the base-frame of the robot, i.e. each position in the matrix represents static directions for the ur5 because it does not move or rotate its base frame
        (distance_to_center, elevation, azimuth) = spherical_coordinates(diff_vector)

        #elevation += pi/2 # So it goes from 0 to pi instead of -pi/2 to pi/2
        #azimuth += pi # So it goes from 0 to 2*pi instead of -pi to pi
        
        #print(elevation)
        #print(azimuth)
        #print('')
        
        elevation_pixel = int(elevation/elevation_pixel_size)
        azimuth_pixel = int(azimuth/azimuth_pixel_size)

        distance_to_border = distance_to_center - (radius + _circle_of_acceptance)
        if distance_to_border < min_distance_activation_threshold:
            #element_activation = 1 - distance_to_border / min_distance_activation_threshold # 1 when on obstacle border, 0 to 1 when distance to border less than threshold, 0 otherwise
            #element_activation /= 10
            #if spherical_pixel_matrix[elevation_pixel, azimuth_pixel] < element_activation:
                #spherical_pixel_matrix[elevation_pixel, azimuth_pixel] = element_activation # Keep the largest activation
            spherical_pixel_matrix[elevation_pixel, azimuth_pixel] = 1#12

    return spherical_pixel_matrix


VIK_num_inputs = 6 + 3
def VIK_IO_from_rawdata(random, datapath, filenames, max_obstacles):

    if not datapath.endswith('/'):
        datapath = datapath + '/'

    summaries = pd.read_csv(datapath + 'episodes_summaries.csv', index_col=[0,1])

    inputs = ['q', 'ee_des']
    outputs = ['f1']

    numsteps_episode = [ int(summaries.at[(filename, 'num_timesteps'), 'extra']) for filename in filenames ]
    numsteps_total = sum(numsteps_episode)
    
    print('Number of steps:', numsteps_total)

    inputs = np.zeros((numsteps_total, VIK_num_inputs))
    outputs = np.zeros((numsteps_total, 6))

    laststep = 0
    for i, filename in enumerate(filenames):
        episode = pd.read_csv(datapath + 'episodes/' + filename, index_col=0).values
        if debug:
            print(filename), print(len(episode)), print(len(np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0))), print(summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values)
        inputs[laststep:laststep + numsteps_episode[i], :] = np.hstack((episode[:,0:6], broadcast_waypoints(summaries, filename, numsteps_episode[i])))
        outputs[laststep:laststep + numsteps_episode[i], :] = episode[:, 9:15]
        laststep = laststep + numsteps_episode[i]

    #return (np.array(inputs).tolist(), np.array(outputs).tolist())
    return (inputs, outputs)

# ee = end to end
CAVIKAUGee_slots_num_inputs = 6+3 + 4*max_obstacles
CAVIKAUGee_slots_num_outputs = 3+6+6
def CAVIKAUGee_slots_IO_from_rawdata(random, datapath, filenames, max_obstacles):

    if not datapath.endswith('/'):
        datapath = datapath + '/'

    summaries = pd.read_csv(datapath + 'episodes_summaries.csv', index_col=[0,1])


    inputs = ['q', 'ee_des', 'obst_slots']
    outputs = ['q_dot_ref']

    numsteps_episode = [ int(summaries.at[(filename, 'num_timesteps'), 'extra']) for filename in filenames ]
    numsteps_total = sum(numsteps_episode)
    
    print('Number of steps:', numsteps_total)

    inputs = np.zeros((numsteps_total, CAVIKAUGee_slots_num_inputs))
    outputs = np.zeros((numsteps_total, CAVIKAUGee_slots_num_outputs))

    laststep = 0
    for i, filename in enumerate(filenames):
        episode = pd.read_csv(datapath + 'episodes/' + filename, index_col=0).values
        episode_summary = summaries.xs(filename, level='filename')
        obstacles = episode_summary[episode_summary.index.str.contains('obstacle_')]

        obstacles_buffer = []
        n_obst = len(obstacles.index)
        if not obstacles.empty:
            make_obstacles_buffer(obstacles, max_obstacles)

        all_timesteps_obstacle_slots = np.zeros((numsteps_episode[i], max_obstacles*4))
        for timestep_i in range(numsteps_episode[i]):
            
            obstacle_slots = make_obstacle_slots(obstacles_buffer, max_obstacles)
            all_timesteps_obstacle_slots[timestep_i, :] = obstacle_slots
        if debug:
            print(filename), print(len(episode)), print(len(np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0))), print(summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values), print(all_timesteps_obstacle_slots)

        
        inputs[laststep:laststep + numsteps_episode[i], :] = np.hstack((episode[:,0:6], broadcast_waypoints(summaries, filename, numsteps_episode[i]), all_timesteps_obstacle_slots))
        outputs[laststep:laststep + numsteps_episode[i], :] = episode[:,6:21]
        laststep = laststep + numsteps_episode[i]

    #return (np.array(inputs).tolist(), np.array(outputs).tolist())
    return (inputs, outputs)

CAVIKee_slots_num_inputs = 6+3 + 4*max_obstacles
def CAVIKee_slots_IO_from_rawdata(random, datapath, filenames, max_obstacles):

    if not datapath.endswith('/'):
        datapath = datapath + '/'

    summaries = pd.read_csv(datapath + 'episodes_summaries.csv', index_col=[0,1])


    inputs = ['q', 'ee_des', 'obst_slots']
    outputs = ['q_dot_ref']

    numsteps_episode = [ int(summaries.at[(filename, 'num_timesteps'), 'extra']) for filename in filenames ]
    numsteps_total = sum(numsteps_episode)
    
    print('Number of steps:', numsteps_total)

    inputs = np.zeros((numsteps_total, CAVIKee_slots_num_inputs))
    outputs = np.zeros((numsteps_total, 6))

    laststep = 0
    for i, filename in enumerate(filenames):
        episode = pd.read_csv(datapath + 'episodes/' + filename, index_col=0).values
        episode_summary = summaries.xs(filename, level='filename')
        obstacles = episode_summary[episode_summary.index.str.contains('obstacle_')]

        obstacles_buffer = []
        n_obst = len(obstacles.index)
        if not obstacles.empty:
            make_obstacles_buffer(obstacles, max_obstacles)

        all_timesteps_obstacle_slots = np.zeros((numsteps_episode[i], max_obstacles*4))
        for timestep_i in range(numsteps_episode[i]):
            
            obstacle_slots = make_obstacle_slots(obstacles_buffer, max_obstacles)
            all_timesteps_obstacle_slots[timestep_i, :] = obstacle_slots

        if debug:
            print(filename), print(len(episode)), print(len(np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0))), print(summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values), print(all_timesteps_obstacle_slots)

        
        inputs[laststep:laststep + numsteps_episode[i], :] = np.hstack((episode[:,0:6], broadcast_waypoints(summaries, filename, numsteps_episode[i]), all_timesteps_obstacle_slots))
        outputs[laststep:laststep + numsteps_episode[i], :] = episode[:, 15:21]
        laststep = laststep + numsteps_episode[i]

    #return (np.array(inputs).tolist(), np.array(outputs).tolist())
    return (inputs, outputs)

CAVIKee_sphere_resolution = (9,18)
CAVIKee_sphere_num_inputs = 6+3 + CAVIKee_sphere_resolution[0]*CAVIKee_sphere_resolution[1]
def CAVIKee_sphere_IO_from_rawdata(random, datapath, filenames, max_obstacles):

    if not datapath.endswith('/'):
        datapath = datapath + '/'

    summaries = pd.read_csv(datapath + 'episodes_summaries.csv', index_col=[0,1])

    inputs = ['q', 'ee_des', 'obst_slots']
    outputs = ['q_dot_ref']

    numsteps_episode = [ int(summaries.at[(filename, 'num_timesteps'), 'extra']) for filename in filenames ]
    numsteps_total = sum(numsteps_episode)
    
    print('Number of steps:', numsteps_total)

    inputs = np.zeros((numsteps_total, CAVIKee_sphere_num_inputs))
    outputs = np.zeros((numsteps_total, 6))

    laststep = 0
    for i, filename in enumerate(filenames):
        episode = pd.read_csv(datapath + 'episodes/' + filename, index_col=0).values
        episode_summary = summaries.xs(filename, level='filename')
        obstacles = episode_summary[episode_summary.index.str.contains('obstacle_')]

        n_obst = len(obstacles.index)
        obstacles_buffer = []
        if not obstacles.empty:
            obstacles_buffer = make_obstacles_buffer(obstacles, max_obstacles)

        all_timesteps_obstacle_pixel_array = np.zeros((numsteps_episode[i], CAVIKee_sphere_resolution[0]*CAVIKee_sphere_resolution[1]))
        for timestep_i in range(numsteps_episode[i]):
            ee = episode[timestep_i,6:9]
            obstacle_pixel_matrix = obstacle_matrix_spherical_base_frame(obstacles_buffer, ee, CAVIKee_sphere_resolution, _min_distance_activation_threshold)
            obstacle_pixel_array = [ pixel for row in obstacle_pixel_matrix for pixel in row ]
            all_timesteps_obstacle_pixel_array[timestep_i, :] = obstacle_pixel_array

        if debug:
            print(filename), print(len(episode)), print(len(np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0))), print(summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values), print(all_timesteps_obstacle_pixel_array)

        
        inputs[laststep:laststep + numsteps_episode[i], :] = np.hstack((episode[:,0:6], broadcast_waypoints(summaries, filename, numsteps_episode[i]), all_timesteps_obstacle_pixel_array))
        outputs[laststep:laststep + numsteps_episode[i], :] = episode[:, 15:21]
        laststep = laststep + numsteps_episode[i]

    #return (np.array(inputs).tolist(), np.array(outputs).tolist())
    return (inputs, outputs) # Goes out of memory

def CAVIKee_sphere_input_from_CAVIKee_slots_IO(slotted_input, max_obstacles, mean, std):
    obstacles_buffer = make_obstacles_buffer(slotted_input[9:], max_obstacles)

    ee = forward_kinematic_position(vector(slotted_input[0:6]))
    
    obstacle_pixel_matrix =  obstacle_matrix_spherical_base_frame(obstacles_buffer, ee, CAVIKee_sphere_resolution, _min_distance_activation_threshold)
    obstacle_pixel_array = [ pixel for row in obstacle_pixel_matrix for pixel in row ]
    
    sphereified_input = np.hstack(((slotted_input[0:9]-mean)/std, obstacle_pixel_array))

    return sphereified_input

CAVIKAUGee_sphere_resolution = (9,18)
CAVIKAUGee_sphere_num_inputs = 6+3 + CAVIKAUGee_sphere_resolution[0]*CAVIKAUGee_sphere_resolution[1]
CAVIKAUGee_sphere_num_outputs = 3+6+6 #15
def CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO(slotted_input, max_obstacles, mean, std):
    obstacles_buffer = make_obstacles_buffer(slotted_input[9:], max_obstacles)

    ee = forward_kinematic_position(vector(slotted_input[0:6]))
    
    obstacle_pixel_matrix =  obstacle_matrix_spherical_base_frame(obstacles_buffer, ee, CAVIKAUGee_sphere_resolution, _min_distance_activation_threshold)
    obstacle_pixel_array = [ pixel for row in obstacle_pixel_matrix for pixel in row ]
    
    sphereified_input = np.hstack(((slotted_input[0:9]-mean)/std, obstacle_pixel_array))

    return sphereified_input

CAVIKAUGee_no_obst_control_num_inputs = 6+3
CAVIKAUGee_no_obst_control_num_outputs = 3+6+6 #15
def CAVIKAUGee_no_obst_control_input_from_CAVIKee_slots_IO(slotted_input, max_obstacles, mean, std):
    return (slotted_input[0:9]-mean)/std
