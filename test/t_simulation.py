from robotics.task import *
from robotics.rtpwrapper import *
from simulation.workspace import *
from simulation.simulation import *
from random import Random
from util.matrix import *
from learning import datagen
from .test_utilities import *
from param_debug import debug
import numpy as np

def test_generate_random_workspace():

    print('Warning: test_generate_random_workspace currently not implemented correctly')
    return

    seed = 1003971741047759123774939937534
    rand = Random()
    rand.seed(seed)
    
    range_limits_cases = [cartesian_range_limits(1, 3, 1, 3, 1, 3),
                          cartesian_range_limits(0.5, 4, 0.5, 4, 0.5, 4),
                          cartesian_range_limits(0.2, 2, 0.2, 2, 0.2, 2),
                          cartesian_range_limits(0.3, 0.35, 0.3, 0.35, 0.3, 0.35)]

    for range_limits in range_limits_cases:
        workspace = generate_random_workspace(rand, range_limits)

        if debug:
            print('range_limits')
            print(range_limits)
            print('resultant workspace')
            print(workspace)

        epsilon = 0.0001
        assert workspace.x_min < -range_limits.x_range_min/2 + epsilon
        assert workspace.y_min < -range_limits.y_range_min/2 + epsilon
        assert workspace.z_min < -range_limits.z_range_min/2 + epsilon

        assert workspace.x_max > range_limits.x_range_min/2 - epsilon
        assert workspace.y_max > range_limits.y_range_min/2 - epsilon
        assert workspace.z_max > range_limits.z_range_min/2 - epsilon

        assert workspace.x_max - workspace.x_min < range_limits.x_range_max + epsilon
        assert workspace.y_max - workspace.y_min < range_limits.y_range_max + epsilon
        assert workspace.z_max - workspace.z_min < range_limits.z_range_max + epsilon

    print('Test: test_generate_random_workspace ASSERT TRUE: True')
    print('')

def test_populate_random_waypoints_within_legal_workspace():

    print('Warning: test_populate_random_waypoints_within_legal_workspace currently not implemented correctly')
    return
    
    seed = 7847781895006737326126164889204
    rand = Random()
    rand.seed(seed)

    max_obstacles = 5
    obstacle_radius_range = spherical_radius_limits(0.1, 0.3)

    n_points = 3
    
    range_limits_cases = [cartesian_range_limits(1, 3, 1, 3, 1, 3),
                          cartesian_range_limits(0.5, 4, 0.5, 4, 0.5, 4),
                          cartesian_range_limits(0.2, 2, 0.2, 2, 0.2, 2),
                          cartesian_range_limits(0.3, 0.35, 0.3, 0.35, 0.3, 0.35)]

    configuration_lower_limit = pi/10
    configuration_upper_limit = 2*pi - configuration_lower_limit
    
    for range_limits in range_limits_cases:
        workspace = generate_random_workspace(rand, range_limits)
        obstacles = populate_random_obstructions_within_workspace(rand, workspace, max_obstacles, obstacle_radius_range, 0.02)
        too_close_to_manipulator_box = cartesian_cube(-range_limits.x_range_min/2, range_limits.x_range_min/2, -range_limits.y_range_min/2, range_limits.y_range_min/2, -range_limits.z_range_min/2, range_limits.z_range_min/2)

        illegal_spaces = [0 for i in range(len(obstacles)+1)]
        illegal_spaces[0:len(obstacles)] = obstacles[:]
        illegal_spaces[len(obstacles)] = too_close_to_manipulator_box
        
        configuration_waypoints, position_waypoints = populate_random_waypoints_within_legal_workspace(rand, workspace, illegal_spaces, n_points, configuration_lower_limit, configuration_upper_limit, 0.05, 0.05)

        print('Warning: test_populate_random_waypoints_within_legal_workspace unfinished')
        
        if debug:
            print('workspace')
            print(workspace)
            print('illegal_spaces')
            print(illegal_spaces)
            print('waypoints')
            print(waypoints)

def test_point_on_cylinder_around_line():

    print('')

    seed = 88567178993995
    rand = Random()
    rand.seed(seed)

    pointAs = (vector([1, 0, 0]), vector([0, 1, 0]), vector([0, 0, -2]), vector([36.1, -144, 1098]))
    pointBs = (vector([2, 0, 0]), vector([0, 2, 0]), vector([0, 0, -1]), vector([35.9, -137, 819]))
    
    distances = (0.1, 0.1, 0.1, 10)

    i = 0
    for pointA, pointB in zip(pointAs, pointBs):
        dist = distances[i]
        i += 1
        
        p = point_on_cylinder_around_line(rand, pointA, pointB, dist)
        
        point_on_line = abs(np.dot(np.transpose((pointB-pointA)/norm(pointB-pointA)), (p-pointA)/norm(pointB-pointA))[0][0])*(pointB-pointA) + pointA
        
        if debug:
            print('')
            print('pointA')
            print(pointA)
            print('pointB')
            print(pointB)
            print('p')
            print(p)
            print("\np's projection onto pointB-pointA would reach this much of the way towards pointB from pointA:")
            print(abs(np.dot(np.transpose((pointB-pointA)/norm(pointB-pointA)), (p-pointA)/norm(pointB-pointA))[0][0]))
            print('\nSince pointB - pointA is:')
            print(pointB-pointA)
            print('Then point_on_line - pointA has to be:')
            print(point_on_line - pointA)
            print('And point_on_line is:')
            print(point_on_line)
            print('\nFollowing two should be orthogonal to eachother:')
            print('point_on_line - pointA')
            print(point_on_line - pointA)
            print('p - point_on_line')
            print(p - point_on_line)
            print('Their dot product should be less than 0.00001:')
            print(abs(np.dot(np.transpose(point_on_line - pointA), p - point_on_line)[0][0]))
            print('The length of p - point_on_line should be equal to the dist, which is ' + str(dist) + ':')
            print('And the actual length of p - point_on_line is norm(p-point_on_line):', norm(p-point_on_line))
            
            print('')
            
        assert abs(np.dot(np.transpose(point_on_line - pointA), p - point_on_line)[0][0]) < 0.00001
        assert dist - 0.00001 < norm(p-point_on_line) < 0.00001 + dist

    print('Test: test_point_on_cylinder_around_line ASSERT TRUE: True')
    print('')
    

def test_simulate_perfect_position_control():
    (ca_tasks, waypoints, rtpwrapped) = simple_world_init()

    workspace_dim = []
    
    status = simulate_perfect_position_control(workspace_dim, ca_tasks, waypoints, rtpwrapped, exit_criteria=exit_criteria_at_end_waypoint_only)

    assert status[0] is "Success"
    print('Test: test_simulate_perfect_position_control ASSERT TRUE: True')
    print('')

def test_simulate_position_control():
    (ca_tasks, waypoints, rtpwrapped) = simple_world_init()

    workspace_dim = []
    
    status = simulate_position_control(workspace_dim, ca_tasks, waypoints, rtpwrapped, exit_criteria=exit_criteria_at_end_waypoint_only)

    assert status[0] is "Success"
    print('Test: test_simulate_position_control ASSERT TRUE: True')
    print('')

def test_simulate_velocity_control():
    (ca_tasks, waypoints, rtpwrapped) = simple_world_init()

    workspace_dim = []
    
    status = simulate_velocity_control(workspace_dim, ca_tasks, waypoints, rtpwrapped, exit_criteria=exit_criteria_at_end_waypoint_only)

    assert status[0] is "Success"
    print('Test: test_simulate_velocity_control ASSERT TRUE: True')
    print('')

def test_simulate(actuators='position', ):
    (ca_tasks, q_init, waypoints, rtpwrapped) = simple_world_init()

    workspace_dim = []
    
    status = simulate(workspace_dim, ca_tasks, (q_init,), waypoints, rtpwrapped, actuators='perfect_position', exit_criteria=exit_criteria_at_end_waypoint_only)

    assert status[0] is "Success"
    print('Test: test_simulate with actuator type', actuators, 'ASSERT TRUE: True')
    print('')

def test_simulate_position_control_static_reference():
    (ca_tasks, waypoints, rtpwrapped) = simple_world_init()

    workspace_dim = []

    q_targets = [vector([1.20415734927135, -0.7054936073432778, 1.6966688430453394, 0.40525013417795686, 0.34272227435507563, 0.10040962840323489]),
                 vector([2.6147068670249105, -0.19480497429217308, 1.4709489674689873, 0.23863576660386782, 0.3523410542437169, 0.0998414434917649]),
                 vector([3.5709332106278917, -0.17869795524542215, 1.4622523706283457, 0.236202991893607, 0.3041916693220885, 0.099805024071921])]
    
    status = simulate_position_control_static_reference(workspace_dim, waypoints, q_targets, rtpwrapped, exit_criteria=exit_criteria_at_end_waypoint_only)

    assert status[0] is "Success"
    print('Test: test_simulate_position_control_static_reference ASSERT TRUE: True')
    print('')

def test_generate_and_simulate(n_episodes=5):
    seed = 84769626268499551048173622910385610472947546719
    rand = Random()
    rand.seed(seed)
    max_timesteps = 10000
    max_obstacles = 12
    
    history = datagen.pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
    for _ in range(n_episodes):
        print(generate_and_simulate(random, history, max_timesteps, exit_criteria_at_end_waypoint_or_i_max, max_obstacles=max_obstacles, actuators='position', record=False))
