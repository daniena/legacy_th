from simulation.workspace import *
from random import Random
from param_debug import debug

def test_generate_random_workspace():

    seed = 1003971741047759123774939937534
    rand = Random()
    rand.seed(seed)
    
    workspace_range_limits_cases = [cartesian_range_limits(1, 3, 1, 3, 1, 3),
                                    cartesian_range_limits(0.5, 4, 0.5, 4, 0.5, 4),
                                    cartesian_range_limits(0.2, 2, 0.2, 2, 0.2, 2),
                                    cartesian_range_limits(0.3, 0.35, 0.3, 0.35, 0.3, 0.35)]

    for range_limits in workspace_range_limits_cases:
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
    
    seed = 7847781895006737326126164889204
    rand = Random()
    rand.seed(seed)

    max_obstacles = 5
    obstacle_radius_range = spherical_radius_limits(0.1, 0.3)

    n_points = 3
    
    workspace_range_limits_cases = [cartesian_range_limits(1, 3, 1, 3, 1, 3),
                                    cartesian_range_limits(0.5, 4, 0.5, 4, 0.5, 4),
                                    cartesian_range_limits(0.2, 2, 0.2, 2, 0.2, 2),
                                    cartesian_range_limits(0.3, 0.35, 0.3, 0.35, 0.3, 0.35)]

    #configuration_
    
    for range_limits in workspace_range_limits_cases:
        workspace = generate_random_workspace(rand, range_limits)
        obstacles = populate_random_obstructions_within_workspace(rand, workspace, max_obstacles, obstacle_radius_range, 0.02)
        too_close_to_manipulator_box = cartesian_manifold_limits(-range_limits.x_range_min/2, range_limits.x_range_min/2, -range_limits.y_range_min/2, range_limits.y_range_min/2, -range_limits.z_range_min/2, range_limits.z_range_min/2)

        illegal_spaces = [0 for i in range(len(obstacles)+1)]
        illegal_spaces[0:len(obstacles)] = obstacles[:]
        illegal_spaces[len(obstacles)] = too_close_to_manipulator_box
        
        waypoints = populate_random_waypoints_within_legal_workspace(rand, workspace, illegal_spaces, n_points, 0.05, 0.05)

        if debug:
            print('workspace')
            print(workspace)
            print('illegal_spaces')
            print(illegal_spaces)
            print('waypoints')
            print(waypoints)

