from collections import namedtuple
from util.matrix import *
from util.rotation import *
from robotics.kinematics import *
from robotics.configuration import *
from param_debug import debug
from math import *
import numpy as np
#from sys import exit


cartesian_range_limits = namedtuple('cartesian_range_limits', ['x_range_min', 'x_range_max', 'y_range_min', 'y_range_max', 'z_range_min', 'z_range_max'])
cartesian_cube = namedtuple('cartesian_cube', ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])
spherical_radius_limits = namedtuple('radius_limits', ['r_min', 'r_max'])
cartesian_sphere = namedtuple('cartesian_sphere', ['center_x', 'center_y', 'center_z', 'radius'])

from .parameters import *

def within_cartesian_cube(cartesian_manifold, position, epsilon):
    x = position[0]
    y = position[1]
    z = position[2]

    if cartesian_manifold.x_min + epsilon < x and x < cartesian_manifold.x_max - epsilon:
        if cartesian_manifold.y_min + epsilon < y and y < cartesian_manifold.y_max - epsilon:
            if cartesian_manifold.z_min + epsilon < z and z < cartesian_manifold.z_max - epsilon:
                return True
    return False

def within_cartesian_sphere(spherical_manifold, position, epsilon):
    x = position[0]
    y = position[1]
    z = position[2]

    distance_from_center =  norm(position - vector([spherical_manifold.center_x, spherical_manifold.center_y, spherical_manifold.center_z]))
    
    if distance_from_center < spherical_manifold.radius + epsilon:
        return True
    else:
        return False

def within_cartesian_space(cartesian_space, position, epsilon):

    if type(cartesian_space).__name__ is 'cartesian_sphere':
        return within_cartesian_sphere(cartesian_space, position, epsilon)

    elif type(cartesian_space).__name__ is 'cartesian_cube':
        return within_cartesian_cube(cartesian_space, position, epsilon)

    else:
        print('Error: Unsupported cartesian_space in within_cartesian_space with cartesian_space:', cartesian_space)
        exit(1)

def point_on_cylinder_around_line(random, pointA, pointB, distance):
    difference_vector = pointB - pointA

    if norm(difference_vector) < 0.0001:
        print("Failure, the points are the same")
        return "failure"

    point_on_line = pointA + difference_vector*random.uniform(0,1)

    orthogonal_vector_to_line = vector([0, 0, 0])
    while norm(orthogonal_vector_to_line) < 0.00001:
        orthogonal_vector_to_line = vector(np.cross(np.transpose(difference_vector), np.transpose(vector([random.uniform(1,2), random.uniform(1,2), random.uniform(1,2)])))[0])
        orthogonal_vector_to_line = orthogonal_vector_to_line/norm(orthogonal_vector_to_line)

    theta = random.uniform(-pi, pi)
    k = difference_vector/norm(difference_vector)
    R = axis_angle_rotation_matrix(k, theta)
    randomly_rotated_orthogonal_vector_to_line = np.dot(R, orthogonal_vector_to_line)
    
    if debug:
        print('point_on_line')
        print(point_on_line)
        print('R')
        print(R)
        print('orthogonal_vector_to_line')
        print(orthogonal_vector_to_line)
        print('randomly_rotated_orthogonal_vector_to_line')
        print(randomly_rotated_orthogonal_vector_to_line)
    
    point_near_line = point_on_line + distance*randomly_rotated_orthogonal_vector_to_line
    return point_near_line
# Uses parameters from from .parameters import *
def point_on_capsule_around_line(random, pointA, pointB, distance):
    difference_vector = pointB - pointA

    if norm(difference_vector) < 0.0001:
        print("Failure, the points are the same")
        return "failure"

    point_on_line = pointA + difference_vector*random.uniform(0,1)

    random_vector = vector(np.random.uniform(-1, 1, 3))
    random_unit_vector = random_vector/norm(random_vector)
    
    point_near_line = point_on_line + random_unit_vector*distance
    
    return point_near_line

        
# Uses parameters from from .parameters import *
def generate_random_workspace(random):

    x_min = -random.uniform(workspace_dim_range.x_range_min, workspace_dim_range.x_range_max)
    y_min = -random.uniform(workspace_dim_range.y_range_min, workspace_dim_range.y_range_max)
    z_min = -random.uniform(workspace_dim_range.z_range_min, workspace_dim_range.z_range_max)

    x_max = random.uniform(workspace_dim_range.x_range_min, workspace_dim_range.x_range_max)
    y_max = random.uniform(workspace_dim_range.y_range_min, workspace_dim_range.y_range_max)
    z_max = random.uniform(workspace_dim_range.z_range_min, workspace_dim_range.z_range_max)

    workspace_dimensions = cartesian_cube(x_min, x_max, y_min, y_max, z_min, z_max)
    return workspace_dimensions

# Uses parameters from from .parameters import *
def populate_random_obstructions_within_workspace(random, workspace, illegal_spaces, n_obstacles):

    if n_obstacles == 0:
        return []

    cartesian_obstacles = [0 for _ in range(n_obstacles)]
    
    for obst_i in range(n_obstacles):
        radius = random.uniform(obstacle_radius_range.r_min, obstacle_radius_range.r_max)
        illegal = True
        while illegal:
            illegal = False
            x = random.uniform(workspace.x_min + border_epsilon, workspace.x_max - border_epsilon)
            y = random.uniform(workspace.y_min + border_epsilon, workspace.y_max - border_epsilon)
            z = random.uniform(workspace.z_min + border_epsilon, workspace.z_max - border_epsilon)
            center = vector([x, y, z])
            
            for illegal_space in illegal_spaces:
                if within_cartesian_space(illegal_space, center, border_epsilon):
                    illegal = True
                    break
        cartesian_obstacles[obst_i] = cartesian_sphere(center[0,0], center[1,0], center[2,0], radius)

    return cartesian_obstacles

# Uses parameters from from .parameters import *
def populate_random_obstructions_within_workspace_biased_near_trajectory(random, workspace, waypoints, illegal_spaces, n_obstacles, n_obst_on_line, n_obst_near_line):
    # Important difference with the unbiased one is that this one allows obstacles to be put on init ee position, and on the space of the manipulator
    
    if n_obstacles == 0:
        return []

    cartesian_obstacles = [0 for _ in range(n_obstacles)]
    
    for obst_i in range(n_obstacles):
        radius = random.uniform(obstacle_radius_range.r_min, obstacle_radius_range.r_max)
        illegal = True

        random_outcome = random.uniform(0,1)
        disallow_starting_on_initial_waypoint = random_outcome > 0.05
        
        while illegal:
            illegal = False

            if obst_i < n_obst_on_line:
                distance = random.uniform(0, radius*0.5)
                center = point_on_cylinder_around_line(random, waypoints[0], waypoints[1], distance) + vector(np.random.normal(0, radius*0.005, 3)) # add some random noise
            elif obst_i < n_obst_on_line + n_obst_near_line:
                distance = random.uniform(radius*0.5, radius*2)
                center = point_on_capsule_around_line(random, waypoints[0], waypoints[1], distance)  + vector(np.random.normal(0, radius*0.05, 3)) # add some larger random noise
            else:
                x = random.uniform(workspace.x_min + border_epsilon, workspace.x_max - border_epsilon)
                y = random.uniform(workspace.y_min + border_epsilon, workspace.y_max - border_epsilon)
                z = random.uniform(workspace.z_min + border_epsilon, workspace.z_max - border_epsilon)
                center = vector([x, y, z])

            obstacle = cartesian_sphere(center[0,0], center[1,0], center[2,0], radius)
            
            if within_cartesian_sphere(obstacle, waypoints[1], border_epsilon): # Warning, allows init position to start inside an obstacle! Turn this off for multi-waypoints in the future.
                    illegal = True
            if disallow_starting_on_initial_waypoint:
               if within_cartesian_sphere(obstacle, waypoints[0], border_epsilon):
                   illegal = True
            for illegal_space in illegal_spaces:
                if within_cartesian_space(illegal_space, center, border_epsilon):
                    illegal = True
                    break
            
        cartesian_obstacles[obst_i] = cartesian_sphere(center[0,0], center[1,0], center[2,0], radius)

    return cartesian_obstacles

# Uses parameters from from .parameters import *
def populate_random_waypoints_within_legal_workspace(random, workspace, illegal_spaces):

    waypoint_configurations = [ vector([0, 0, 0, 0, 0, 0]) for _ in range(n_waypoints) ]
    waypoint_cartesian_positions = [ vector([0, 0, 0]) for _ in range(n_waypoints) ]

    for i in range(n_waypoints):
        q1 = 0
        q2 = 0
        q3 = 0
        q4 = 0
        q5 = 0
        q6 = 0
        x = 0
        y = 0
        z = 0
        waypoint_configuration = vector([0, 0, 0, 0, 0, 0])
        waypoint_cartesian_position = vector([0, 0, 0])
        
        illegal = True
        while illegal:
            illegal = False
            
            waypoint_configuration = vector(generate_random_configuration_limited(random))
            waypoint_cartesian_position = vector(forward_kinematic_position(waypoint_configuration))

            if not within_cartesian_cube(workspace, waypoint_cartesian_position, border_epsilon):
                illegal = True
                continue

            if i > 0:
                prev_waypoint_cartesian_position = waypoint_cartesian_positions[i-1]
                if norm(waypoint_cartesian_position - prev_waypoint_cartesian_position) < min_adjacent_waypoint_distance:
                    # Is too close to previous waypoint
                    illegal = True
                    continue
            
            for illegal_space in illegal_spaces:
                if within_cartesian_space(illegal_space, waypoint_cartesian_position, border_epsilon):
                    illegal = True
                    break
        
        waypoint_configurations[i] = waypoint_configuration
        waypoint_cartesian_positions[i] = waypoint_cartesian_position

    return (waypoint_configurations, waypoint_cartesian_positions)

# Uses parameters from from .parameters import *
def generate_random_populated_workspace(random, n_obstacles, n_waypoints):
    
    workspace = generate_random_workspace(random)
    cartesian_obstacles = populate_random_obstructions_within_workspace(random, workspace, too_close_to_manipulator_space, n_obstacles)

    illegal_spaces = cartesian_obstacles + [too_close_to_manipulator_space]
    
    waypoint_configurations, waypoint_cartesian_positions = populate_random_waypoints_within_legal_workspace(random,
                                                                                                             workspace,
                                                                                                             illegal_spaces)

    return (workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions)

# Uses parameters from from .parameters import *
def generate_random_populated_workspace_obstacles_biased_near_trajectory(random, n_obstacles, n_obst_on_line, n_obst_near_line):
    
    workspace = generate_random_workspace(random)

    illegal_spaces_for_waypoints = [too_close_to_manipulator_space]
    
    waypoint_configurations, waypoint_cartesian_positions = populate_random_waypoints_within_legal_workspace(random, workspace, illegal_spaces_for_waypoints)

    illegal_spaces_for_obstacles = []

    cartesian_obstacles = populate_random_obstructions_within_workspace_biased_near_trajectory(random,
                                                                                               workspace,
                                                                                               waypoint_cartesian_positions,
                                                                                               illegal_spaces_for_obstacles,
                                                                                               n_obstacles, n_obst_on_line, n_obst_near_line)

    return (workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions)
