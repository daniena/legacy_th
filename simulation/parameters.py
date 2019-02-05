from .workspace import cartesian_range_limits, spherical_radius_limits, cartesian_cube
from robotics.ur5_param import configuration_limits, configuration_offsets

dim_range = (1, 1.01)
workspace_dim_range = cartesian_range_limits(*dim_range, *dim_range, *dim_range)
border_epsilon = 0.020001
obstacle_radius_range = spherical_radius_limits(0.04, 0.3)
n_waypoints = 2
min_adjacent_waypoint_distance = 0.6
too_close_to_manipulator_space = cartesian_cube(-0.3, 0.3, -0.3, 0.3, -1, 1)
timestep = 0.02
circle_of_acceptance = 0.02
max_obstacles = 5

# Two things worth to re-consider:
#   min_adjacent_waypoint_distance should be smaller for the algorithm to be more general
#   too_close_to_manipulator_space should be smaller for the algorithm to be more general (this should only cover the base link and the pillar it is standing on)

#   But making them smaller may make it take too much time to get diverse results (obstacles have a tendency to clump up)
