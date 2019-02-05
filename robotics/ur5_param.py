
#DH convention parameters
from math import pi

d6_extension = 0.095 # Extension to put the ee_position on the visible ball geom at the end of the end effector

d1 =  0.0892
a2 = -0.425
a3 = -0.392
d4 =  0.1093
d5 =  0.09475
d6 =  0.0825 + d6_extension

alpha1 =  pi/2
alpha4 =  pi/2
alpha5 = -pi/2


inner_lim = pi/3
outer_lim = pi-pi/3
configuration_limits = [(-pi, pi), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim)]
# configuration_limits_positive = [(-pi, pi), (inner_lim, outer_lim), (inner_lim, outer_lim), (inner_lim, outer_lim), (inner_lim, outer_lim), (inner_lim, outer_lim)]
configuration_offsets = [0, -pi/2, 0, -pi/2, 0, 0]

inverse_sensitivity = inner_lim/3
inverse_configuration_limits = [(offset - inverse_sensitivity, offset + inverse_sensitivity) for offset in configuration_offsets]
