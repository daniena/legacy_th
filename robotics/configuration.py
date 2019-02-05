from .ur5_param import configuration_limits, configuration_offsets, inverse_configuration_limits
from math import pi

def generate_random_configuration_limited(random):
    q_flip_sign = [random.choice([-1, 1]) for _ in range(5)]
                
    q1 = random.uniform(configuration_limits[0][0], configuration_limits[0][1])                + configuration_offsets[0] # TODO: Change these to list comprehension
    q2 = random.uniform(configuration_limits[1][0], configuration_limits[1][1])*q_flip_sign[0] + configuration_offsets[1]
    q3 = random.uniform(configuration_limits[2][0], configuration_limits[2][1])*q_flip_sign[1] + configuration_offsets[2]
    q4 = random.uniform(configuration_limits[3][0], configuration_limits[3][1])*q_flip_sign[2] + configuration_offsets[3]
    q5 = random.uniform(configuration_limits[4][0], configuration_limits[4][1])*q_flip_sign[3] + configuration_offsets[4]
    q6 = random.uniform(configuration_limits[5][0], configuration_limits[5][1])*q_flip_sign[4] + configuration_offsets[5]

    return [q1, q2, q3, q4, q5, q6]

def generate_random_configuration_no_limits(random):
    q_flip_sign = [random.choice([-1, 1]) for _ in range(5)]
                
    q1 = random.uniform(-pi, pi) + configuration_offsets[0] # TODO: Change these to list comprehension
    q2 = random.uniform(-pi, pi) + configuration_offsets[1]
    q3 = random.uniform(-pi, pi) + configuration_offsets[2]
    q4 = random.uniform(-pi, pi) + configuration_offsets[3]
    q5 = random.uniform(-pi, pi) + configuration_offsets[4]
    q6 = random.uniform(-pi, pi) + configuration_offsets[5]

    return [q1, q2, q3, q4, q5, q6]

def is_inverse(q):
    inverse_q = [0 for _ in q]
    i = 0
    for q_i, inv_lim_i in zip(q, inverse_configuration_limits):
        if q_i > inv_lim_i[0] and  q_i < inv_lim_i[1]:
            inverse_q[i] = 1
        i += 1
    return inverse_q
