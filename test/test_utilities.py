from robotics.task import *
from robotics.rtpwrapper import *
from util.matrix import *
from numpy import *
from simulation.workspace import *
from simulation.simulation import *
from random import Random
from util.file_operations import remove_files_with_ending

def simple_world_init():
    rtpwrapped = rtpUR5()

    p_w0 = vector([0, -0.5, 0])
    p_w1 = vector([0.486, -0.066, -0.250])
    p_w2 = vector([0.320, 0.370, -0.250])
    
    waypoints = (p_w0, p_w1, p_w2)
    
    ## Set up CA tasks:
    p_0_a = vector([0.4 , -0.25 , -0.33])
    sigma_a_min = 0.18
    sigma_a_max = infty
    
    p_0_b = vector([0.4 , 0.15  , -0.33])
    sigma_b_min = 0.15
    sigma_b_max = infty

    p = []
    top_priority_set_based_tasks = [ca_task(p, p_0_a, sigma_a_min, sigma_a_max),
                                    ca_task(p, p_0_b, sigma_b_min, sigma_b_max)]

    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']

    return (ca_tasks, waypoints, rtpwrapped)

def simple_world_init_new():
    rtpwrapped = rtpUR5()

    q_init = vector([-48, -158, -108, -90, -85, 0])*pi/180
    p_w1 = vector([0.486, -0.066, -0.250])
    p_w2 = vector([0.320, 0.370, -0.250])
    
    waypoints = (p_w1, p_w2)
    
    ## Set up CA tasks:
    p_0_a = vector([0.4 , -0.25 , -0.33])
    sigma_a_min = 0.18
    sigma_a_max = infty
    
    p_0_b = vector([0.4 , 0.15  , -0.33])
    sigma_b_min = 0.15
    sigma_b_max = infty

    p = []
    top_priority_set_based_tasks = [ca_task(p, p_0_a, sigma_a_min, sigma_a_max),
                                    ca_task(p, p_0_b, sigma_b_min, sigma_b_max)]

    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']

    return (ca_tasks, q_init, waypoints, rtpwrapped)
