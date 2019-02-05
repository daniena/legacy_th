from numpy import *
from numpy.linalg import *
from collections import namedtuple
from param_debug import debug

ca_task = namedtuple('collision_avoidance_task', ['p', 'p_0', 'sigma_min', 'sigma_max'])
fov_task = namedtuple('field_of_view_task', ['a', 'a_0', 'sigma_min', 'sigma_max'])

# sigma of top priority set based task
def sigma_tp_set_based(z, z_0, sigma_min, sigma_max):
    # Top priority set-based task
    sigma = norm(z_0-z)
    if debug and sigma < sigma_min:
        print('MTPIK_ERROR: Set-based task borders violation: sigma:', sigma, '& sigma_min:', sigma_min)
    if debug and sigma > sigma_max:
        print('MTPIK_ERROR: Set-based task borders violation: sigma:', sigma, '& sigma_max:', sigma_max)
    return sigma

# sigma of low priority set based task
def sigma_lp_set_based(z, z_0, sigma_min, sigma_max):
    # Low priority set-based task
    sigma = norm(z_0-z)
    if debug and sigma < sigma_min:
        print('MTPIK_WARNING: Set-based task borders transgression: sigma:', sigma, '& sigma_min:', sigma_min)
    if debug and sigma > sigma_max:
        print('MTPIK_WARNING: Set-based task borders transgression: sigma:', sigma, '& sigma_max:', sigma_max)
    return sigma

# sigma of equality based task
def sigma_eq_based(z, z_0):
    # Lowest priority equality-based task
    return norm(a_des - a)

def in_T_C(sigma_dot, sigma, sigma_min, sigma_max, epsilon):
            # Implementation directly from Moe et al. : Experimental results for set-based control within the singularity-robust multiple task-priority inverse kinematics framework, {2015 IEEE International Conference on Robotics and Biomimetics (ROBIO)}, doi={10.1109/ROBIO.2015.7418940}, https://ieeexplore.ieee.org/document/7418940, and : Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results, {Frontiers in Robotics and AI}, DOI={10.3389/frobt.2016.00016}, https://www.frontiersin.org/article/10.3389/frobt.2016.00016
    if sigma > sigma_min + epsilon and sigma < sigma_max - epsilon:
        return True
    elif sigma <= sigma_min + epsilon and sigma_dot >= 0:
        return True
    elif sigma <= sigma_min + epsilon and sigma_dot < 0:
        return False
    elif sigma >= sigma_max - epsilon and sigma_dot <= 0:
        return True
    elif sigma >= sigma_max - epsilon and sigma_dot > 0:
        return False

def Jca(p, p_0, sigma_ca, J_pos):
            # Implementation directly from Moe et al. : Experimental results for set-based control within the singularity-robust multiple task-priority inverse kinematics framework, {2015 IEEE International Conference on Robotics and Biomimetics (ROBIO)}, doi={10.1109/ROBIO.2015.7418940}, https://ieeexplore.ieee.org/document/7418940, and : Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results, {Frontiers in Robotics and AI}, DOI={10.3389/frobt.2016.00016}, https://www.frontiersin.org/article/10.3389/frobt.2016.00016
    x = transpose(p - p_0)/sigma_ca
    J_ca = dot(x, J_pos)
    return J_ca

def Jfov1dof(a, a_des, sigma_fov, J_fov3dof):
            # Implementation directly from Moe et al. : Experimental results for set-based control within the singularity-robust multiple task-priority inverse kinematics framework, {2015 IEEE International Conference on Robotics and Biomimetics (ROBIO)}, doi={10.1109/ROBIO.2015.7418940}, https://ieeexplore.ieee.org/document/7418940, and : Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results, {Frontiers in Robotics and AI}, DOI={10.3389/frobt.2016.00016}, https://www.frontiersin.org/article/10.3389/frobt.2016.00016
    x = transpose(a - a_des)/sigma_fov
    J_fov = dot(x, J_fov3dof)
    return J_fov

def a_desired_fov_on_target(p, p_des):
    return (p_des - p)/norm(p_des-p)

# Unused:
def a_desired_fov_on_nearby_obstacle_then_on_target(p, p_des, ca_tasks):
    pass
