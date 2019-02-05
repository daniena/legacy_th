from math import *
from numpy import *
from util.matrix import *
from .task import *
from .rtpwrapper import * # Try to eliminate rtp dependency, it is slow
from .ur5_param import d1, d4, d5, d6, a2, a3, alpha1, alpha4, alpha5
from param_debug import debug

def DH_transformation(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    

    def T_z(phi, d):
        return array([[cos(phi), -sin(phi), 0,  0],
                      [sin(phi),  cos(phi), 0,  0],
                      [       0,         0, 1,  d],
                      [       0,         0, 0,  1]])
    def T_x(alpha, a):
        return array([[1,          0,           0,  a],
                      [0, cos(alpha), -sin(alpha),  0],
                      [0, sin(alpha),  cos(alpha),  0],
                      [0,          0,           0,  1]])

    T_from_link1_to_world = dot(T_z(q1, d1), T_x(alpha1,  0))
    T_from_link2_to_link1 = dot(T_z(q2,  0), T_x(     0, a2))
    T_from_link3_to_link2 = dot(T_z(q3,  0), T_x(     0, a3))
    T_from_link4_to_link3 = dot(T_z(q4, d4), T_x(alpha4,  0))
    T_from_link5_to_link4 = dot(T_z(q5, d5), T_x(alpha5,  0))
    T_from_link6_to_link5 = dot(T_z(q6, d6), T_x(     0,  0)) # This is not working, currently

    return multidot([T_from_link1_to_world, T_from_link2_to_link1, T_from_link3_to_link2, T_from_link4_to_link3, T_from_link5_to_link4, T_from_link6_to_link5])

def forward_kinematic_position(q):    
    # Author of this function: Signe Moe
    q1 = q[0];
    q2 = q[1];
    q3 = q[2];
    q4 = q[3];
    q5 = q[4];
    q6 = q[5];

    p_ee = array([ (d6*(cos(q5)*sin(q1) - cos(q2 + q3 + q4)*cos(q1)*sin(q5)) + d4*sin(q1) + a2*cos(q1)*cos(q2) + d5*sin(q2 + q3 + q4)*cos(q1) + a3*cos(q1)*cos(q2)*cos(q3) - a3*cos(q1)*sin(q2)*sin(q3)),
            (a2*cos(q2)*sin(q1) - d4*cos(q1) - d6*(cos(q1)*cos(q5) + cos(q2 + q3 + q4)*sin(q1)*sin(q5)) + d5*sin(q2 + q3 + q4)*sin(q1) + a3*cos(q2)*cos(q3)*sin(q1) - a3*sin(q1)*sin(q2)*sin(q3)),
            (d1 + a3*sin(q2 + q3) + a2*sin(q2) - d5*(cos(q2 + q3)*cos(q4) - sin(q2 + q3)*sin(q4)) - d6*sin(q5)*(cos(q2 + q3)*sin(q4) + sin(q2 + q3)*cos(q4))) ])
    return p_ee

def forward_kinematic_field_of_view(q):
    a = DH_transformation(q)[0:3, 2]
    return a
    
def jacobian_linear_velocity(q, rtpwrapper):
    J_pos = rtpwrapper.Jpos(transpose(q))[0:3,:] # rtowrapper.Jpos name no longer makes sense, it is jacobian_velocity
    return J_pos

def jacobian_field_of_view(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    
    # Found using MATLAB, understood through guidance from Signe Moe
    Jfov3dof = array([[cos(q1)*cos(q5) + cos(q2 + q3 + q4)*sin(q1)*sin(q5), sin(q2 + q3 + q4)*cos(q1)*sin(q5), sin(q2 + q3 + q4)*cos(q1)*sin(q5), sin(q2 + q3 + q4)*cos(q1)*sin(q5), - sin(q1)*sin(q5) - cos(q2 + q3 + q4)*cos(q1)*cos(q5), 0],
                        [cos(q5)*sin(q1) - cos(q2 + q3 + q4)*cos(q1)*sin(q5), sin(q2 + q3 + q4)*sin(q1)*sin(q5), sin(q2 + q3 + q4)*sin(q1)*sin(q5), sin(q2 + q3 + q4)*sin(q1)*sin(q5),   cos(q1)*sin(q5) - cos(q2 + q3 + q4)*cos(q5)*sin(q1), 0],
                        [0,        -cos(q2 + q3 + q4)*sin(q5),        -cos(q2 + q3 + q4)*sin(q5),        -cos(q2 + q3 + q4)*sin(q5),                            -sin(q2 + q3 + q4)*cos(q5), 0]])
    return Jfov3dof

# ca = ["ca", p_0, sigma_min, sigma_max]
# fov = ["fov", args]

# This function is deprecated, though still used in testing branch (I think...)
def inverse_kinematic_priority_set_based_tasks_q_dot_ref(p, p_des,  q, rtpwrapper, priority_set_based_tasks, epsilon, timestep):
        # Implementation based on articles by Moe et al. : Experimental results for set-based control within the singularity-robust multiple task-priority inverse kinematics framework, {2015 IEEE International Conference on Robotics and Biomimetics (ROBIO)}, doi={10.1109/ROBIO.2015.7418940}, https://ieeexplore.ieee.org/document/7418940, and : Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results, {Frontiers in Robotics and AI}, DOI={10.3389/frobt.2016.00016}, https://www.frontiersin.org/article/10.3389/frobt.2016.00016
    
    # Set up equality based tasks:
    sigma_pos = p
    sigma_pos_des = p_des
    sigma_pos_err = p_des - p
    J_pos = jacobian_linear_velocity(q, rtpwrapper)
    
    Lambda = diag([0.3, 0.3, 0.3])
    f1 = multidot([dagger(J_pos), Lambda, sigma_pos_err])*timestep # Assuming no obstacle, this is the update that takes the end effector on a direct path to the desired point p_des
    
    J_aug = []
    num_active = 0
    active_set = [ 0 for _ in priority_set_based_tasks]
    
    for i, task in enumerate(priority_set_based_tasks):
    
        task_sigma = -1
        task_sigma_naive = 0
        task_J = []
        
        if type(task).__name__ is 'collision_avoidance_task':
            task_sigma = sigma_tp_set_based(task.p, task.p_0, task.sigma_min, task.sigma_max)
            task_J = Jca(task.p, task.p_0, task_sigma, J_pos)
            task_sigma_naive = sigma_tp_set_based(rtpwrapper.fpos(transpose(q+f1)), task.p_0, task.sigma_min, task.sigma_max)
        if type(task).__name__ is 'field_of_view_task': # Not yet implemented
            task_sigma = sigma_tp_set_based(task.a, task.a_0, task.sigma_min, task.sigma_max)
            task_J = Jfov(task.a, task.a_0, task_sigma, q)
            task_sigma_naive = sigma_tp_set_based([0, 0, 1], task.a_0, task.sigma_min, task.sigma_max)

        active = not in_T_C(dot(task_J,f1), task_sigma_naive, task.sigma_min, task.sigma_max, epsilon)
        
        if active:
            active_set[i] = 1
            if num_active == 0:
                J_aug = task_J
                num_active = 1
            else:
                J_aug = vstack((J_aug, task_J))
                num_active += 1
    if debug:
        print('active sets:', active_set)

    if num_active == 0:
        return (f1, f1, active_set)
    if num_active >= 1:
        Null_space_projection = eye(J_aug.shape[1]) - dot(dagger(J_aug), J_aug)
        return (dot(Null_space_projection, f1), f1, active_set)

#------------------------------------------------------------------------------------------------------------------------------------------------

def null_space_projection(null_space_of_jacobian):
    return  eye(null_space_of_jacobian.shape[1]) - dot(dagger(null_space_of_jacobian), null_space_of_jacobian)

def inverse_kinematic_priority_set_based_tasks_q_dot_ref_new(p, p_des, a, a_des,  q, rtpwrapper, priority_set_based_tasks, epsilon, timestep):
    # Implementation based on articles by Moe et al. : Experimental results for set-based control within the singularity-robust multiple task-priority inverse kinematics framework, {2015 IEEE International Conference on Robotics and Biomimetics (ROBIO)}, doi={10.1109/ROBIO.2015.7418940}, https://ieeexplore.ieee.org/document/7418940, and : Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results, {Frontiers in Robotics and AI}, DOI={10.3389/frobt.2016.00016}, https://www.frontiersin.org/article/10.3389/frobt.2016.00016
    
    # Set up equality based tasks:
    sigma_pos = p
    sigma_pos_des = p_des
    sigma_pos_err = sigma_pos_des - sigma_pos
    J_pos = jacobian_linear_velocity(q, rtpwrapper)

    sigma_fov = norm(a_des-a)
    sigma_fov_des = 0
    sigma_fov_err = sigma_fov_des - sigma_fov
    J_fov3dof = jacobian_field_of_view(q)
    J_fov = Jfov1dof(a, a_des, sigma_fov, J_fov3dof)
    
    Lambda_1 = diag([0.3, 0.3, 0.3])
    Lambda_2 = 0.05

    N1 = null_space_projection(J_pos)
    f1 = multidot([dagger(J_pos), Lambda_1, sigma_pos_err])*timestep + multidot([N1, dagger(J_fov), Lambda_2, sigma_fov_err])*timestep # Assuming no obstacle, this is the update that takes the end effector on a direct path to the desired point p_des, and points it in the direction of a_des
    
    J_aug = []
    num_active = 0
    active_set = [ 0 for _ in priority_set_based_tasks]
    
    for i, task in enumerate(priority_set_based_tasks):
    
        task_sigma = -1
        task_sigma_naive = 0
        task_J = []
        
        if type(task).__name__ is 'collision_avoidance_task':
            task_sigma = sigma_tp_set_based(task.p, task.p_0, task.sigma_min, task.sigma_max)
            task_J = Jca(task.p, task.p_0, task_sigma, J_pos)
            task_sigma_naive = sigma_tp_set_based(rtpwrapper.fpos(transpose(q+f1)), task.p_0, task.sigma_min, task.sigma_max)
        if type(task).__name__ is 'field_of_view_task':
            task_sigma = sigma_tp_set_based(task.a, task.a_0, task.sigma_min, task.sigma_max)
            task_J = Jfov(task.a, task.a_0, task_sigma, q)
            task_sigma_naive = sigma_tp_set_based([0, 0, 1], task.a_0, task.sigma_min, task.sigma_max)

        active = not in_T_C(dot(task_J,f1), task_sigma, task.sigma_min, task.sigma_max, epsilon)
        
        if active:
            active_set[i] = 1
            if num_active == 0:
                J_aug = task_J
                num_active = 1
            else:
                J_aug = vstack((J_aug, task_J))
                num_active += 1
    if debug:
        print('active sets:', active_set)

    if num_active == 0:
        q_dot_ref = f1
    if num_active >= 1:
        N_pos = null_space_projection(J_aug)
        N_fov = null_space_projection(vstack((J_pos, J_aug)))
        q_dot_ref = multidot([N_pos, dagger(J_pos), Lambda_1, sigma_pos_err])*timestep + multidot([N_fov, dagger(J_fov), Lambda_2, sigma_fov_err])
    return (q_dot_ref, f1, active_set)
