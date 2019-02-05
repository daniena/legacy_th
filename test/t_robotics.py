from numpy import *
from util.matrix import *
from robotics.rtpwrapper import *
from robotics.task import *
from robotics.kinematics import *
from param_debug import debug
from math import *

def test_forward_kinematic_field_of_view():
    print('')

    # Note, a = z_6, i.e. the actuation axis of the sixth joint, z axis of the sixth joint
    
    # Test 1
    q = vector([-1.432, -1.191, -1.810, -2.341, -2.144, -0.652])
    a_correct = vector([0.6056, -0.4149, 0.6791])
    a = forward_kinematic_field_of_view(q)

    epsilon = 0.001
    for i in range(3):
        if debug:
            print('a[i] - a_correct[i]:')
            print(a[i] - a_correct[i])
        assert a[i] - a_correct[i] <  epsilon
        assert a[i] - a_correct[i] > -epsilon

    # Test 2
    q = vector([-0.915, -1.685, -1.741, -2.373, -2.156, -2.009])
    a_correct = vector([0.8877, -0.2479, 0.3880])
    a = forward_kinematic_field_of_view(q)

    epsilon = 0.001
    for i in range(3):
        if debug:
            print('a[i] - a_correct[i]:')
            print(a[i] - a_correct[i])
        assert a[i] - a_correct[i] <  epsilon
        assert a[i] - a_correct[i] > -epsilon

    # Test 3
    q = vector([0.915, -1.685, -0.5, -2.373, -2.156, -2.009])
    a_correct = vector([-0.5160, 0.2352, 0.8237])
    a = forward_kinematic_field_of_view(q)

    epsilon = 0.001
    for i in range(3):
        if debug:
            print('a[i] - a_correct[i]:')
            print(a[i] - a_correct[i])
        assert a[i] - a_correct[i] <  epsilon
        assert a[i] - a_correct[i] > -epsilon

    # Test 4
    q = vector([0.915, -1.685, -0.5, 1.3, -2.156, -2.009])
    a_correct = vector([-0.1159, 0.7552, -0.6451])
    a = forward_kinematic_field_of_view(q)

    epsilon = 0.001
    for i in range(3):
        if debug:
            print('a[i] - a_correct[i]:')
            print(a[i] - a_correct[i])
        assert a[i] - a_correct[i] <  epsilon
        assert a[i] - a_correct[i] > -epsilon

    print('Test: forward_kinematic_field_of_view for various configurations, ASSERT TRUE:', True)
    print('')

def test_Jpos():
    print('')
    q = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    solution = array([[0.2682, 0.2124, 0.1702, 0.0927, -0.0784, -0.0000], [-0.7639, 0.0213, 0.0171, 0.0093, 0.0004, 0.0000], [0, -0.7868, -0.3639, 0.0203, -0.0241, -0.0000], [0, 0.0998, 0.0998, 0.0998, 0.2940, 0.0044], [0, -0.9950, -0.9950, -0.9950,  0.0295, -0.9996], [1.0000, 0, 0, 0, -0.9553, -0.0295]])

    rtpwrapper = rtpUR5()

    if debug:
        print('_____test_Jpos_____')
        print('q')
        print(q)
        print('solution')
        print(solution)
        print('rtpwrapper.Jpos(q)')
        print(rtpwrapper.Jpos(q))
    
    #Tests whether the calculated Jacobian is correct
    print('Test: rtpwrapper.Jpos(q), ASSERT TRUE: ', matrix_double_equality(rtpwrapper.Jpos(q),solution,0.00005))
    
    #Add more only if needed
    print('')

def test_Jfov():
    print('')
    q = vector([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    a = vector([0, sin(pi/128), cos(pi/128)])
    a_des = vector([0, 0, 1])

    sigma_fov = sigma_tp_set_based(a, a_des, -infty, infty)

    J_fov3dof = jacobian_field_of_view(q)
    J_fov = Jfov1dof(a, a_des, sigma_fov, J_fov3dof)

    if debug:
        print('_____test_Jfov_____')
        print('q')
        print(q)
        print('a')
        print(a)
        print('a_des')
        print(a_des)
        print('J_fov')
        print(J_fov)
    
    #Add tests only if used

    print('')
    
def test_inverse_kinematic_priority_set_based_tasks_q_dot_ref():

    print('Warning: test_inverse_kinematic_priority_set_based_tasks_q_dot_ref currently not implemented correctly')
    return

    rtpwrapper = rtpUR5()
    
    p_w1 = vector([0.486, -0.066, -0.250])
    p_w2 = vector([0.320, 0.370, -0.250])

    ## Set up CA tasks:
    p_0_a = vector([0.4 , -0.25 , -0.33])
    sigma_a_min = 0.18
    sigma_a_max = infty

    p_0_b = vector([0.4 , 0.15  , -0.33])
    sigma_b_min = 0.15
    sigma_b_max = infty

    starting_positions = [p_0_a + vector([0, 0, 0.19]), # above
                          p_0_a + vector([0, 0.19, 0]), # to the right
                          p_0_a + vector([0, -0.19, 0]), # to the left
                          p_0_a + vector([0.19, 0, 0]), # in front of
                          p_0_a + vector([-0.19, 0, 0]), # behind
                          p_0_b + vector([0, 0, 0.19]), # above
                          p_0_b + vector([0, 0.19, 0]), # to the right
                          p_0_b + vector([0, -0.19, 0]), # to the left
                          p_0_b + vector([0.19, 0, 0]), # in front of
                          p_0_b + vector([-0.19, 0, 0])] # behind

    for start in starting_positions:
        p_w = [start, p_w1, p_w2]
        
        phase = 0
        epsilon = 0.02
        timestep = 0.02
        p_initial = p_w[0]
        p_target = p_w[0]
        q_dot_ref = 0
        q = vector([1.6, -0.5, 1, 0.1, 0.5, 0.1])
        i = 0
        while True:
            i = i +1
            #p = rtpwrapper.fpos(transpose(q))
            p = vector(forward_kinematic_position(q))
            top_priority_set_based_tasks = []

            if norm(p-p_target) < epsilon:
                if phase >= len(p_w)-1:
                    if debug is True:
                        print('End-waypoint reached!')
                        print('End-waypoint:', list(p_target[0:3,0]))
                        print('Current position:', list(p[0:3,0]))
                        print('Configuration is:', list(q[0:6,0]))
                    print('Test: test_inverse_kinematic_priority_set_based_tasks_q_dot_ref for start position\n', start, '\nASSERT TRUE:', 'True')
                    break
                else:
                    phase += 1
                    p_target = p_w[phase]
                    if debug is True:
                        print('Changing phase to:', phase)
                        print('Current position is:', list(p[0:3,0]))
                        print('Configuration is:', list(q[0:6,0]))
            
            if phase == 0:
                q_dot_ref, _, _ = inverse_kinematic_priority_set_based_tasks_q_dot_ref(p, p_initial, q, rtpwrapper, [], epsilon, timestep)
            elif phase >= 1 :
                #top_priority_set_based_tasks = [ca_task(p, p_0_a, sigma_a_min, sigma_a_max),
                #                           ca_task(p, p_0_b, sigma_b_min, sigma_b_max)]
                top_priority_set_based_tasks = [ca_task(p, p_0_a, sigma_a_min, sigma_a_max),
                                                ca_task(p, p_0_b, sigma_b_min, sigma_b_max)]
                for task in top_priority_set_based_tasks:
                    if type(task).__name__ is 'collision_avoidance_task':
                        task_sigma = sigma_tp_set_based(task.p, task.p_0, task.sigma_min, task.sigma_max)
                        if task_sigma < task.sigma_min:
                            print('Test: test_get_q_dot_ref_top_priority_set_based_perfect_position_control for start position\n', start, '\nASSERT TRUE:', 'False (obstacle collision)')
                    if type(task).__name__ is 'field_of_view_task':
                        task_sigma = sigma_tp_set_based(a, task.a_0, task.sigma_min, task.sigma_max)
                        if task_sigma < task.sigma_min or task_sigma > task.sigma_max:
                            print('Test: test_get_q_dot_ref_top_priority_set_based_perfect_position_control for start position\n', start, '\nASSERT TRUE:', 'False (fov constraint violation)')
                    
                q_dot_ref, _, _ = inverse_kinematic_priority_set_based_tasks_q_dot_ref(p, p_target, q, rtpwrapper, top_priority_set_based_tasks, epsilon, timestep)
            q = q + q_dot_ref
            if i%10 == 0 and debug is True:
                print(i)
                print(q)
                print(q_dot_ref)
                print(p_target)
                print(p-p_target)
            if i >= 3000:
                print('Test: test_inverse_kinematic_priority_set_based_tasks_q_dot_ref for start position\n', start, '\nASSERT TRUE:', 'False (took too long)')
    print('')

    
    
    
