from mujoco_py import load_model_from_path, MjSim, MjViewer
from .write_XML import *
from util.matrix import *
from robotics.kinematics import *
from robotics.task import *
from robotics.rtpwrapper import *
from robotics.ur5_param import *
from robotics.configuration import generate_random_configuration_no_limits, is_inverse
from simulation.workspace import *
import os
#from collections import namedtuple
import pandas as pd
import numpy as np
from random import Random
from param_debug import test, debug, view
from session import max_obstacles
from learning.rawdata import make_obstacle_slots
#from sys import exit

def exit_criteria_at_end_waypoint_only(step, max_steps, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w_current, phase, end_phase, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints):
    if phase == end_phase:
        return ("Success", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, step)
    else:
        return ("Continue",)

def exit_criteria_at_end_waypoint_or_i_max(step, max_steps, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w_current, phase, end_phase, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints):
    if phase == end_phase:
        if any(obstacle_highest_percentage_to_center_penetrated):
            return ("Failure: Violation", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, step)
        else:
            return ("Success", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, step)
    else:
        if step >= max_steps:
            return ("Failure: Max timesteps", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, step)
        return ("Continue",)

def exit_criteria_at_i_max_only(step, max_steps, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w_current, phase, end_phase, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints):
    if step >= max_steps:
        return ("Success", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, step)
    else:
        return ("Continue",)
        
def _loop(top_priority_set_based_tasks, initial_configuration, waypoint_positions, rtpwrapped, max_steps, history, actuators='position', circle_of_acceptance=0.02, exit_criteria=None, random=None, record=False, inference_model=None):
    model = []
    path = ''
    if not test:
        path = './simulation/xmls/world.xml'
    else:
        path = './test/xmls/world.xml' # Testing and generating at the same time would lock the file for generation, causing a crash

    attempt = 0
    while not model:
        try:
            model = load_model_from_path(path)
        except:
            if attempt == 0:
                print('Warning, ' + path + ' locked for reading. Avoid race conditions.')
            elif attempt >= 10000:
                print('Cannot access ' + path + ' after 10000 attempts. Exiting.')
                exit(1)

            
    timestep = 0.02

    sim = MjSim(model)
    viewer = []
    if view:
        viewer = MjViewer(sim)
    sim_state = sim.get_state()
    
    phase = 0
    q = [0 for _ in range(6)]
    q_prev = [0 for _ in range(6)]
    q_ref = [0 for _ in range(6)]
    q_ref_prev = [0 for _ in range(6)]
    q_dot_ref = [0 for _ in range(6)]
    f1 = [0 for _ in range(6)]
    p_w = waypoint_positions
    p_target = p_w[0]
    collision_detected = False

    num_timesteps_obstacle_actively_avoided = [0 for _ in top_priority_set_based_tasks]
    set_of_actively_avoided_obstacles_at_same_timestep = set()
    obstacle_highest_percentage_to_center_penetrated = [0 for _ in top_priority_set_based_tasks]
    union_set_of_singular_joints = [ 0 for _ in range(6) ]
    
    while True:
        joint_address = [sim.model.get_joint_qpos_addr("joint1"),
                         sim.model.get_joint_qpos_addr("joint2"),
                         sim.model.get_joint_qpos_addr("joint3"),
                         sim.model.get_joint_qpos_addr("joint4"),
                         sim.model.get_joint_qpos_addr("joint5"),
                         sim.model.get_joint_qpos_addr("joint6")]
        if debug:
            print('joint_address')
            print(joint_address)

        sim_state.qpos[joint_address[0:6]] = list(initial_configuration[0:6,0])
        sim.set_state(sim_state)
        sim.forward()

        #scenario_workspace = pd.DataFrame( empty )
        #scenario_trajectory = pd.DataFrame( empty and large enough)
        
    
        i = 0
        while True:
            q_prev = q
            q = vector(sim.data.qpos[joint_address[0:6]])
            p = vector(forward_kinematic_position(q))

            a = vector(forward_kinematic_field_of_view(q))
            a_target = a_desired_fov_on_target(p, p_target)
                
            if norm(p-p_target) < circle_of_acceptance and actuators is not 'random_position':
                phase += 1
                if phase >= len(p_w):
                    if debug:
                        print('End-waypoint reached!')
                        print('End-waypoint:', list(p_target[0:3,0]))
                        print('Current position:', list(p[0:3,0]))
                        print('Current configuration:', list(q[0:6,0]))
                    
                    if exit_criteria is not None:
                        status = exit_criteria(i, max_steps, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w[phase-1], phase, len(p_w), num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints)
                        if  status[0] is not "Continue":
                            return status
                    
                    break;
                else:
                    p_target = p_w[phase]
                    if debug:
                        print('Changing phase to:', phase)
                        print('Configuration is:', list(q[0:6,0]))
                        print('ee position is:', list(p[0:3,0]))
            if i%1000 == 0 and debug:
                sensor_p = vector(sim.data.sensordata)
                print('p')
                print(p)
                print('sensor_p')
                print(sensor_p)
                print('ee position difference has norm:' +  str(norm(p[0:3,0] - sensor_p[0:3,0])))
            
            if exit_criteria is not None:
                status = exit_criteria(i, max_steps, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w[phase], phase, len(p_w), num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints)
                if  status[0] is not "Continue":
                    return status
            
            # Update task variable p/a and check for task constraint violation 
            for index, task in enumerate(top_priority_set_based_tasks):
                if type(task).__name__ is 'collision_avoidance_task':
                    top_priority_set_based_tasks[index] = ca_task(p, task.p_0, task.sigma_min, task.sigma_max)
                    task = top_priority_set_based_tasks[index]
                    task_sigma = sigma_tp_set_based(task.p, task.p_0, task.sigma_min, task.sigma_max)
                    if task_sigma < task.sigma_min:
                        penetration_percentage = (task.sigma_min - task_sigma)/task.sigma_min
                        if obstacle_highest_percentage_to_center_penetrated[index] < penetration_percentage:
                            obstacle_highest_percentage_to_center_penetrated[index] = penetration_percentage
                            if not collision_detected:
                                collision_detected = True
                                print('Warning: Collision detected.')
                if type(task).__name__ is 'field_of_view_task':
                    top_priority_set_based_tasks[index] = ca_task(a, task.a_0, task.sigma_min, task.sigma_max)
                    task = top_priority_set_based_tasks[index]
                    task_sigma = sigma_tp_set_based(a, task.a_0, task.sigma_min, task.sigma_max)

            activation_mask = [0 for _ in top_priority_set_based_tasks]

            if inference_model is None:
                #try:
                (q_dot_ref, f1, activation_mask) = inverse_kinematic_priority_set_based_tasks_q_dot_ref_new(p, p_target, a, a_target, q, rtpwrapped, top_priority_set_based_tasks, circle_of_acceptance, timestep)
                #except LinAlgError:
                #    return ("Failure: Inverse Matrix", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, i-1)
            else:
                obstacles_buffer = [ list(task.p_0[:,0]) + [task.sigma_min]  for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task' ]
                q_dot_ref = inference_model.simulation_input_to_output(list(q[:,0]), list(p[:,0]), list(p_target[:,0]), list(a[:,0]), list(a_target[:,0]), obstacles_buffer, max_obstacles)
                f1 = vector([0, 0, 0, 0, 0, 0])
                
            for activation_index in range(len(num_timesteps_obstacle_actively_avoided)):
                num_timesteps_obstacle_actively_avoided[activation_index] += activation_mask[activation_index]
            if any(activation_mask):
                set_of_actively_avoided_obstacles_at_same_timestep.add(tuple(activation_mask))
            
            q_ref_prev = q_ref
            q_ref = q + q_dot_ref

            if record:
                inverse_joints = is_inverse(q)
                for q_inv_index, q_inv_i in enumerate(inverse_joints):
                    if q_inv_i:
                        union_set_of_singular_joints[q_inv_index] = 1

                timestep_data = list(q[:,0]) + list(p[:,0]) + list(f1[:,0]) + list(q_dot_ref[:,0]) + activation_mask + inverse_joints
                
                history.iloc[i][0:len(timestep_data)] = timestep_data
            
            if actuators is 'position':
                sim.data.ctrl[joint_address[0:6]] = list(q_ref[0:6,0])
            elif actuators is 'velocity':
                sim.data.ctrl[joint_address[0:6]] = list(q_dot_ref[0:6,0])
            elif actuators is 'perfect_position':
                sim.data.qpos[joint_address[0:6]] = list(q_ref[0:6,0])
            elif actuators is 'random_position':
                sim.data.qpos[joint_address[0:6]] = generate_random_configuration_no_limits(random) #[random.uniform(-pi, pi)] + [random.uniform(-pi + pi/3, -pi/3)*random.choice([-1,1]) for _ in range(5)]
            else:
                print('Error: No actuators', actuators, 'in simulation._loop()')
                exit(1)

            i += 1
            try:
                sim.forward()
                sim.step()
            except:
                return ("Failure: MuJoCo unstable simulation", num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, i)
                
            if view:
                try:
                    viewer.render()
                except Exception as e:
                    print('viewer excited with the error message:', repr(e))
                    break
        
        if os.getenv('TESTING') is not None:
            break

def _loop_static_reference(waypoints, q_targets, rtpwrapped, circle_of_acceptance=0.02, exit_criteria=None):
    model = load_model_from_path('./simulation/xmls/world.xml')

    sim = MjSim(model)
    viewer = []
    if view:
        viewer = MjViewer(sim)
    sim_state = sim.get_state()
    
    phase = 0
    q = []
    q_prev = []
    q_ref = []
    q_ref_prev = []
    q_dot_ref = []
    p_w = waypoints
    p_initial = p_w[0]
    p_target = p_w[0]
    q_target = q_targets[0]
    timestep = 0.02
    while True:
        joint_address = [sim.model.get_joint_qpos_addr("joint1"),
                         sim.model.get_joint_qpos_addr("joint2"),
                         sim.model.get_joint_qpos_addr("joint3"),
                         sim.model.get_joint_qpos_addr("joint4"),
                         sim.model.get_joint_qpos_addr("joint5"),
                         sim.model.get_joint_qpos_addr("joint6")]
        if debug:
            print('joint_address')
            print(joint_address)
        sim_state.qpos[joint_address[0:6]] = [2.557336100227215, 0.13124086714164798, 0.5755827073670068, -0.22698177425694496, 0.6509848793673406, 0.1] #[1.6, -0.5, 1, 0.1, 0.5, 0.1]
        sim.set_state(sim_state)
        sim.forward()
    
        i = 0
        while True:
            i += 1
            q_prev = q
            q = vector(sim.data.qpos[joint_address[0:6]])
            p = vector(forward_kinematic_position(q))
                
            if norm(p-p_target) < circle_of_acceptance:
                if phase >= len(p_w)-1:
                    if debug:
                        print('End-waypoint reached!')
                        print('End-waypoint:', list(p_target[0:3,0]))
                        print('Current position:', list(p[0:3,0]))
                    
                    if exit_criteria is not None:
                        status = exit_criteria(i, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w[phase], phase, len(p_w)-1)
                        if  status[0] is not "Continue":
                            return status
                    
                    break;
                else:
                    phase += 1
                    p_target = p_w[phase]
                    q_target = q_targets[phase]
                    if debug:
                        print('Changing phase to:', phase)
                        print('Configuration is:', list(q[0:6,0]))
            
            if exit_criteria is not None:
                status = exit_criteria(i, q, q_prev, q_ref, q_ref_prev, q_dot_ref, p, p_w[phase], phase, len(p_w))
                if  status[0] is not "Continue":
                    
                    return status

            q_ref = q_target
            
            sim.data.ctrl[joint_address[0:6]] = list(q_ref[0:6,0])
            sim.forward()
            sim.step()
                
            if view:
                try:
                    viewer.render()
                except Exception as e:
                    print('viewer excited with the error message:', repr(e))
                    break
        
        if os.getenv('TESTING') is not None:
            break
    
def simulate_perfect_position_control(workspace_dim, top_priority_set_based_tasks, waypoints, rtpwrapped, exit_criteria=None):
    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']
    
    write_workspace(workspace_dim, ca_tasks, waypoints)
    write_actuators('velocity') # So that the unset position controller does not actuate forces on the manipulator

    return _loop(top_priority_set_based_tasks, waypoints, rtpwrapped, actuators='perfect_position', exit_criteria=exit_criteria)

def simulate_position_control(workspace_dim, top_priority_set_based_tasks, waypoints, rtpwrapped, exit_criteria=None):
    actuators = 'position'
    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']
    
    write_workspace(workspace_dim, ca_tasks, waypoints)
    write_actuators(actuators)

    return _loop(top_priority_set_based_tasks, waypoints, rtpwrapped, actuators=actuators, exit_criteria=exit_criteria)

def simulate_velocity_control(workspace_dim, top_priority_set_based_tasks, waypoints, rtpwrapped, exit_criteria=None):
    actuators = 'velocity'
    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']
    
    write_workspace(workspace_dim, ca_tasks, waypoints)
    write_actuators(actuators)

    return _loop(top_priority_set_based_tasks, waypoints, rtpwrapped, actuators=actuators, exit_criteria=exit_criteria)

def simulate_position_control_static_reference(workspace_dim, waypoints, q_targets, rtpwrapped, exit_criteria=None):
    actuators = 'position'
    
    write_workspace(workspace_dim, [], waypoints)
    write_actuators(actuators)

    return _loop_static_reference(waypoints, q_targets, rtpwrapped, actuators=actuators, exit_criteria=exit_criteria)

def simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_steps, history, actuators='position', exit_criteria=None, random=None, record=False, inference_model=None):
    ca_tasks = [task for task in top_priority_set_based_tasks if type(task).__name__ is 'collision_avoidance_task']

    write_workspace(ca_tasks, waypoint_cartesian_positions)
    if actuators is 'perfect_position':
        write_actuators('velocity')
    elif actuators is 'random_position':
        write_actuators('velocity')
    else:
        write_actuators(actuators)

    return _loop(top_priority_set_based_tasks, waypoint_configurations[0], waypoint_cartesian_positions[1:], rtpwrapped, max_steps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)
    

def generate_and_simulate(random, history, max_timesteps, exit_criteria, max_obstacles=5, actuators='position', record=False, inference_model=None):
    
    n_obstacles = 0
    if max_obstacles > 0:
        n_obstacles = random.randint(0,max_obstacles)
    
    workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions = generate_random_populated_workspace(random, n_obstacles)
    
    ca_tasks = [ 0 for _ in cartesian_obstacles ]
    for obst_i, obstacle in enumerate(cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, infty)
    

    top_priority_set_based_tasks = ca_tasks

    rtpwrapped = rtpUR5()
    (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)

    return (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)

def generate_and_simulate_biased_near_trajectory(random, history, max_timesteps, exit_criteria, obstacles_on_line_between_waypoints_percentage, obstacles_near_line_between_waypoints_percentage, max_obstacles=5, actuators='position', record=False, inference_model=None):
    
    dim_range = (1, 1.01)
    workspace_dim_range = cartesian_range_limits(*dim_range, *dim_range, *dim_range)
    border_epsilon = 0.020001
    obstacle_radius_range = spherical_radius_limits(0.04, 0.3)
    n_waypoints = 2
    min_adjacent_waypoint_distance = 0.6
    too_close_to_manipulator_space = cartesian_cube(-0.3, 0.3, -0.3, 0.3, -1, 1)
    inner_lim = pi/3
    outer_lim = pi-pi/3
    configuration_limits = [(-pi, pi), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim), (-outer_lim, -inner_lim)] 

    obstacles_randomly_placed_percentage = 1 - (obstacles_on_line_between_waypoints_percentage + obstacles_near_line_between_waypoints_percentage)
    if 0 > obstacles_on_line_between_waypoints_percentage + obstacles_near_line_between_waypoints_percentage > 1:
        print('Error: percentages cannot be higher than 1 or less than 0')
        exit(1)
    
    n_obstacles = 0
    n_obst_on_line = 0
    n_obst_near_line = 0
    if max_obstacles > 0:
        n_obstacles = random.randint(0,max_obstacles)

        for _ in range(n_obstacles):
            random_outcome = random.uniform(0,1)
            if random_outcome < obstacles_on_line_between_waypoints_percentage:
                n_obst_on_line += 1
            elif random_outcome < obstacles_on_line_between_waypoints_percentage + obstacles_near_line_between_waypoints_percentage:
                n_obst_near_line += 1
            # else n_obst_unbiased += 1, but redundant

    workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions = generate_random_populated_workspace_obstacles_biased_near_trajectory(random,
                                                                                                                                                                 n_obstacles,
                                                                                                                                                                 n_obst_on_line,
                                                                                                                                                                 n_obst_near_line)
    
    ca_tasks = [ 0 for _ in cartesian_obstacles ]
    for obst_i, obstacle in enumerate(cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, infty)
    

    top_priority_set_based_tasks = ca_tasks

    rtpwrapped = rtpUR5()
    (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)

    return (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)

def generate_and_simulate_forced_bias_near_trajectory(random, history, max_timesteps, exit_criteria, n_obstacles, n_obst_on_line, n_obst_near_line, actuators='position', record=False, inference_model=None):

    workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions = generate_random_populated_workspace_obstacles_biased_near_trajectory(random,
                                                                                                                                                                 n_obstacles,
                                                                                                                                                                 n_obst_on_line,
                                                                                                                                                                 n_obst_near_line)
    
    ca_tasks = [ 0 for _ in cartesian_obstacles ]
    for obst_i, obstacle in enumerate(cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, infty)
    

    top_priority_set_based_tasks = ca_tasks

    rtpwrapped = rtpUR5()
    (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)

    return (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)

def generate_and_simulate_forced_bias_near_trajectory_old(random, history, max_timesteps, exit_criteria, n_obstacles, n_obst_on_line, n_obst_near_line, actuators='position', record=False, inference_model=None):

    workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions = generate_random_populated_workspace_obstacles_biased_near_trajectory(random,
                                                                                                                                                                 n_obstacles,
                                                                                                                                                                 n_obst_on_line,
                                                                                                                                                                 n_obst_near_line)
    
    ca_tasks = [ 0 for _ in cartesian_obstacles ]
    for obst_i, obstacle in enumerate(cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, infty)
    

    top_priority_set_based_tasks = ca_tasks

    rtpwrapped = rtpUR5()
    (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)

    return (waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)

def generate_and_simulate_forced_bias_pattern_near_trajectory(random, history, max_timesteps, exit_criteria, max_obstacles=5, actuators='position', record=False, inference_model=None):

    outcome = random.randint(0,5)
    
    n_obstacles = 0
    n_obst_on_line = 0
    n_obst_near_line = 0

    if max_obstacles > 0:
        if outcome % 5 == 0:
            n_obstacles = random.randint(0, max_obstacles)
        elif outcome % 5 < 3 :
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles - n_obst_on_line)
        else:
            # Switch the bias to let n_obst_near_line also have a higher chance of rolling a higher amount of obstacles over n_obst_on_line, so n_obst_on_line has the same average as n_obst_near_line
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles - n_obst_near_line)
        
    
    return generate_and_simulate_forced_bias_near_trajectory(random,
                                                             history,
                                                             max_timesteps, exit_criteria,
                                                             n_obstacles, n_obst_on_line, n_obst_near_line,
                                                             actuators=actuators,
                                                             record=record,
                                                             inference_model=inference_model)

def generate_and_simulate_forced_bias_near_trajectory_infer_and_compare(random, history_a, history_b, history_no_ca, inference_model_a, inference_model_b, max_timesteps, exit_criteria, n_obstacles, n_obst_on_line, n_obst_near_line, actuators='position', record=False):

    workspace, cartesian_obstacles, waypoint_configurations, waypoint_cartesian_positions = generate_random_populated_workspace_obstacles_biased_near_trajectory(random,
                                                                                                                                                                 n_obstacles,
                                                                                                                                                                 n_obst_on_line,
                                                                                                                                                                 n_obst_near_line)
    
    ca_tasks = [ 0 for _ in cartesian_obstacles ]
    for obst_i, obstacle in enumerate(cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, infty)
    

    top_priority_set_based_tasks = ca_tasks

    output = []
    
    rtpwrapped = rtpUR5()
    for history, inference_model in zip((history_a, history_b),(inference_model_a, inference_model_b)):
        (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)

        output += [(waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)]

    (end_result, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps) = simulate([], waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history_no_ca, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=None)

    output += [(waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, num_timesteps_obstacle_actively_avoided, set_of_actively_avoided_obstacles_at_same_timestep, obstacle_highest_percentage_to_center_penetrated, union_set_of_singular_joints, num_timesteps, end_result)]

        
    return tuple(output)

def generate_and_simulate_forced_bias_pattern_near_trajectory_infer_and_compare(random, history_a, history_b, history_no_ca, inference_model_a, inference_model_b,  max_timesteps, exit_criteria, max_obstacles=5, actuators='position', record=False):

    outcome = random.randint(0,5)
    
    n_obstacles = 0
    n_obst_on_line = 0
    n_obst_near_line = 0

    if max_obstacles > 0:
        if outcome % 5 == 0:
            n_obstacles = random.randint(0, max_obstacles)
        elif outcome % 5 < 3 :
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles - n_obst_on_line)
        else:
            # Switch the bias to let n_obst_near_line also have a higher chance of rolling a higher amount of obstacles over n_obst_on_line, so n_obst_on_line has the same average as n_obst_near_line
            n_obstacles = random.randint(1, max_obstacles)
            n_obst_near_line = random.randint(0, n_obstacles)
            n_obst_on_line = random.randint(0, n_obstacles - n_obst_near_line)
        
    
    return generate_and_simulate_forced_bias_near_trajectory_infer_and_compare(random,
                                                                               history_a, history_b, history_no_ca,
                                                                               inference_model_a, inference_model_b,
                                                                               max_timesteps, exit_criteria,
                                                                               n_obstacles, n_obst_on_line, n_obst_near_line,
                                                                               actuators=actuators,
                                                                               record=record)
