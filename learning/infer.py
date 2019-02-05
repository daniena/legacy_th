import session
from session import CAI_random
from robotics.task import ca_task
from robotics.rtpwrapper import rtpUR5
from simulation.simulation import generate_and_simulate_forced_bias_pattern_near_trajectory_infer_and_compare, exit_criteria_at_end_waypoint_or_i_max, simulate
from simulation.parameters import circle_of_acceptance as sim_coa
from simulation.parameters import max_obstacles
from simulation.workspace import cartesian_sphere
from learning.datagen import pandas_episode_trajectory_initialize
from learning.models import thesis_load_model, load_model_and_history_from_latest_checkpoint
from random import Random
from keras.models import load_model
from learning.rawdata import make_obstacle_slots, CAVIKAUGee_sphere_num_inputs, CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO
from learning.dataset import normalize_parameters, normalize_assuming_given_descriptors, denormalize_assuming_given_descriptors
from learning.plot import *
from util.matrix import vector
from util.file_operations import *
from session import CAI_args
import pandas as pd
import numpy as np
import os
import re
import copy
from learning.rawdata import CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO

from simulation.parameters import timestep as sim_timestep
from simulation.parameters import circle_of_acceptance as sim_coa

class Inference_Model():
    def __init__(self, name, keras_model, datapath):
        self.name = name
        self.keras_model = keras_model
        self.datapath = datapath
        self.step_input_parser_func = None
        self.step_output_parser_func = None
        self.input_normalization_params = None
        self.output_normalization_params = None
        self.set_parser_func()
        self.set_normalization_params()

    def CAVIKAUGee_slots_input_parser(self, q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles):
        obstacle_slots = make_obstacle_slots(obstacles_buffer, max_obstacles)
        input_tensor = np.reshape(np.array(q + ee_d + obstacle_slots), (1,29))
        normalized_input_tensor = normalize_assuming_given_descriptors(input_tensor, *self.input_normalization_params)
        return normalized_input_tensor

    def CAVIKAUGee_sphere_input_parser(self, q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles):
        obstacle_slots = make_obstacle_slots(obstacles_buffer, max_obstacles)
        slotted_input_tensor = np.reshape(np.array(q + ee_d + obstacle_slots), (1,29))
        
        normalized_spherical_input_tensor = CAVIKAUGee_sphere_input_from_CAVIKee_slots_IO(slotted_input_tensor[0], max_obstacles, self.input_normalization_params[0][0:9], self.input_normalization_params[1][0:9])
        return np.reshape(normalized_spherical_input_tensor, (1,CAVIKAUGee_sphere_num_inputs))

    def CAVIKAUGee_no_obst_input_parser(self, q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles):
        input_tensor = np.reshape(np.array(q + ee_d), (1,9))
        normalized_input_tensor = normalize_assuming_given_descriptors(input_tensor, self.input_normalization_params[0][:9], self.input_normalization_params[1][:9])
        return normalized_input_tensor

    def CAVIKAUGee_output_parser_q_dot_ref(self, normalized_output_tensor):

        output_tensor = denormalize_assuming_given_descriptors(normalized_output_tensor, *self.output_normalization_params)
        
        q_dot_ref = output_tensor[0,9:15] #q_dot_ref
        #q_dot_ref = output_tensor[0,3:9] #f1
        return q_dot_ref

    def VIK_input_parser(self, q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles):
        input_tensor = np.reshape(np.array(q + ee + ee_d), (1,12))
        # There was no normalization for VIK in early trials
        return input_tensor

    def VIK_output_parser(self, output_tensor):
        # There was no normalization for VIK in early trials, and its output was simply f1, i.e. qdotref assuming no obstacles!
        return output_tensor

    def set_normalization_params(self):

        (_, _, datasetnames, _, _, datasetpaths, _, _, _) = CAI_args(self.datapath)

        datasetpath = ''
        datasetname = ''
        datasettype = ''
        if self.name[0:10] == 'CAVIKAUGee' or self.name[0:3] == 'VIK':
            datasetpath = datasetpaths[2]
            datasetname = datasetnames[2]
        else:
            print('Inference_Model not yet supported')
            exit(1)
        
        found_params_json = False
        for filename in os.listdir(datasetpath):
            if filename.find('_training_normalization_params') > -1:
                params = np.load(datasetpath + '/' + datasetname + '_training_normalization_params.npy')
                self.input_normalization_params = params[0]
                self.output_normalization_params = params[1]
                
                found_params_json = True
                break
                
        if not found_params_json:
            print('Could not find normalization parameters, loading, calculating, and storing them for later expedience.')
            input_nor_params = normalize_parameters(np.load(datasetpath + datasetname + '_training_inputs.npy'))
            output_nor_params = normalize_parameters(np.load(datasetpath + datasetname + '_training_outputs.npy'))
            
            np.save(datasetpath + '/' + datasetname + '_training_normalization_params', (input_nor_params, output_nor_params))

            self.input_normalization_params = input_nor_params
            self.output_normalization_params = output_nor_params

    def set_parser_func(self):
        if self.name[0:15] == 'CAVIKAUGee_slot':
            self.step_input_parser_func = self.CAVIKAUGee_slots_input_parser
            self.step_output_parser_func = self.CAVIKAUGee_output_parser_q_dot_ref
        elif self.name[0:17] == 'CAVIKAUGee_sphere':
            self.step_input_parser_func = self.CAVIKAUGee_sphere_input_parser
            self.step_output_parser_func = self.CAVIKAUGee_output_parser_q_dot_ref
        if self.name[0:18] == 'CAVIKAUGee_no_obst':
            self.step_input_parser_func = self.CAVIKAUGee_no_obst_input_parser
            self.step_output_parser_func = self.CAVIKAUGee_output_parser_q_dot_ref
        if self.name[0:3] == 'VIK':
            self.step_input_parser_func = self.VIK_input_parser
            self.step_output_parser_func = self.VIK_output_parser
        
    def simulation_input_to_output(self, q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles):
        input_to_model = self.step_input_parser_func(q, ee, ee_d, a, a_d, obstacles_buffer, max_obstacles)
        output_from_model = self.keras_model.predict(input_to_model)
        q_dot_ref = vector(self.step_output_parser_func(output_from_model))
        q_dot_ref = vector(np.squeeze(q_dot_ref))
        return q_dot_ref
    
class Telemetry():
    def __init__(self, num_episodes, max_timesteps, max_obstacles, seed, plotpath, modelname):
        self.telemetryname = 'seed' + str(seed) + '_' + modelname + '_nume' + str(num_episodes) + '_numt' + str(max_timesteps)
        self.telemetrypath = plotpath + '/' + self.telemetryname 
        self.episode_index = 0
        self.max_timesteps = max_timesteps
        self.max_obstacles = max_obstacles
        self.inference_episodes_numsteps = [ 0 for _ in range(num_episodes) ]
        self.solution_episodes_numsteps = [ 0 for _ in range(num_episodes) ]

        self.avg_inf_distance_to_ee_d = np.zeros(max_timesteps)
        self.avg_sol_distance_to_ee_d = np.zeros(max_timesteps)
        self.avg_inf_tracking_error_against_solution = np.zeros(max_timesteps)

        self.obst_metrics_df = pd.DataFrame(index=range(max_obstacles), columns=[ col for i in range(num_episodes) for col in ['ep' + str(i) + '_sol_actively_avoided_with_success', 'ep' + str(i) + '_inf_actively_avoided_with_success', 'ep' + str(i) + '_sol_failed_avoidance', 'ep' + str(i) + '_sol_failed_avoidance'] ])
        self.task_performance_df = pd.DataFrame(index=range(num_episodes), columns=['inference_converged', 'solution_converged', 'both_converged', 'inf_converged_but_sol_did_not', 'sol_converged_but_inf_did_not', 'inf_actively_avoided_with_success', 'sol_actively_avoided_with_success', 'both_actively_avoided_with_success', 'inf_actively_avoided_when_sol_failed', 'sol_actively_avoided_when_inf_failed', 'inf_failed_avoidance', 'sol_failed_avoidance', 'both_failed_avoidance', 'inf_failed_avoidance_but_sol_did_not', 'sol_failed_avoidance_but_inf_did_not'])

        self.environments_df = pd.DataFrame(index=range(num_episodes), columns = ['q1_0', 'q2_0', 'q3_0', 'q4_0', 'q5_0', 'q6_0', 'x_0', 'y_0', 'z_0', 'x_d', 'y_d', 'z_d', 'x_o1', 'y_o1', 'z_o1', 'r_o1'] + [e for i in range(max_obstacles) for e in ['x_o' + str(i), 'y_o' + str(i), 'z_o' + str(i), 'r_o' + str(i)]])
        
        #self.df = pd.DataFrame(index=np.arange(0, max_timesteps*sim_timestep, sim_timestep), columns = ['avg_inf_distance_to_ee # Todo: Gathering of telemetry should be put in a dataframe and saved so that the information is fully available after the program is finished running 

        set_plotpath(self.telemetrypath)

    def get_timeline(self, numsteps):
        return np.arange(0, numsteps*sim_timestep, sim_timestep)
    def remove_nan_from_df(self, df):
        df.replace(["NaN", 'NaT'], np.nan, inplace = True)
        df = df.dropna(axis=0, how='all')
        return df
    def append_shortest_list_with_its_last_element_until_matching_length(self, list_a, list_b):
        largest_length = len(list_a) if len(list_a) > len(list_b) else len(list_b)
        list_a += [ list_a[-1] for _ in range(largest_length - len(list_a)) ]
        list_b += [ list_b[-1] for _ in range(largest_length - len(list_b)) ]
        return list_a, list_b
    
    def task_distance(self, ee, sigma_0, sigma_min):
        return [np.linalg.norm(ee_timestep - np.transpose(sigma_0)) - sigma_min for ee_timestep in ee]
    def task_inference_error(self, ee_infer, ee_solution, sigma_0, sigma_min):
        task_distance_infer = self.task_distance(ee_infer, sigma_0, sigma_min)
        task_distance_solution = self.task_distance(ee_solution, sigma_0, sigma_min)
        self.append_shortest_list_with_its_last_element_until_matching_length(task_distance_infer, task_distance_solution)
        return [ sol_d - inf_d for inf_d, sol_d in zip(task_distance_infer, task_distance_solution) ]

    def did_converge(self, ee, ee_d):
        return 1 if np.linalg.norm(ee[-1,:] - np.transpose(ee_d)) < sim_coa + 0.001 else 0
    def gather(self, history_inference, history_solution, history_no_ca, ee_0, ee_d, q_0, cartesian_obstacles, alternate_episode_index_name=None):
        if alternate_episode_index_name is None:
            alternate_episode_index_name = self.episode_index
        
        history_inference = self.remove_nan_from_df(history_inference)
        history_solution = self.remove_nan_from_df(history_solution)
        
        ee_inf = history_inference.values[:,6:9]
        ee_sol = history_solution.values[:,6:9]
        inf_numsteps = ee_inf.shape[0]
        sol_numsteps = ee_sol.shape[0]
        most_timesteps = inf_numsteps if inf_numsteps > sol_numsteps else sol_numsteps
        
        inf_distance_to_ee_d = self.task_distance(ee_inf, ee_d, 0)
        sol_distance_to_ee_d = self.task_distance(ee_sol, ee_d, 0)
        inf_tracking_error_against_solution = self.task_inference_error(ee_inf, ee_sol, ee_d, 0)

        solution_obstacle_border_distance = np.zeros((self.max_obstacles, sol_numsteps))
        inference_obstacle_border_distance = np.zeros((self.max_obstacles, inf_numsteps))
        sol_actively_avoided_with_success = np.zeros(self.max_obstacles)
        inf_actively_avoided_with_success = np.zeros(self.max_obstacles)
        sol_failed_avoidance = np.zeros(self.max_obstacles)
        inf_failed_avoidance = np.zeros(self.max_obstacles)

        for obst_i, obstacle in enumerate(cartesian_obstacles):
            solution_obstacle_border_distance[obst_i,:] = self.task_distance(ee_sol, vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius)
            inference_obstacle_border_distance[obst_i,:] = self.task_distance(ee_inf, vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius)

            sol_actively_avoided_with_success[obst_i] = 1 if all( border_distance > 0 for border_distance in solution_obstacle_border_distance[obst_i,:] ) and any( border_distance < sim_coa for border_distance in solution_obstacle_border_distance[obst_i,:] ) else 0
            inf_actively_avoided_with_success[obst_i] = 1 if all( border_distance > 0 for border_distance in inference_obstacle_border_distance[obst_i,:] ) and any( border_distance < sim_coa for border_distance in inference_obstacle_border_distance[obst_i,:] ) else 0

            sol_failed_avoidance[obst_i] = 1 if any( border_distance < 0 for border_distance in solution_obstacle_border_distance[obst_i,:] ) else 0
            inf_failed_avoidance[obst_i] = 1 if any( border_distance < 0 for border_distance in inference_obstacle_border_distance[obst_i,:] ) else 0

        inference_converged = self.did_converge(ee_inf, ee_d)
        solution_converged = self.did_converge(ee_sol, ee_d)
        both_converged = solution_converged*inference_converged
        inf_converged_but_sol_did_not = inference_converged*(1-solution_converged)
        sol_converged_but_inf_did_not = solution_converged*(1-inference_converged)
        both_actively_avoided_with_success = sum( sol_actively_avoided_with_success*inf_actively_avoided_with_success )
        inf_actively_avoided_when_sol_failed = sum( inf_actively_avoided_with_success*sol_failed_avoidance )
        sol_actively_avoided_when_inf_failed = sum( sol_actively_avoided_with_success*inf_failed_avoidance )
        both_failed_avoidance = sum( sol_failed_avoidance*inf_failed_avoidance )
        inf_failed_avoidance_but_sol_did_not = sum( inf_failed_avoidance*(1-sol_failed_avoidance) )
        sol_failed_avoidance_but_inf_did_not = sum( sol_failed_avoidance*(1-inf_failed_avoidance) )

        self.task_performance_df.at[self.episode_index, 'inference_converged'] = inference_converged
        self.task_performance_df.at[self.episode_index, 'solution_converged'] = solution_converged
        self.task_performance_df.at[self.episode_index, 'both_converged'] = both_converged
        self.task_performance_df.at[self.episode_index, 'inf_converged_but_sol_did_not'] = inf_converged_but_sol_did_not
        self.task_performance_df.at[self.episode_index, 'sol_converged_but_inf_did_not'] = sol_converged_but_inf_did_not
        self.task_performance_df.at[self.episode_index, 'sol_actively_avoided_with_success'] = sum(sol_actively_avoided_with_success)
        self.task_performance_df.at[self.episode_index, 'inf_actively_avoided_with_success'] = sum(inf_actively_avoided_with_success)
        self.task_performance_df.at[self.episode_index, 'both_actively_avoided_with_success'] = both_actively_avoided_with_success
        self.task_performance_df.at[self.episode_index, 'sol_actively_avoided_when_inf_failed'] = sol_actively_avoided_when_inf_failed
        self.task_performance_df.at[self.episode_index, 'inf_actively_avoided_when_sol_failed'] = inf_actively_avoided_when_sol_failed
        self.task_performance_df.at[self.episode_index, 'sol_failed_avoidance'] = sum(sol_failed_avoidance)
        self.task_performance_df.at[self.episode_index, 'inf_failed_avoidance'] = sum(inf_failed_avoidance)
        self.task_performance_df.at[self.episode_index, 'both_failed_avoidance'] = both_failed_avoidance
        self.task_performance_df.at[self.episode_index, 'sol_failed_avoidance_but_inf_did_not'] = sol_failed_avoidance_but_inf_did_not
        self.task_performance_df.at[self.episode_index, 'inf_failed_avoidance_but_sol_did_not'] = inf_failed_avoidance_but_sol_did_not
        
        self.obst_metrics_df.ix[:,self.episode_index*4] = sol_actively_avoided_with_success
        self.obst_metrics_df.ix[:,self.episode_index*4+1] = inf_actively_avoided_with_success
        self.obst_metrics_df.ix[:,self.episode_index*4+2] = sol_failed_avoidance
        self.obst_metrics_df.ix[:,self.episode_index*4+3] = inf_failed_avoidance

        self.environments_df.ix[self.episode_index,0:6] = q_0[:,0]
        self.environments_df.ix[self.episode_index,6:9] = ee_0[:,0]
        self.environments_df.ix[self.episode_index,9:12] = ee_d[:,0]
        for obst_i, obstacle in enumerate(cartesian_obstacles):
            self.environments_df.ix[self.episode_index,12+obst_i*4:12+(1+obst_i)*4] = obstacle

        # Count averages if the solution converged (if not, it is likely singular or stuck)
        #if solution_converged: # If not, then its num timesteps will be zero, and it will not have added anything to the averages, i.e. the remaining code still correctly calculates the averages
        self.inference_episodes_numsteps[self.episode_index] = inf_numsteps
        self.solution_episodes_numsteps[self.episode_index] = sol_numsteps
        
        self.avg_inf_distance_to_ee_d[:inf_numsteps] += np.array(inf_distance_to_ee_d)
        self.avg_sol_distance_to_ee_d[:sol_numsteps] += np.array(sol_distance_to_ee_d)
        self.avg_inf_tracking_error_against_solution[:most_timesteps] += np.array(inf_tracking_error_against_solution)

        # Plots
        fig, _ = plot_tracking_compare_models([history_inference], ['Error'], ee_d) # ToDo: Function name, etc
        save_plot(fig, self.telemetrypath + '/00inference_tracking_distance', 'episode_' + str(alternate_episode_index_name))

        fig, _ = plot_tracking_and_ca(history_inference, ee_d, cartesian_obstacles)
        save_plot(fig, self.telemetrypath + '/01inference_tracking_and_border_distance', 'episode_' + str(alternate_episode_index_name))

        fig, _ = plot_tracking_and_ca(history_solution, ee_d, cartesian_obstacles)
        save_plot(fig, self.telemetrypath + '/02solution_tracking_and_border_distance', 'episode_' + str(alternate_episode_index_name))

        fig, _ = plot_tracking_compare(history_inference, history_solution, ee_d)
        save_plot(fig, self.telemetrypath + '/03tracking_comparison', 'episode_' + str(alternate_episode_index_name))

        fig, _ = plot_ca_compare([history_solution, history_inference, history_no_ca], cartesian_obstacles)
        save_plot(fig, self.telemetrypath + '/04border_distance_comparison', 'episode_' + str(alternate_episode_index_name))   

        # Remember episode number
        self.episode_index += 1

    def end_and_save(self):
        
        inf_average_divide = np.zeros(self.max_timesteps)
        sol_average_divide = np.zeros(self.max_timesteps)
        compare_average_divide = np.zeros(self.max_timesteps)

        for i in range(self.max_timesteps):
            inf_average_divide[i] += sum([ 1 if numsteps > i else 0 for numsteps in self.inference_episodes_numsteps])
            sol_average_divide[i] += sum([ 1 if numsteps > i else 0 for numsteps in self.solution_episodes_numsteps])
            compare_average_divide[i] = inf_average_divide[i] if inf_average_divide[i] > sol_average_divide[i] else sol_average_divide[i]

            if inf_average_divide[i] == 0: inf_average_divide[i] = 1
            if sol_average_divide[i] == 0: sol_average_divide[i] = 1
            if compare_average_divide[i] == 0: compare_average_divide[i] = 1
            
        self.avg_inf_distance_to_ee_d /= inf_average_divide
        self.avg_sol_distance_to_ee_d /= sol_average_divide
        self.avg_inf_tracking_error_against_solution /= compare_average_divide
        timesteps = self.get_timeline(self.max_timesteps)
        
        plot(timesteps, self.avg_inf_distance_to_ee_d, marker='x', init=True) # init should not need to be set to the user. It should be enough to call close=True.
        plot(timesteps, self.avg_sol_distance_to_ee_d, marker='+')
        plot(timesteps, self.avg_inf_tracking_error_against_solution, marker='1', legend=['average distance to desired end effector position', 'solution average distance to desired end effector position', 'average error from solution'], plotfilename='avg_ee_error_compared_to_solution', plotpath=self.telemetrypath, save=True, close=True)

        metricspath = self.telemetrypath + '/metrics'
        make_path(metricspath)
        self.obst_metrics_df.to_csv(metricspath + '/obst_metrics.csv')
        self.task_performance_df.to_csv(metricspath + '/task_performance.csv')
        self.environments_df.to_csv(metricspath + '/environments.csv')






        
def get_episode_better(path, episode_num):
    episode_summaries = pd.read_csv(path + '/episodes_summaries.csv', index_col=[0,1])
    episode_summary = episode_summaries.xs('episode' + str(episode_num) + '.csv', level='filename')

    initial_config = [episode_summary.loc['waypoint_0']['q1'],
                      episode_summary.loc['waypoint_0']['q2'],
                      episode_summary.loc['waypoint_0']['q3'],
                      episode_summary.loc['waypoint_0']['q4'],
                      episode_summary.loc['waypoint_0']['q5'],
                      episode_summary.loc['waypoint_0']['q6']]
    initial_position = [episode_summary.loc['waypoint_0']['x'],
                        episode_summary.loc['waypoint_0']['y'],
                        episode_summary.loc['waypoint_0']['z']]
    desired_position = [episode_summary.loc['waypoint_1']['x'],
                        episode_summary.loc['waypoint_1']['y'],
                        episode_summary.loc['waypoint_1']['z']]

    obstacles = episode_summary[episode_summary.index.str.contains('obstacle_')]
    cartesian_obstacles = [ cartesian_sphere(obstacles.loc[obstacle_name]['x'],
                                   obstacles.loc[obstacle_name]['y'],
                                   obstacles.loc[obstacle_name]['z'],
                                   obstacles.loc[obstacle_name]['radius']) for obstacle_name in obstacles.index]

    print(initial_config)
    return (vector(initial_config), 0), (vector(initial_position), vector(desired_position)), cartesian_obstacles
        
def observe(seed, datapath, modelpath, modelnames, modelinitials, num_episodes, max_timesteps, max_obstacles, from_episode_num=None, from_latest_checkpoint=True, from_checkpoint_num=None, actuators='position', verbose=False):

    (random, randomstate, seed) = CAI_random(seed)

    if from_episode_num is not None:
        num_episodes=1

    plotpath = datapath + '/sessions/CAI/plots'
                
    if not modelpath.endswith('/'):
        modelpath = modelpath + '/'

    inference_models = []
    for modelname in modelnames:
        model = None
        if from_latest_checkpoint:
            model, _ = load_model_and_history_from_latest_checkpoint(random, modelpath, modelname)
        else:
            model = thesis_load_model(random, modelpath, modelname + '_checkpoint_' + str(from_checkpoint_num))

        model.summary()

        inference_models += [Inference_Model(modelname, model, datapath)]

    simulation_histories = [ [] for _ in range(num_episodes)]
    simulation_waypoints = [ [] for _ in range(num_episodes)]

    model_histories = [ [] for _ in modelinitials]

    telemetries = []
    for modelinitial in modelinitials:
        telemetries += [Telemetry(num_episodes, max_timesteps, max_obstacles, seed, plotpath, modelinitial)]

    rtpwrapped = rtpUR5()
    for i in range(num_episodes):
        if from_episode_num is not None:
            i = from_episode_num

        history_infer = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
        history_solution = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
        history_no_ca = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)
        
        if from_episode_num is None:
            ((waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles, _, _, _, _, _, _), _, _) = generate_and_simulate_forced_bias_pattern_near_trajectory_infer_and_compare(random, history_infer, history_solution, history_no_ca, inference_models[0], None, max_timesteps, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, max_obstacles=max_obstacles, actuators='perfect_position', record=True)
            ca_tasks = [ 0 for _ in cartesian_obstacles ]
            for obst_i, obstacle in enumerate(cartesian_obstacles):
                ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, np.infty)
                
            top_priority_set_based_tasks = ca_tasks
        else:
            waypoint_configurations, waypoint_cartesian_positions, cartesian_obstacles = get_episode_better(datapath + '/rawdata/ca_trackpos/', from_episode_num)
            ca_tasks = [ 0 for _ in cartesian_obstacles ]
            for obst_i, obstacle in enumerate(cartesian_obstacles):
                ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, np.infty)
                
            top_priority_set_based_tasks = ca_tasks
            simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history_infer, actuators=actuators, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, random=random, record=True, inference_model=inference_models[0])
            simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history_solution, actuators=actuators, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, random=random, record=True, inference_model=None)
            simulate([], waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history_no_ca, actuators=actuators, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, random=random, record=True, inference_model=None)
            
        if from_episode_num is None:    
            telemetries[0].gather(history_infer, history_solution, history_no_ca, waypoint_cartesian_positions[0], waypoint_cartesian_positions[1], waypoint_configurations[0], cartesian_obstacles)
        else:
            telemetries[0].gather(history_infer, history_solution, history_no_ca, waypoint_cartesian_positions[0], waypoint_cartesian_positions[1], waypoint_configurations[0], cartesian_obstacles, alternate_episode_index_name=from_episode_num)

        model_histories[0] = copy.deepcopy(history_infer)
        model_histories_index = 1
        for telemetry, inference_model in zip(telemetries[1:], inference_models[1:]):
            history_infer = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)

            simulate(top_priority_set_based_tasks, waypoint_configurations, waypoint_cartesian_positions, rtpwrapped, max_timesteps, history_infer, actuators=actuators, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, random=random, record=True, inference_model=inference_model)

            if from_episode_num is None:
                telemetry.gather(history_infer, history_solution, history_no_ca, waypoint_cartesian_positions[0], waypoint_cartesian_positions[1], waypoint_configurations[0], cartesian_obstacles)
            else:
                telemetry.gather(history_infer, history_solution, history_no_ca, waypoint_cartesian_positions[0], waypoint_cartesian_positions[1], waypoint_configurations[0], cartesian_obstacles, alternate_episode_index_name=from_episode_num)

            model_histories[model_histories_index] = copy.deepcopy(history_infer)
            model_histories_index += 1

        fig, _ = plot_ca_compare(model_histories, cartesian_obstacles)
        save_plot(fig, plotpath + '/seed' + str(seed) + '_nume' + str(num_episodes) + '_numt' + str(max_timesteps) + '_border_distance_model_comparisons', 'episode_' + str(i))

        fig, _ = plot_tracking_compare_models(model_histories, ('sphere', 'slot', 'control'), waypoint_cartesian_positions[1], markevery=1000)
        save_plot(fig, plotpath + '/seed' + str(seed) + '_nume' + str(num_episodes) + '_numt' + str(max_timesteps) + '_position_error_model_comparisons', 'episode_' + str(i))

    for telemetry in telemetries:
        telemetry.end_and_save()

def get_episode(telemetrypath, episode_num):
    environment_df = pd.read_csv(telemetrypath + '/metrics/environments.csv')
    environment_df.replace(["NaN", 'NaT', 'infty'], np.nan, inplace = True)
    q_0 = environment_df.values[episode_num, 1:7]
    ee_0 = environment_df.values[episode_num, 7:10]
    ee_d = environment_df.values[episode_num, 10:13]
    cartesian_obstacles = []
    for i in range(max_obstacles):
        center =  environment_df.values[episode_num, 13 + i*4: 13 + (i+1)*4 - 1]
        r = environment_df.values[episode_num, 13 + (i+1)*4 - 1]

        illegal = False
        for e in center + [r]:
            if np.isnan(e):
                illegal = True
                break
        if illegal:
            continue

        print(center)
        print(r)
        cartesian_obstacles += [cartesian_sphere(center[0], center[1], center[2], r)]
        print(cartesian_obstacles)

    return vector(q_0), (vector(ee_0), vector(ee_d)), cartesian_obstacles

def observe_episode(seed, datapath, modelpath, modelname, episode_configuration_init, episode_waypoint_cartesian_positions, episode_cartesian_obstacles, max_timesteps, from_latest_checkpoint=True, exit_criteria=exit_criteria_at_end_waypoint_or_i_max, max_obstacles=max_obstacles, actuators='perfect_position', record=True):

    (random, randomstate, seed) = CAI_random(seed)
    
    if not modelpath.endswith('/'):
        modelpath = modelpath + '/'

    model = None
    if from_latest_checkpoint:
        model, _ = load_model_and_history_from_latest_checkpoint(random, modelpath, modelname)
    else:
        model, _ = load_model_and_history(random, modelpath + modelname + '.h5')

    model.summary()

    inference_model = Inference_Model(modelname, model, datapath)
    history = pandas_episode_trajectory_initialize(max_timesteps, max_obstacles)

    ca_tasks = [ 0 for _ in episode_cartesian_obstacles ]
    for obst_i, obstacle in enumerate(episode_cartesian_obstacles):
        ca_tasks[obst_i] = ca_task(vector([0,0,0]), vector([obstacle.center_x, obstacle.center_y, obstacle.center_z]), obstacle.radius, np.infty)

    top_priority_set_based_tasks = ca_tasks

    rtpwrapped = rtpUR5()
    simulate(top_priority_set_based_tasks, (episode_configuration_init, 0), episode_waypoint_cartesian_positions, rtpwrapped, max_timesteps, history, actuators=actuators, exit_criteria=exit_criteria, random=random, record=record, inference_model=inference_model)
    




def history_infer_distance_from_desired_position(history_infer, waypoint_cartesian_position):
    return norm(history_infer[:, 6:9].values, np.repeat(list(waypoint_cartesian_position[0:3,0]), len(history_infer), axis=0))
