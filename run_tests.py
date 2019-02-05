import sys
import session

import test.t_util as t_util
import test.t_robotics as t_robotics
import test.t_simulation as t_simulation
import test.t_learning as t_learning
import test.t_run as t_run

from param_debug import exhaustive

def run_all_tests():
    print('Warning, safe_q_ranges use not implemented yet')
    
    if not exhaustive:
        print("Warning, test not exhaustive. Use 'exhaustive=False' only for testing tasks that have low computational intensity.")

    # util
    t_util.test_remove_files_with_ending()
    t_util.test_make_path()
    
    # robotics
    t_util.test_matrix_double_equality()
    t_util.test_multidot()
    t_util.test_dagger()
    t_util.test_axis_angle_rotation_matrix()
    t_robotics.test_forward_kinematic_field_of_view()
    t_robotics.test_Jpos()
    t_robotics.test_Jfov()
    t_robotics.test_inverse_kinematic_priority_set_based_tasks_q_dot_ref()
    
    # simulation
    t_simulation.test_generate_random_workspace()
    t_simulation.test_populate_random_waypoints_within_legal_workspace()
    t_simulation.test_point_on_cylinder_around_line()
    print(exhaustive)
    if exhaustive:
        t_simulation.test_simulate_perfect_position_control()
        t_simulation.test_simulate_position_control_static_reference()
        t_simulation.test_simulate_position_control()
        t_simulation.test_simulate_velocity_control()

    # learning
    t_learning.test_pandas_episode_summary()
    #t_learning.test_VIK1_session_self_equality()
    #t_learning.test_VIK1_session()

    # run
    t_run.test_CAI_session()
    
if __name__ == '__main__':

    run_all_tests()
