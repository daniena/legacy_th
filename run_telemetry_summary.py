import pandas as pd
import os
from util.file_operations import load_json


def telemetry_model_summary(subpath):
    plotpath = os.getcwd() + '/data/sessions/CAI/plots'
    
    #subpath = 'seed101_CAVIKAUGee_sphere_nume10_numt30000'
    subpath = 'seed102_CAVIKAUGee_sphere_nume30_numt1000'

    metricspath = plotpath + '/' + subpath + '/' + 'metrics'

    df = pd.read_csv(metricspath + '/' + 'task_performance.csv')
    print('times solution converged:', sum(df.loc[:,'solution_converged']),
          '\ntimes model converged:', sum(df.loc[:,'inference_converged']),
          '\ntimes both converged:', sum(df.loc[:,'both_converged']),
          '\ntimes solution converged but model did not:', sum(df.loc[:,'sol_converged_but_inf_did_not']),
          '\ntimes model converged but solution did not:', sum(df.loc[:,'inf_converged_but_sol_did_not']),
          '\ntimes solution actively avoided with success:', sum(df.loc[:,'sol_actively_avoided_with_success']),
          '\ntimes model actively avoided with success:', sum(df.loc[:,'inf_actively_avoided_with_success']),
          '\ntimes both actively avoided with success:', sum(df.loc[:,'both_actively_avoided_with_success']),
          '\ntimes solution actively avoided with success when the model failed:', sum(df.loc[:,'sol_actively_avoided_when_inf_failed']),
          '\ntimes model actively avoided with success when the solution failed:', sum(df.loc[:,'inf_actively_avoided_when_sol_failed']),
          '\ntimes model failed avoidance:', sum(df.loc[:,'inf_failed_avoidance']),
          '\ntimes solution failed avoidance:', sum(df.loc[:,'sol_failed_avoidance']),
          '\ntimes both failed avoidance:', sum(df.loc[:,'both_failed_avoidance']),
          '\ntimes solution failed avoidance but model did not:', sum(df.loc[:,'sol_failed_avoidance_but_inf_did_not']),
          '\ntimes model failed avoidance but solution did not:', sum(df.loc[:,'inf_failed_avoidance_but_sol_did_not']))

def make_telemetry_summary():
    subpaths = ['seed102_CAVIKAUGee_sphere_nume30_numt1000',
                'seed102_CAVIKAUGee_slot_nume30_numt1000',
                'seed102_CAVIKAUGee_no_obst_nume30_numt1000']

    for subpath in subpaths:
        print('Summarizing from subpath: ' + subpath)
        telemetry_model_summary(subpath)


if __name__ == '__main__':
    make_telemetry_summary()
