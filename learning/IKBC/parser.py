import pandas as pd
import numpy as np
from param_debug import debug

def parse_dataset(datapath, filenames):

    if not datapath.endswith('/'):
        datapath = datapath + '/'

    summaries = pd.read_csv(datapath + 'episodes_summaries.csv', index_col=[0,1])

    inputs = ['q', 'p', 'p_des']
    outputs = ['f1']

    numsteps_episode = [ int(summaries.at[(filename, 'num_timesteps'), 'extra']) for filename in filenames ]
    numsteps_total = sum(numsteps_episode)
    
    print('Number of steps:', numsteps_total)

    inputs = np.zeros((numsteps_total, 6+3+3))
    outputs = np.zeros((numsteps_total, 6))

    laststep = 0
    for i, filename in enumerate(filenames):
        episode = pd.read_csv(datapath + 'episodes/' + filename, index_col=0).values
        if debug:
            print(filename), print(len(episode)), print(len(np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0))), print(summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values)
        inputs[laststep:laststep + numsteps_episode[i], :] = np.hstack((episode[:,0:9], np.repeat([summaries.loc[(filename, 'waypoint_1')].iloc[6:9].values], numsteps_episode[i], axis=0)))
        outputs[laststep:laststep + numsteps_episode[i], :] = episode[:, 9:15]
        laststep = laststep + numsteps_episode[i]

    return (np.array(inputs).tolist(), np.array(outputs).tolist())
