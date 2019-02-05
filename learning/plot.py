from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
from util.matrix import vector
from util.file_operations import make_path
from learning.models import load_model_and_history_from_latest_checkpoint
import os
import re

from simulation.parameters import timestep as sim_timestep
from simulation.parameters import circle_of_acceptance as sim_coa

norm = (0.0892 + 0.425 + 0.392 + 0.09475)/100# The fully extended length of the UR5 = d1 + a2 + a3 + d5 = 1.00095 done with the implemented UR5 DH parameters similarly to figure 2 in https://ieeexplore.ieee.org/document/7844896

def set_plotpath(plotpath):
    set_plotpath.plotpath = plotpath
def get_plotpath():
    return set_plotpath.plotpath

def _phold(x=None, y=None, marker=None, legend=None, title=None, plotfilename=None, fig=None, ax=None):
    _phold.x = x
    _phold.y = y
    _phold.marker = marker
    _phold.legend = legend
    _phold.title = title
    _phold.plotfilename = plotfilename
    _phold.fig = fig
    _phold.ax = ax

def _psettings(show=False, init=False, save=False, close=False, plotpath=None):
    _psettings.show = show
    _psettings.init = init
    _psettings.save = save
    _psettings.close = close
    if plotpath is not None:
        set_plotpath(plotpath)

def _color(c):
    _color.c = c

def _plot_init(fig, ax, init=True):
    if init:
        plt.gcf().clear()
        _phold() # Sets all _phold. variables to default values
        _psettings() 
        
        _color.c = None
        
    if init:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        _phold.fig = fig
        _phold.ax = ax
        
    return fig, ax

def _plot_close():
    plt.close(_phold.fig)
    _phold()
    _psettings()

def save_plot(fig, plotpath, plotfilename):
    make_path(plotpath)
    if not plotpath[-1] == '/':
        plotpath += '/'
        
    fig.savefig(plotpath + plotfilename + '.eps', bbox_inches="tight", pad_inches=0) # removes whitespace around plot

def _plot():

    try:
        if _phold.marker is None:
            _phold.ax.plot(_phold.x, _phold.y)
        else:
            _phold.ax.plot(_phold.x, _phold.y, marker=_phold.marker, markevery=100)
            print('works')
    except Exception as e:
        print(e)
        if _phold.marker is None:
            _phold.ax.plot(_phold.x[:len(_phold.x)-1], _phold.y)
        else:
            _phold.ax.plot(_phold.x[:len(_phold.x)-1], _phold.y, marker=_phold.marker, markevery=100)
            print('works')
    print(_phold.marker)

    if _phold.legend:
        _phold.ax.legend(_phold.legend, loc=1)
    
    if _phold.title:
        plt.title()
    
    if _psettings.show:
        plt.show()

    if _psettings.save:
        save_plot(_phold.fig, get_plotpath(), _phold.plotfilename)

    if _psettings.close:
        _plot_close()

    return _phold.fig, _phold.ax

def plot(x, y, marker=None, legend=None, title=None, plotpath=None, plotfilename=None, show=False, init=False, save=False, close=False):
    _plot_init(None, None, init=init)
    _phold.x = x
    _phold.y = y
    if marker is not None:
        _phold.marker = marker
    if legend is not None:
        _phold.legend = legend
    if title is not None:
        _phold.title = title
    if plotfilename is not None:
        _phold.plotfilename = plotfilename
    if plotpath is not None:
        set_plotpath(plotpath)
    _psettings(show=show, init=init, save=save, close=close)
    _plot()

def plot_training_score(training_history, num_ticks, show=True, init=True, save=False, plotpath=None, plotfilename=None):
    fig, ax = _plot_init(None, None, init=init)
    #print('Availible variables to plot: {}'.format(training_history.training_history.keys()))

    
    
    epochs = len(training_history['loss'])

    # Visualize the plot, to be applied after traing is complete
    ax.plot(range(1,epochs + 1), training_history['loss'], 'o')
    ax.plot(range(1,epochs + 1), training_history['val_loss'])
    ax.legend(['Training loss', 'Validation loss'], loc=7)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(range(2,epochs + 1, int(epochs/num_ticks)))
    #plt.title('Loss of training set vs validation set')

    if show:
        plt.show()

    if save:
        save_plot(fig, plotpath, plotfilename + '_history_loss')
    
    fig, ax = _plot_init(None, None, init=init)
              
    ax.plot(range(1,epochs + 1), training_history['acc'], 'o')
    ax.plot(range(1,epochs + 1), training_history['val_acc'])
    ax.legend(['Training accuracy', 'Validation accuracy'], loc=7)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(range(2,epochs + 1, int(epochs/num_ticks)))
    #plt.title('Accuracy of training set vs validation set')

    if save:
        save_plot(fig, plotpath, plotfilename + '_history_acc')

    if show:
        plt.show()

    return fig, ax

def plot_training_scores(train_histories, val_histories, ylabel, num_ticks, legends, show=True, init=True, save=False, plotpath=None, plotfilename=None):
    fig, ax = _plot_init(None, None, init=init)

    min_epochs = min([ len(train_histories[i]) for i in range(len(train_histories)) ])

    train_markers = ['o', 'v', 'X']
    val_markers = ['d', '1', 'x']

    for ((train_history, val_history), (train_mark, val_mark)) in zip(zip(train_histories, val_histories), zip(train_markers, val_markers)):
        ax.plot(range(1,min_epochs + 1), train_history[:min_epochs], marker=train_mark, linestyle="None", fillstyle='none')
        ax.plot(range(1,min_epochs + 1), val_history[:min_epochs], marker=val_mark, linestyle="None", fillstyle='none')

    use_legends = [ l for legend  in legends for l in [legend + '_train', legend + '_val'] ]
        
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend(use_legends)
    plt.xticks(range(2,min_epochs + 1, int(min_epochs/num_ticks)))

    if save:
        save_plot(fig, plotpath, plotfilename)

    if show:
        plt.show()

    return fig, ax


def plot_training_score_cmap(training_history, num_ticks, show=True, init=True):
    if init:
        plt.gcf().init()
    #print('Availible variables to plot: {}'.format(training_history.training_history.keys()))
    
    epochs = len(training_history['loss'])
    
    # Visualize the plot, to be applied after traing is complete
    plot_cmap(plt,range(1,epochs+1), training_history['loss'], 'Blues', max(training_history['loss'] + [1]), reverse=True)
    plot_cmap(plt,range(1,epochs+1), training_history['val_loss'], 'Greens', max(training_history['val_loss'] + [1]), reverse=True)
    plt.legend(['Training loss is blue', 'Validation loss is green'])
    plt.set_xlabel('epochs')
    plt.set_ylabel('loss')
    plt.xticks(range(2,epochs + 1, int(epochs/num_ticks)))
    plt.title('Loss of training set vs validation set')
    plt.show()
    
    plot_cmap(plt,range(1,epochs+1), training_history['acc'], 'Blues', 1, reverse=True)
    plot_cmap(plt,range(1,epochs+1), training_history['val_acc'], 'Greens', 1, reverse=True)
    plt.legend(['Training accuracy is blue', 'Validation accuracy is green'])
    plt.set_xlabel('epochs')
    plt.set_ylabel('accuracy')
    plt.xticks(range(2,epochs + 1, int(epochs/num_ticks)))
    plt.title('Accuracy of training set vs validation set')
    if show:
        plt.show()

    return plt

cmaps = OrderedDict()
cmaps['Perceptually Uniform Sequential'] = ['viridis', 'plasma',
                                            'inferno', 'magma']
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
_DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
       'Sequential (2)': 1.4, 'Diverging': 1.4, 'Qualitative': 1.4,
       'Miscellaneous': 1.4}
def plot_cmap(ax, x, y, cmap, vmax, reverse=False):

    
    cmap_category = ''
    for key, value in cmaps.items():
        if cmap in value:
            cmap_category = key
    dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing

    # Get RGB values for colormap and convert the colormap in
    # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
    rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

    # Plot colormap L values.  Do separately for each category
    # so each plot can be pretty.  To make scatter markers change
    # color along plot:
    # http://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
    
    if cmap_category == 'Sequential':
        # These colormaps all start at high lightness but we want them
        # reversed to look nice in the plot, so reverse the order.
        if reverse:
            c_ = y
        else:
            c_ = y[::-1]
    else:
        if reverse:
            c_ = y[::-1]
        else:
            c_ = y

    ax.scatter(x, y, c=c_, cmap=plt.get_cmap(cmap), vmin=0, vmax=vmax)
    #https://stackoverflow.com/questions/39735147/how-to-color-matplotlib-scatterplot-using-a-continuous-value-seaborn-color

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
def continuous_colormap_plot_stub():
    #Source: https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html

    x = np.linspace(0, 3 * np.pi, 500)
    y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0])
    
    # Use a boundary norm instead
    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    fig.colorbar(line, ax=axs[1])
    
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(-1.1, 1.1)
    plt.show()






    

def plot_position_task_sigma(history, sigma_0, sigma_min, normalize=False, marker=None, markevery=100, fillstyle='full', label=None, fig=None, ax=None, show=False, init=False):
    fig, ax = _plot_init(fig, ax, init=init)

    if isinstance(history, pd.DataFrame):
        ee = history.values[:,6:9]
    else:
        ee = history
    
    y = [np.linalg.norm(ee_timestep - np.transpose(sigma_0)) - sigma_min for ee_timestep in ee]
    x = np.arange(0, len(y)*sim_timestep, sim_timestep)
    _phold(x=x, y=y)

    if normalize:
        y = [ element/norm for element in y ]

    try:
        if marker is None:
            ax.plot(x, y, label=label)
        else:
            ax.plot(x, y, marker=marker, markevery=markevery, fillstyle=fillstyle, label=label)
    except Exception as e:
        print(e)
        if marker is None:
            ax.plot(x[:len(x)-1], y, label=label)
        else:
            ax.plot(x[:len(x)-1], y, marker=marker, markevery=markevery, fillstyle=fillstyle, label=label)
    
    if show:
        plt.show()

    return fig, ax

def plot_position_task_sigma_difference(history_inference, history_solution, sigma_0, sigma_min, marker=None, fig=None, ax=None, show=False, init=False):
    fig, ax = _plot_init(fig, ax, init=init)

    if isinstance(history_inference, pd.DataFrame):
        ee_a = history_inference.values[:,6:9]
    else:
        ee_a = history_inference
    if isinstance(history_solution, pd.DataFrame):
        ee_b = history_solution.values[:ee_a.shape[0],6:9]
    else:
        ee_b = history_solution[:ee_a.shape[0],:]

    y_a = [np.linalg.norm(ee_timestep - np.transpose(sigma_0)) - sigma_min for ee_timestep in ee_a]
    y_b = [np.linalg.norm(ee_timestep - np.transpose(sigma_0)) - sigma_min for ee_timestep in ee_b]

    y = [ abs(y_bi - y_ai) for y_ai, y_bi in zip(y_a, y_b) ]
    x = np.arange(0, len(y)*sim_timestep, sim_timestep)
    _phold(x=x)

    try:
        if marker is None:
            ax.plot(x, y)
        else:
            ax.plot(x, y, marker=marker, markevery=100)
    except:
        if marker is None:
            ax.plot(x[:len(x)-1], y)
        else:
            ax.plot(x[:len(x)-1], y, marker=marker, markevery=100)
    
    if show:
        plt.show()

    return fig, ax

def plot_dashed_line_at_y(at_y, fig=None, ax=None, label=None, show=False, init=False):
    # Assumes _phold(x) has been called to hold x
    fig, ax = _plot_init(fig, ax, init=init)

    x = _phold.x
    y = [at_y for _ in x]
    ax.plot(x, y, '--', color='k', label=label)
    
    if show:
        plt.show()

    return fig, ax

def plot_tracking(history, ee_d, show=False, init=True):
    fig, ax = plot_position_task_sigma(history, ee_d, 0, init=init, marker='*')
    y_min = min(_phold.y)
    x_of_y_min = _phold.y.index(y_min)

    dashed_line_at_y = sim_coa # 0.02
    plot_dashed_line_at_y(dashed_line_at_y, fig=fig, ax=ax)
    ax.plot((x_of_y_min*sim_timestep, x_of_y_min*sim_timestep), (dashed_line_at_y, y_min), 'darkorange', marker='x')

    ax.legend(['ee distance from ee_d', 'circle of acceptance = ' + str(sim_coa), 'smallest distance from coa: ' + str(y_min - sim_coa)], loc=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [m]')
    
    if show:
        plt.show()

    return fig, ax
    
def plot_tracking_and_ca(history, ee_d, cartesian_obstacles, show=False, init=True):
    fig, ax = plot_position_task_sigma(history, ee_d, 0, init=init, marker='*')

    markers = ['v', '|', '.', '1', '4']
    
    for obstacle, marker in zip(cartesian_obstacles, markers):
        oc = vector([obstacle.center_x, obstacle.center_y, obstacle.center_z])
        r = obstacle.radius
        plot_position_task_sigma(history, oc, r, fig=fig, ax=ax, marker=marker)

    ax.legend(['|eed - ee|'] + ['|d' + str(obst_i) + '|' for obst_i in range(len(cartesian_obstacles))], loc=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [m]')
    
    at_y = 0
    plot_dashed_line_at_y(at_y, fig=fig, ax=ax)
        
    if show:
        plt.show()

    return fig, ax

def plot_tracking_compare(history_inference, history_solution, ee_d, show=False, init=True):
    fig, ax = plot_position_task_sigma(history_inference, ee_d, 0, init=init, marker='*')
    plot_position_task_sigma(history_solution, ee_d, 0, fig=fig, ax=ax, marker='|')
    plot_position_task_sigma_difference(history_inference, history_solution, ee_d, 0, fig=fig, ax=ax, marker='x')

    ax.legend(['tracking', 'solution tracking', 'tracking error'], loc=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [m]')

    if show:
        plt.show()

    return fig, ax

def plot_ca_compare(histories, cartesian_obstacles,show=False, init=True):
    fig, ax = _plot_init(None, None, init=init)

    markers = ['v', '|', '.', '1', '4']

    
    
    axs = [ plt.subplot(len(histories), 1, i+1) for i in range(len(histories)) ]

    y_inside_obstacle = [[ 0 for _ in range(history.values.shape[0]) ] for history in histories]
    y_max = [0, 0, 0]
    for obst_i, (obstacle, marker) in enumerate(zip(cartesian_obstacles, markers)):
        oc = vector([obstacle.center_x, obstacle.center_y, obstacle.center_z])
        r = obstacle.radius

        for ax_index, (history_i, ax_i) in enumerate(zip(histories, axs)):
            plot_position_task_sigma(history_i, oc, r, fig=fig, ax=ax_i, marker=marker, label='|d' + str(obst_i) + '|')
            for index, yi in enumerate(_phold.y):
                if yi <= 0:
                    y_inside_obstacle[ax_index][index] += 1
                if yi > y_max[ax_index]:
                    y_max[ax_index] = yi

    for ax_index, (y_ax, ax_i) in enumerate(zip(y_inside_obstacle, axs)):
        prev = 0
        for index, ynum_inside in enumerate(y_ax):
            if prev == 0 and ynum_inside > 0:
                ax_i.plot((index*sim_timestep, index*sim_timestep), (0.16*y_max[ax_index], -0.16*y_max[ax_index]), 'red')
            if ynum_inside == 0 and prev > 0:
                ax_i.plot((index*sim_timestep, index*sim_timestep), (0.16*y_max[ax_index], -0.16*y_max[ax_index]), 'red')
            prev = ynum_inside

    for ax_i in axs:
        plot_dashed_line_at_y(0, fig=fig, ax=ax_i)
        ax_i.legend(prop={'size': 6}, loc=1)
        ax_i.set_xlabel('Time [s]')
        ax_i.set_ylabel('Distance [m]')

    plt.subplots_adjust(hspace=0.5)
    
    if show:
        plt.show()

    return fig, ax

def plot_tracking_compare_models(histories, labels, ee_d, show=False, init=True, markevery=100):
    fig, ax = _plot_init(None, None, init=init)
    
    COA_normd = 0.02/norm
    
    error_markers = ['v', '|', '.']
    closest_to_COA_markers = ['x', '1', '2']

    for history, error_marker, closest_to_marker, label in zip(histories, error_markers, closest_to_COA_markers, labels):
        print(label)
        plot_position_task_sigma(history, ee_d, 0, normalize=True, fig=fig, ax=ax, marker=error_marker, label=label, fillstyle='none', markevery=markevery)
        y_min = min(_phold.y)
        x_of_y_min = _phold.y.index(y_min)
        ax.plot((x_of_y_min*sim_timestep, x_of_y_min*sim_timestep), (COA_normd, y_min/norm), marker=closest_to_marker, label=label + ', smallest value: ' + str(y_min/norm), fillstyle='none', markersize=12)

    plot_dashed_line_at_y(COA_normd, fig=fig, ax=ax, label='COA')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [%]')
    ax.legend(prop={'size': 6}, loc=1)

    plt.subplots_adjust(hspace=0.5)
    
    if show:
        plt.show()

    return fig, ax

def plot_all_most_recent_checkpoints(random, checkpointpath, performance_threshold_to_show=0.0, containing=None, show=False, init=True, save=True, plotpath=None):
    # Add title to plot

    models_covered = []

    commentblock = """
    print('')
    for history_csv in os.listdir(historiespath):
    modelname = history_csv.replace('_history.json', '')
    #_, history = load_model_and_history_from_latest_checkpoint(checkpointpath, modelname)
    history = load_json(historiespath, modelname + '_history')
    print (modelname)
    plot_training_score(history, 15)
    """

    #history_plot_foo = plot_training_score_cmap
    history_plot_foo = plot_training_score

    for filename in os.listdir(checkpointpath):
        if containing is not None:
            if 0 > filename.find(containing):
                continue
        checkpoint_and_extension = re.findall(r'_checkpoint*.*', filename)[0]
        modelname = filename.replace(checkpoint_and_extension,'')
        
        already_covered = False
        for name in models_covered:
            if modelname == name:
                already_covered = True
                break
        if already_covered: continue
    
        models_covered += [modelname]

        history = {}
        try: model, history = load_model_and_history_from_latest_checkpoint(random, checkpointpath, modelname)
        except Exception as e:
            try: print('ERROR when loading model and history ' + modelname + ':')
            except: pass
            print(e)
            try: print('Failed to load model:' + modelanme)
            except: pass

        if history:
            if not (np.array(history['val_acc']) > performance_threshold_to_show).any():
                print('Model ' + modelname + ' skipped because val acc threshold set higher than what it achieved.')
                print('')
                continue
        fig = None
        ax = None
        try:
            try:
                fig, ax = history_plot_foo(history, 15, show=show, save=save, plotpath=plotpath, plotfilename=modelname)
                save_plot(fig, plotpath, modelname + '_history_acc')
            except:
                try:
                    fig, ax = history_plot_foo(history, 5, show=show, save=save, plotpath=plotpath, plotfilename=modelname)
                    save_plot(fig, plotpath, modelname + '_history_acc')
                except:
                    try:
                        fig, ax = history_plot_foo(history, 3, show=show, save=save, plotpath=plotpath, plotfilename=modelname)
                        save_plot(fig, plotpath, modelname + '_history_acc')
                    except:
                        try:
                            fig, ax = history_plot_foo(history, 1, show=show, save=save, plotpath=plotpath, plotfilename=modelname)
                            save_plot(fig, plotpath, modelname + '_history_acc')
                        except Exception as e:
                            print(e)
            print('Plotted ' + modelname)
        except Exception as e:
            try: print('ERROR when plotting history of ' + modelname + ':')
            except: pass
            print(e)
            print('Failed to plot history:')
            try: print(history)
            except: pass
            print('')
