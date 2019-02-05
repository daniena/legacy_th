import os
import json
import numpy
from param_debug import debug

def make_path(path):
    superpath = '/'
    for foldername in path.split("/"):
        if not os.path.isdir(superpath + foldername):
            if debug:
                print('project/util/file_operations.make_path: Making path ' + superpath + foldername)
            os.mkdir(superpath + foldername)
        superpath = superpath + foldername + '/'

def remove_files_with_ending(path, ending):
    if not os.path.isdir(path):
        print('Attempted to remove files with ending ' + ending + ' from path ' + path + ', but no such path was found.')
        return
    
    for filename in os.listdir(path):
        if filename.endswith(ending):
            os.unlink(path + '/' + filename)

def save_json(path, name, jsonable):
    make_path(path)
    if not path.endswith('/'):
        path = path + '/'
    with open(path + name + '.json', 'w') as wjson:
        json.dump(jsonable, wjson)

def load_json(path, name):
    make_path(path)
    if not path.endswith('/'):
        path = path + '/'
    with open(path + name + '.json', 'r') as rjson:
        jsonable = json.load(rjson)
        return jsonable

def save_numpy(path, name, array):
    make_path(path)
    if not path.endswith('/'):
        path = path + '/'
    numpy.save(path + name, array)

def load_numpy(path, name):
    make_path(path)
    if not path.endswith('/'):
        path = path + '/'
    array = numpy.load(path + name + '.npy')
    return array
