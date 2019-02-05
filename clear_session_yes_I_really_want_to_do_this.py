from session import *


if __name__ == '__main__':
    datapath = os.getcwd() + '/data'
    
    (sessionname, rawdatanames, datasetnames, sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath) = CAI_args(datapath)
    session_clear(sessionpath, rawdatapaths, datasetpaths, checkpointpath, modelspath, historiespath)
