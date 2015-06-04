__author__ = 'Simon'

import scipy.io as io
import os

NUM_DATA_SETS = 4
NUM_TAR_IND_FILES = 10
N_ROUND = 10


class Data:
    def __init__(self):
        self.domain_names = []
        self.Xsource = []           # features du domaine source
        self.ysource = []           # labels du domaine source
        self.Xtarget = []           # features du domaine target
        self.ytarget = []           # labels du domaine target
        self.nRound = 0             # utilisé par auteurs; à vérifier
        self.tar_train_index = []
        self.tar_test_index = []
        self.tar_background_index = []


# fonction qui ouvre les données qui sont en format matlab et les retourne
# sous forme d'un objet Data
def load_matlab_data():
    data = Data()
    data.domain_names.extend(['U00', 'U01', 'U02', 'U03'])
    for i in range(NUM_DATA_SETS):
        fullfilename = os.path.join('data', 'emailspam', "data_"+str(i+1)+".mat")
        imported_file = io.loadmat(fullfilename)
        if i+1 < NUM_DATA_SETS:
            data.Xsource.append(imported_file['features'])
            data.ysource.append(imported_file['labels'])
        else:
            data.Xtarget.append(imported_file['features'])
            data.ytarget.append(imported_file['labels'])
    data.nRound = N_ROUND
    tar_ind_dir = os.path.join('data', 'emailspam', 'tar_ind2')
    for j in range(NUM_TAR_IND_FILES):
        index = j+1
        fullfilename = os.path.join(tar_ind_dir, str(index)+".mat")
        imported_file = io.loadmat(fullfilename)
        data.tar_train_index.append(imported_file['tar_training_ind'])
        data.tar_test_index.append(imported_file['test_ind'])
        data.tar_background_index.append(imported_file['tar_background_ind'])
    return data