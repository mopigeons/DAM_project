__author__ = 'Simon'

import scipy.io as io
import numpy as np
import scipy as sp
import os
from scipy.sparse import csc_matrix

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


def calc_kernel_S(kernel_type, kernel_param, S):
    if kernel_type == "linear":
        K = S
    elif kernel_type == "poly":
        K = np.power((S+1), kernel_param)
    else:
        print("error in calc_kernel_S : kernel_type unrecognized!")
    return K

def calc_ap(gt, desc):
    assert len(gt) == len(desc)
    gt = np.asarray(gt).flatten()
    desc = np.asarray(desc).flatten()
    desc *= -1
    ind = desc.argsort()
    dv = desc
    dv.sort()
    dv = (-1*dv)
    gt = gt[ind]
    pos_ind = np.where(gt > 0) # tuple where first element is the array containing the elements where gt[i] > 0
    npos = len(pos_ind[0])
    if npos == 0:
        ap = 0
    else:
        npos_array = np.array(range(npos))+1
        pos_ind_array = np.array(pos_ind).flatten() + 1
        divarray = (npos_array/pos_ind_array)
        ap = np.average(divarray)
    return [ap]


def log_print(log_file, varargin):
    with open(log_file, "a") as file:
        file.write(varargin)


def save_mmd_fr(data):
    result_dir = 'results'
    kernel_types = ['linear', 'poly']
    kernel_params = [[0], []]
    # ajout de valeurs 1.1, 1.2, 1.3, 1.4, 1.5 à kernel_params[1]
    for i in range(5):
        ind = (10.0+float(i)+1)/10.0
        kernel_params[1].append(ind)
    for s in range(len(data.Xsource)):
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]

        mmd_dir = os.path.join(result_dir, 'mmd_values_fr', data.domain_names[s])
        if not (os.path.exists(result_dir)):
            os.mkdir(result_dir)
        if not (os.path.exists(os.path.join(result_dir, 'mmd_values_fr'))):
            os.mkdir(os.path.join(result_dir, 'mmd_values_fr'))
        if not (os.path.exists(mmd_dir)):
            os.mkdir(mmd_dir)
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]
        Xsparse = sp.sparse.vstack([data.Xtarget[0], csc_matrix(Xsource)])
        S = Xsparse*Xsparse.transpose()
        S = S.todense()
        y = np.concatenate((data.ytarget[0], ysource))
        src_index = [i + data.Xtarget[0].shape[0] for i in range(Xsource.shape[0])]
        tar_index = [i for i in range(data.Xtarget[0].shape[0])]
        ss = np.zeros((len(src_index)+len(tar_index), 1))
        ss[src_index] = 1/len(src_index)*100
        ss[tar_index] = -1/len(tar_index)*100

        for kt in range(len(kernel_types)):
            kernel_type = kernel_types[kt]
            for kp in range(len(kernel_params[kt])):
                kernel_param = kernel_params[kt][kp]
                K = calc_kernel_S(kernel_type, kernel_param, S)
                K[src_index][:, src_index] *= 2
                K[tar_index][:, tar_index] *= 2

                mmd_file = os.path.join(mmd_dir, 'mmd_'+str(kernel_type)+'_'+str(kernel_param)+'.mat')
                if os.path.exists(mmd_file):
                    mmd_value = io.loadmat(mmd_file)['mmd_value']
                else:
                    mmd_value = (ss.transpose()*K*ss)
                    io.savemat(mmd_file, {'mmd_value': mmd_value})






