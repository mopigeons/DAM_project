__author__ = 'Simon'

from sklearn import svm
import utility as ut
import os
import scipy.io as io
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
import sklearn.svm.libsvm as ls

def run_svm_fr(data):
    print("***---***---***   svm_fr   ***---***---***")
    # setpaths? will it be necessary to go get other data?
    # parametres
    C = 1
    kernel_types = ["linear", "poly"]
    kernel_params = [[0]]
    kernel_params = [[0], []]
    # ajout de valeurs 1.1, 1.2, 1.3, 1.4, 1.5 à kernel_params[1]
    for i in range(5):
        ind = (10.0+float(i)+1)/10.0
        kernel_params[1].append(ind)
    result = main_svm_fr(data, C, kernel_types, kernel_params)
    # todo: line 14+ in matlab code
    return result  # todo : changer le retour


def main_svm_fr(data, C, kernel_types, kernel_params):
    results_directory_name = "results"
    result_file = os.path.join(results_directory_name, "svm_fr", "result_main_svm_fr.txt")
    # crée le répertoire "results" et le sous-répertoire "svm_fr"
    if not (os.path.exists(results_directory_name)):
        os.mkdir(results_directory_name)
    if not (os.path.exists(os.path.join(results_directory_name, 'svm_fr'))):
        os.mkdir(os.path.join(results_directory_name, 'svm_fr'))
    # ici: on commence à la ligne 10 du fichier matlab
    for s in range(len(data.Xsource)):
        # création de répertoires results/svm_fr/decision_values/U00, U01, U02..
        dv_dir = os.path.join(results_directory_name, 'svm_fr', 'decision_values', data.domain_names[s])
        if not (os.path.exists(results_directory_name)):
            os.mkdir(results_directory_name)
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]
        Xsparse = sp.sparse.vstack([data.Xtarget[0], csc_matrix(Xsource)])
        S = Xsparse*Xsparse.transpose()
        S = S.todense()
        y = np.concatenate((data.ytarget[0], ysource))
        src_index = [i + data.Xtarget[0].shape[0] for i in range(Xsource.shape[0])]
        tar_index = [i for i in range(data.Xtarget[0].shape[0])]
        # passé ici: ligne 26+ du code matlab
        for kt in range(len(kernel_types)):
            kernel_type = kernel_types[kt]
            for kp in range(len(kernel_params[kt])):
                kernel_param = kernel_params[kt][kp]
                K = ut.calc_kernel_S(kernel_type, kernel_param, S)
                K[src_index][:, src_index] *= 2
                K[tar_index][:, tar_index] *= 2
                for r in range(data.nRound):
                    tar_train_index = data.tar_train_index[r]
                    tar_test_index = data.tar_test_index[r]
                    train_index = np.concatenate((src_index, np.squeeze(tar_train_index)))
                    dv_file = os.path.join(dv_dir, "dv_round="+str(r)+"_C="+str(C)+"_"+kernel_type+"_"+str(kernel_param)+".mat")
                    if os.path.exists(dv_file):
                        decision_values = io.loadmat(dv_file)
                    else:
                        Ymatrix = np.asarray(np.squeeze(y[train_index]), dtype=float)
                        Xmatrix = np.ascontiguousarray(np.concatenate((np.asarray([range(len(train_index))]).transpose(),
                                                      K[train_index][:, train_index]), axis=1))
                        model = ls.fit(Xmatrix, Ymatrix, C=C)
                        #model = ls.fit(Xmatrix, Ymatrix)
                        # todo: line 44 of matlab code (svmpredict)





    return "a string of words that marks the end of this function"  # todo: changer le retour