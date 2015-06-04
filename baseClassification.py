__author__ = 'Simon'

from sklearn import svm
import utility as ut
import os
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix

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

    return "a string of words that marks the end of this function"  # todo: changer le retour