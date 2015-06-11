__author__ = 'Simon'

from utility import Data
import os
import numpy as np
import scipy as sp
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

def launch_univerdam(data):
    # ----------------------------------------------------------------------------------------------------------------
    # définition des paramètres tel que faites par les auteurs
    # (en général, ils ont validé empiriquement, voir p. 512-513
    # pour une discussion des paramètres experimentaux)
    # ----------------------------------------------------------------------------------------------------------------
    c = 1
    lambda_l = 1
    lambda_d1 = 1
    lambda_d2 = 1
    beta = 100
    thr = 0.3
    virtual_label_type = 'svm_fr'
    kernel_types = ['linear', 'poly']
    kernel_params = [[0], []]
    # ajout de valeurs 1.1, 1.2, 1.3, 1.4, 1.5 à kernel_params[1]
    for i in range(5):
        ind = (10.0+float(i)+1)/10.0
        kernel_params[1].append(ind)
    # ----------------------------------------------------------------------------------------------------------------
    result = main_univerdam(data, c, lambda_l, lambda_d1, lambda_d2, thr, beta, virtual_label_type, kernel_types,
                            kernel_params)
    # todo: traitement à finir (ligne 17+ du code Matlab)
    return "tomato"     # todo: this isn't the actual return


def main_univerdam(data, C, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params):
    results_directory_name = 'results'
    result_filename = os.path.join(results_directory_name, 'univerdam', 'results_main_univerdam.txt')
    # crée le répertoire "results" et le sous-répertoire "univerdam"
    if not (os.path.exists(results_directory_name)):
        os.mkdir(results_directory_name)
    if not (os.path.exists(os.path.join(results_directory_name, 'univerdam'))):
        os.mkdir(os.path.join(results_directory_name, 'univerdam'))
    X_data = []
    y_data = []
    for indx in range(len(data.Xtarget)):
        X_data.append(data.Xtarget[indx])
    for ind in range(len(data.ytarget)):
        y_data.append(data.ytarget[ind])
    domain_index = [[]]
    offset = len(data.ytarget[0])
    for j in range(len(data.ytarget[0])):
        domain_index[0].append(j+1)
    for i in range(len(data.Xsource)):
        X_data.append(data.Xsource[i])
        y_data.append(data.ysource[i])
        domain_index.append([])
        for k in range(len(data.ysource[i])):
            domain_index[i+1].append(k+1+offset)
        offset += len(data.ysource)
    X = []
    X.extend(X_data[0])
    for j1 in range(len(X_data)-1):
        X.extend(X_data[j1+1])
    y = []
    for i1 in range(len(y_data)):
        for i2 in range(len(y_data[i1])):
            y.append(y_data[i1][i2][0])
    tar_index = domain_index[0]
    src_index = []
    for q in range(len(domain_index)-1):
        src_index.extend(domain_index[q+1])
    X_sparse = []
    for row in X_data:
        X_sparse.append(row)
    X_sparse = sp.sparse.vstack((row for row in X_sparse))
    K = X_sparse*X_sparse.transpose()
    K = K.todense()

    for r in range(data.nRound):
        tar_train_index = np.squeeze(data.tar_train_index[r])
        tar_test_index = np.squeeze(data.tar_test_index[r])
        all_test_dv = []
        mmd_values = []
        # Definitions:
        # DV : Decision Values
        # MMD : Maximum Mean Discrepancy
        for s in range(len(data.Xsource)):
            # vérification du type de virtual_label_type (paramètre)
            # Pour l'instant on passe seulement du '_fr', mais le "if" ici prévoit
            # si on veut essayer d'autres types de classificateur (tel qu'essayé par
            # les auteurs)
            if virtual_label_type[-2:] == "fr":
                dv_dir = os.path.join(results_directory_name, 'svm_fr', 'decision_values', data.domain_names[s])
                mmd_dir = os.path.join(results_directory_name, 'mmd_values_fr', data.domain_names[s])
            elif virtual_label_type[-2:] == "_s":
                dv_dir = os.path.join(results_directory_name, 'svm_s', 'decision_values', data.domain_names[s])
                mmd_dir = os.path.join(results_directory_name, 'mmd_values_at', data.domain_names[s])
            elif virtual_label_type[-2:] == "st":
                dv_dir = os.path.join(results_directory_name, 'svm_at', 'decision_values', data.domain_names[s])
                mmd_dir = os.path.join(results_directory_name, 'mmd_values_at',  data.domain_names[s])
            for kt in range(len(kernel_types)):
                kernel_type = kernel_types[kt]
                for kp in range(len(kernel_params[kt])):
                    kernel_param = kernel_params[kt][kp]
                    dv_file = os.path.join(dv_dir, "dv_round="+str(r)+"_C="+str(C)+"_"+kernel_type+"_"
                                           +str(kernel_param)+".mat")
                    if os.path.exists(dv_file):
                        decision_values = io.loadmat(dv_file)['decision_values']
                    else:
                        print('You need to run the required baseline algorithms to obtain the decision values required by algorithm')
                        return -1
                    mmd_file = os.path.join(mmd_dir, 'mmd_'+str(kernel_type)+'_'+str(kernel_param)+'.mat')
                    if os.path.exists(mmd_file):
                        mmd_value = (io.loadmat(mmd_file))['mmd_value']
                    else:
                        print('please run the proper save_mmd first to prepare the mmd values required by this algorithm')
                        return -1
                    mmd_values.append(mmd_value)
                    all_test_dv.append(decision_values)
        f_s = all_test_dv
        gamma_s = np.exp((-1*beta)*np.power(np.asarray(mmd_values), 2))
        gamma_s = np.array(gamma_s)
        gamma_s = gamma_s/np.sum(gamma_s)
        print(gamma_s)


    return "result"     # todo: this isn't the actual return
