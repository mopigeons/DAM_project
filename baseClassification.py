__author__ = 'Simon'

from sklearn import svm
import utility as ut
import os
import scipy.io as io
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
import datetime
import sklearn.svm.libsvm as ls
import sklearn.svm as svm


class Result:
    def __init__(self):
        self.ap_sigmoid = 0
        self.acc_sigmoid = 0
        self.ap_no_sigmoid = 0
        self.acc_no_sigmoid = 0


def run_svm_fr(data):
    print("***---***---***   svm_fr   ***---***---***")
    # setpaths? will it be necessary to go get other data?
    # parametres
    C = 1
    kernel_types = ["linear", "poly"]
    kernel_params = [[0], []]
    # ajout de valeurs 1.1, 1.2, 1.3, 1.4, 1.5 à kernel_params[1]
    for i in range(5):
        ind = (10.0+float(i)+1)/10.0
        kernel_params[1].append(ind)
    result = main_svm_fr(data, C, kernel_types, kernel_params)
    kernel_types = ['linear']
    kernel_params = [0]
    result_dir = "results"
    result_file = os.path.join(result_dir, "svm_fr", "result_main_svm_fr.mat")
    io.savemat(result_file, {'result': result, 'C':C, 'kernel_types':kernel_types, 'kernel_params':kernel_params})


def main_svm_fr(data, C, kernel_types, kernel_params):
    results_directory_name = "results"
    result_file = os.path.join(results_directory_name, "svm_fr", "result_main_svm_fr.txt")
    # crée le répertoire "results" et le sous-répertoire "svm_fr"
    if not (os.path.exists(results_directory_name)):
        os.mkdir(results_directory_name)
    if not (os.path.exists(os.path.join(results_directory_name, 'svm_fr'))):
        os.mkdir(os.path.join(results_directory_name, 'svm_fr'))
    ut.log_print(result_file, '<==========  BEGIN @ '+datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")+'C = '+
                 str(C)+' ============>\n')
    # ici: on commence à la ligne 10 du fichier matlab
    for s in range(len(data.Xsource)):
        # création de répertoires results/svm_fr/decision_values/U00, U01, U02..
        dv_dir = os.path.join(results_directory_name, 'svm_fr', 'decision_values', data.domain_names[s])
        if not (os.path.exists(results_directory_name)):
            os.mkdir(results_directory_name)
        if not (os.path.exists(os.path.join(results_directory_name, 'svm_fr'))):
            os.mkdir(os.path.join(results_directory_name, 'svm_fr'))
        if not (os.path.exists(os.path.join(results_directory_name, 'svm_fr', 'decision_values'))):
            os.mkdir(os.path.join(results_directory_name, 'svm_fr', 'decision_values'))
        if not (os.path.exists(dv_dir)):
            os.mkdir(dv_dir)
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
                        # todo : need to test this once we have dv_file examples. need to make sure it works.
                        decision_values = io.loadmat(dv_file)['decision_values']
                    else:
                        Ymatrix = np.asarray(np.squeeze(y[train_index]), dtype=float)
                        #Xmatrix = np.ascontiguousarray(np.concatenate((np.asarray([range(len(train_index))]).transpose(),
                                                      #K[train_index][:, train_index]), axis=1))
                        Xmatrix = np.asarray(K[train_index][:, train_index])
                        classifier = svm.SVC(C=C, kernel='precomputed')
                        classifier.fit(Xmatrix, Ymatrix)
                        #Ypredictmatrix = np.ascontiguousarray(np.concatenate((np.asarray([range(len(tar_index))]).transpose(), K[tar_index][:, train_index]), axis=1))
                        Ypredictmatrix = np.asarray(K[tar_index][:, train_index])
                        #decision_values = classifier.predict_proba(Ypredictmatrix)
                        decision_values = classifier.decision_function(Ypredictmatrix)
                        #decision_values = np.array(decision_values * classifier.classes_[0])
                        decision_values = np.array(decision_values)
                        io.savemat(dv_file, {'decision_values': decision_values})

    # starting at line 54 matlab code
    results = []
    for r in range(data.nRound):
        tar_train_index = data.tar_train_index[r]
        tar_test_index = np.array(data.tar_test_index[r]-1).flatten()
        all_test_dv = []
        for s in range(len(data.Xsource)):
            dv_dir = os.path.join(results_directory_name, 'svm_fr', 'decision_values', data.domain_names[s])
            for kt in range(len(kernel_types)):
                kernel_type = kernel_types[kt]
                for kp in range(len(kernel_params[kt])):
                    kernel_param = kernel_params[kt][kp]
                    # starting at line 65 in MatLab code
                    dv_file = os.path.join(dv_dir, 'dv_round='+str(r)+'_C='+str(C)+'_'+str(kernel_type)+'_'+str(kernel_param)+'.mat')
                    decision_values = io.loadmat(dv_file)['decision_values']
                    ap_datayt = np.squeeze(np.asarray(data.ytarget))
                    ap_dv = np.squeeze(decision_values)
                    ap = ut.calc_ap(ap_datayt[tar_test_index], ap_dv[tar_test_index])
                    # acc = np.average(ap_datayt[tar_test_index] == np.sign(ap_dv[tar_test_index]))
                    aux_acc = (ap_datayt[tar_test_index] == np.sign(ap_dv[tar_test_index]))
                    aux_sum = 0
                    for elem in aux_acc:
                        aux_sum += elem
                    acc = aux_sum/len(aux_acc)
                    #todo: acc above seems to be mostly okay, but is sometimes a bit lower than matlab equivalent...
                    ut.log_print(result_file, str(ap)+'\t'+str(acc)+' @ round='+str(r)+', C='+str(C)
                                 +', kernel='+str(kernel_type)+', kernel_param='+str(kernel_param)
                                 +', '+str(data.domain_names[s])+'\n')
                    all_test_dv.append(decision_values)
        all_test_dv = np.array(all_test_dv)
        dv = np.average(1/(1+np.exp(-1*all_test_dv)), 0)
        # line 76 of matlab code
        ap_datayt = np.squeeze(np.asarray(data.ytarget))
        ap_dv = np.squeeze(dv)
        ap = ut.calc_ap(ap_datayt[tar_test_index], ap_dv[tar_test_index])
        aux_acc = (ap_datayt[tar_test_index] == np.sign(ap_dv[tar_test_index]))
        aux_sum = 0
        for elem in aux_acc:
            aux_sum += elem
        acc = aux_sum/len(aux_acc)
        result = Result()
        result.ap_sigmoid = ap
        result.acc_sigmoid = acc
        ut.log_print(result_file, 'SIGMOID '+str(ap)+'\t'+str(acc)+' @ round='+str(r)+', C='+str(C)+'\n')
        dv = np.average(all_test_dv, 0)
        ap_datayt = np.squeeze(np.asarray(data.ytarget))
        ap_dv = np.squeeze(dv)
        ap = ut.calc_ap(ap_datayt[tar_test_index], ap_dv[tar_test_index])
        aux_acc = (ap_datayt[tar_test_index] == np.sign(ap_dv[tar_test_index]))
        aux_sum = 0
        for elem in aux_acc:
            aux_sum += elem
        acc = aux_sum/len(aux_acc)
        result.ap_no_sigmoid = ap
        result.acc_no_sigmoid = acc
        results.append(result)
        ut.log_print(result_file, 'NO SIGMOID '+str(ap)+'\t'+str(acc)+' @ round='+str(r)+', C='+str(C)+'\n')

    ut.log_print(result_file, '<==========  END: '+datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
                 + ', C = ' + str(C) + ' ============>\n')
    return results