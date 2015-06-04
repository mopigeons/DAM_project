__author__ = 'Simon'

from sklearn import svm
import utility as ut

def run_svm_fr(data):
    print("svm fr")
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

    return "a string of words"  # todo: changer le retour