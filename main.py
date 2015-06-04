__author__ = 'Simon'

import baseClassification as bC
import utility as ut
import univerdam as un



def main():
    data = ut.load_matlab_data()
    # todo: why is there a random permutation in matlab load_data file (lines 20-24)?
    un.launch_univerdam(data)

    # todo:
    #run each s_i through svm
    #(keep results)
    # [continue]



if __name__ == "__main__":
    main()