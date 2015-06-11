__author__ = 'Simon'

import baseClassification as bC
import utility as ut
import univerdam as un


def main():
    data = ut.load_matlab_data()
    # todo: why is there a random permutation in matlab load_data file (lines 20-24)?
    print(bC.run_svm_fr(data))
    ut.save_mmd_fr(data)
    un.launch_univerdam(data)


if __name__ == "__main__":
    main()