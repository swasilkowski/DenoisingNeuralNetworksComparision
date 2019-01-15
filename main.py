import os
import sys
import numpy as np

from rbm import *
from datareader import *
from neuralnetworks import *

train_samples = 5000
test_samples = 100
window = 20000

runRBM = True

path = "C:\\all\\"

def main(argv):
    trainPath = os.path.normpath(path + "\\audio_train\\")
    testPath = os.path.normpath(path + "\\audio_test\\")

    trainFiles = os.listdir(trainPath)
    testFiles = os.listdir(testPath)

    trainX, trainY, testX, testY, samplerate, testsampleinfo = read_data(trainFiles, trainPath, testFiles, testPath, window, train_samples, test_samples)
    print("Data ready")

    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    testX = np.asarray(testX)

    test_output = None

    if(runRBM):

        rbm = RBM(window, trainX)

        test_output = []
        for test in testX:
            output = rbm.test(test.reshape(1,-1))
            test_output.append(output[0][1])

        test_output = np.asarray(test_output)

    if(runRBM == False):
        autoencoder = AutoEncoder(window, trainX, trainY)
        test_output = autoencoder.test(testX)
    

    merged = merge_samples(test_output, testsampleinfo, samplerate)

    output = compare_tracks(testsampleinfo)

if (__name__ == "__main__"):
    main(sys.argv[1:])
