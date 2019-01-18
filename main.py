import os
import sys
import numpy as np
import time

from rbm import *
from datareader import *
from neuralnetworks import *

train_samples = 500
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

    run_case(trainX, testX, trainY, samplerate, testsampleinfo, 0.5, 5, 0.1, 100)


def run_case(trainX, testX, trainY, samplerate, testsampleinfo, hid_coef, epochs, train_rate, batch_size):
    start = time.time()

    test_output = None

    if(runRBM):

        print('RBM training: window = ' + str(window) + ', trainset = ' + str(train_samples) + ', testset = ' + str(test_samples) +', hidden = ' + str(hid_coef*window) + ', epochs = ' + str(epochs) + ', train_rate = ' + str(train_rate) +  ', batch_size = ' + str(batch_size))

        rbm = RBM(window, trainX, hid_coef, epochs, train_rate, batch_size)

        test_output = []
        for test in testX:
            output = rbm.test(test.reshape(1,-1))
            test_output.append(output[0][1])

        test_output = np.asarray(test_output)

    if(runRBM == False):
        autoencoder = AutoEncoder(window, trainX, trainY)
        test_output = autoencoder.test(testX)
    

    merged = merge_samples(test_output, testsampleinfo, samplerate)

    mean_error = compare_tracks(testsampleinfo)

    end = time.time()
    run_time = end - start
    print(run_time)


if (__name__ == "__main__"):
    main(sys.argv[1:])
