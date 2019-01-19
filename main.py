import os
import sys
import numpy as np
import time

from rbm import *
from datareader import *
from neuralnetworks import *

train_samples = 1000
test_samples = 300
window = 10000

runRBM = False

path = "C:\\all\\"

def main(argv):

    trainPath = os.path.normpath(path + "\\audio_train\\")
    testPath = os.path.normpath(path + "\\audio_test\\")

    trainFiles = os.listdir(trainPath)
    testFiles = os.listdir(testPath)

    for epoch in [10, 15, 20]:
        for samples in [500, 1000, 2000]:
            if(epoch == 10 and samples != 2000):
                continue
            train_samples = samples
            trainX, trainY, testX, testY, samplerate, testsampleinfo = read_data(trainFiles, trainPath, testFiles, testPath, window, train_samples, test_samples)
            print("Data ready")

            trainX = np.asarray(trainX)
            trainY = np.asarray(trainY)
            testX = np.asarray(testX)

            run_case(trainX, testX, trainY, train_samples, samplerate, testsampleinfo, 0.5, epoch, 0.01, 1000)


def run_case(trainX, testX, trainY, train_samples, samplerate, testsampleinfo, hid_coef, epochs, train_rate, batch_size):
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
        autoencoder = AutoEncoder(window, trainX, trainY, batch_size, epochs)
        test_output = autoencoder.test(testX)
    

    merged = merge_samples(test_output, testsampleinfo, samplerate)

    #samplename = "samples-rbm-"+str(train_samples) + "tr-" + str(test_samples) + "te-" + str(hid_coef) + "h-" + str(epochs) + "e-" + str(train_rate) + "r-" + str(batch_size)+ 'b'

    mean_error = compare_tracks(testsampleinfo)

    end = time.time()
    run_time = end - start
    print(run_time)

    if(runRBM):
        param_string = str(window) + "  " + str(train_samples) + "  " + str(test_samples) + "  " + str(hid_coef) + "  " + str(epochs) + "  " + str(train_rate) + "  " + str(batch_size) + "  " + str(mean_error) + "  " + str(run_time)
        print(param_string)
        with open("outputs.txt", "a") as myfile:
            myfile.write(param_string + "\n")
        os.rename("samples", "samples-rbm-"+str(train_samples) + "tr-" + str(test_samples) + "te-" + str(hid_coef) + "h-" + str(epochs) + "e-" + str(train_rate) + "r-" + str(batch_size)+ 'b')
        os.makedirs("samples")

if (__name__ == "__main__"):
    main(sys.argv[1:])
