import os
import sys
import numpy as np

from rbm import *
from datareader import *
from neuralnetworks import *

train_samples = 5000
test_samples = 200
window = 10000

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

    #rbm = RBM(window, trainX)
    #patern = rbm.test(testX)

    #autoencoder = AutoEncoder(window, trainX, trainY)
    #test_output = autoencoder.test(testX)

    # noise = patern[0][1]

    # denoised = denoise(testX, noise)

    #merged = merge_samples(testX, testsampleinfo, samplerate)

    output = compare_tracks(testsampleinfo)

if (__name__ == "__main__"):
    main(sys.argv[1:])
