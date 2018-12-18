import os
import sys
import numpy as np
import soundfile as sf
from sklearn.neural_network import BernoulliRBM

samples = 10
window = 100
white_noise = 0.05

path = "F:\\all\\"

def main(argv):
    trainPath = os.path.normpath(path + "\\audio_train\\")
    testPath = os.path.normpath(path + "\\audio_test\\")

    trainFiles = os.listdir(trainPath)[:samples]
    testFiles = os.listdir(testPath)[:samples]

    rbm = BernoulliRBM()

    for sample in trainFiles:
        data, samplerate = sf.read(trainPath + "\\" + sample)
        noise = np.random.normal(0, white_noise, data.size)
        signal = data + noise

        sf.write(sample.replace(".wav", "_noise.wav"), signal, samplerate)

        for i in range(0, len(sample), window): 
            X = signal[i:i + window].reshape(1,-1)
            Y = data[i:i + window].reshape(1,-1)
            rbm.partial_fit(X,Y)

    for sample in testFiles:
        data, samplerate = sf.read(testPath + "\\" + sample)
        noise = np.random.normal(0, white_noise, data.size)
        signal = data + noise

        sf.write(sample.replace(".wav", "_test_noise.wav"), signal, samplerate)

        outputSignal = []

        for i in range(0, len(sample), window): 
            X = signal[i:i + window].reshape(1,-1)
            output = rbm.score_samples(X)
            outputSignal.append(output)

        sf.write(sample.replace(".wav", "_test_denoised.wav"), outputSignal, samplerate)  

if (__name__ == "__main__"):
    main(sys.argv[1:])
