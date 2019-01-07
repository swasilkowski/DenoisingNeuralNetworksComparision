import numpy as np
import soundfile as sf

white_noise = 0.05
save_files = True
np.random.seed(0)

def read_data(trainFiles, trainPath, testFiles, testPath, window, samples_no):
    trainX, trainY, samplerate = read_train_data(trainFiles, trainPath, window, samples_no)
    testX, testY, testsampleinfo = read_test_data(testFiles, testPath, window, samples_no)

    return (trainX, trainY, testX, testY, samplerate, testsampleinfo)


def read_train_data(trainFiles, trainPath, window, samples_no):
    trainX = []
    trainY = []

    for sample in trainFiles:
        data, samplerate = sf.read(trainPath + "\\" + sample)
        noise = np.random.normal(0, white_noise, data.size)
        signal = data + noise

        if(save_files):
            sf.write("samples\\" + sample.replace(".wav", "_noise.wav"), signal, samplerate)

        for i in range(0, len(signal), window):
            if(i+window < len(signal)):
                trainX.append(signal[i:i + window])
                trainY.append(data[i:i + window])
                if(len(trainX) >= samples_no):
                    return (trainX, trainY, samplerate)

    return (trainX, trainY, samplerate)

def read_test_data(testFiles, testPath, window, samples_no):
    testX = []
    testY = []
    testsampleinfo = []

    for sample in testFiles:
        data, samplerate = sf.read(testPath + "\\" + sample)
        noise = np.random.normal(0, white_noise, data.size)
        signal = data + noise

        if(save_files):
            sf.write("samples\\" + sample.replace(".wav", "_test_noise.wav"), signal, samplerate)

        for i in range(0, len(signal), window):
            if(i+window < len(signal)):
                testX.append(signal[i:i + window])
                testY.append(data[i:i + window])
                testsampleinfo.append(sample)
                if(len(testX) >= samples_no):
                    return (testX, testY, testsampleinfo)

    return (testX, testY, testsampleinfo)

def denoise(samples, noise):
    for sample in samples:
        sample = sample - noise
    return samples

def merge_samples(samples, sampleinfo, samplerate):
    filename = sampleinfo[0]
    outputSignal = samples[0].tolist()
    signals = []
    for indx, sample in enumerate(samples):
        if(indx == 0): 
            continue
        if(filename == sampleinfo[indx]):
            outputSignal.extend(sample.tolist())
        else:
            sf.write("samples\\" + sampleinfo[indx-1].replace(".wav", "_test_denoised.wav"), outputSignal, samplerate)
            signals.append(outputSignal)
            filename = sampleinfo[indx]
            outputSignal = sample.tolist()

    return signals