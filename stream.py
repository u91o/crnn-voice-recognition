import matplotlib.pyplot as plt
import os
import sounddevice as sd
import tensorflow as tf
import time

from scipy.io.wavfile import read, write

import config
import dataset
import model


sr = 16000
duration = 1 # seconds
frames = int(sr * duration)
device = 'pulse'
channels = 1
dtype = 'int16'

sd.default.device = 'pulse'

def sanity_test():
    """
    Sanity test to make sure things are working
    """
    for f in os.listdir(config.test_dir):
        print("TEST:", f)
        fp = os.path.join(config.test_dir, f)
        samples = dataset.samples_from_wav(fp)
        model.predict(model.crnn, samples)
        print("##########")

def predict_recording():
    """
    Helper function that allows the user to interactively make predictions
    on recorded audio
    """
    while True:
        input("recording!")
        time.sleep(0.2)
        samples = sd.rec(frames=frames, samplerate=sr, channels=channels,
                         dtype=dtype, blocking=True)
        input("playing!")
        sd.play(samples, sr, blocking=True)
        samples = tf.reshape(samples, [-1])
        model.predict(model.crnn, samples)

def record_ambient_noise(n):
    """
    Helper function that allows a user to continuously record n number of
    audio samples for the model to train with
    """
    for i in range(n):
        print("recording", i)
        samples = sd.rec(frames=frames, samplerate=sr, channels=channels,
                         dtype=dtype, blocking=True)
        fp = os.path.join(config.data_dir, '_unknown_', str(i)+'.wav')
        write(fp, sr, samples)

def record_samples(folder):
    """
    Helper function that allows the user to specify a folder to save
    recorded samples into
    """
    import random, string
    name = ''.join(random.choices(string.digits+string.ascii_letters, k=6))
    while True:
        input("recording")
        time.sleep(0.2)
        samples = sd.rec(frames=frames, samplerate=sr, channels=channels,
                         dtype=dtype, blocking=True)
        fp = os.path.join(config.data_dir, folder, '_user_'+ name + '.wav')
        write(fp, sr, samples)

def callback(indata, frames, time, status):
    """
    Callback function for InputStream
    """
    if status:
        print("the stream has died")
    if any(indata):
        samples = tf.reshape(indata, [-1])
        model.predict(model.crnn, samples)
    else:
        print("no input")

def predict_input_stream():
    """
    Allows the user to make predictions on a continous audio stream
    """
    with sd.InputStream(samplerate=sr, blocksize=frames, device=device,
                        channels=channels, dtype=dtype, callback=callback):
        while True:
            time.sleep(0.01)

def plot_wav(fp):
    sr, sig = read(fp)
    plt.plot(sig)
    plt.show()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # sanity_test()
    # record_samples('happy')
    # predict_recording()
    # record_ambient_noise()
    predict_input_stream()

    # plot_wav('rec.wav')
    # plot_wav('test/happy.wav')

