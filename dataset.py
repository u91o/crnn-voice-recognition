import os.path
import scipy.io.wavfile
import tensorflow as tf
import tensorflow_datasets as tfds

import config


def pad_array(arr):
    """ pad_arr
    If the input array is less than the length specified in config.height,
    this function will zero-pad it to provide a uniformly sized array
    """
    i = config.height - arr.shape[0]
    arr = tf.pad(arr, [[0,i], [0,0]])
    return arr

def extract_mfcc(signal, sample_rate=16000):
    """ extract_mfcc
    This function serves to extract a set of mfcc features from the input
    signal
    """
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(signal, frame_length=1024, frame_step=256,
                           fft_length=1024)
    spectrograms = tf.abs(stfts)
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    # Compute MFCCs from log_mel_spectrograms and take the specified width.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :config.width]
    # pad arrays
    mfccs = tf.py_function(pad_array, [mfccs], tf.float32)

    return mfccs

def process_samples(audio, label):
    """
    Wrapper function to extract mfcc features from a Tensorflow Dataset object
    """
    audio = tf.cast(audio, tf.float32)
    features = extract_mfcc(audio)
    features.set_shape((config.height, config.width))

    return features, label

def samples_from_wav(fp):
    """
    Extracts the audio signal from a WAV file
    """
    sr, samples = scipy.io.wavfile.read(fp)
    assert sr == 16000
    return samples

class Dataset:
    """
    Class for custom datasets
    """
    sample_count = 0
    dataset = None
    labels = []

    def __init__(self):
        self._load_dataset()
        self._prepare_dataset()

    def _dataset_generator(self):
        for i, label in enumerate(self.labels):
            for f in os.listdir(os.path.join(config.data_dir, label)):
                fp = os.path.join(config.data_dir, label, f)
                samples = samples_from_wav(fp)
                if samples.shape == (16000,):
                    x = tf.convert_to_tensor(samples)
                    y = tf.convert_to_tensor(i, tf.int64)
                    yield (x,y)

    def _load_dataset(self):
        """
        Loads training data from config.data_dir into a generator function
        """
        assert os.path.exists(config.data_dir)
        self.labels = os.listdir(config.data_dir)

        for label in self.labels:
            self.sample_count += len(os.listdir('data/' + label))

        self.dataset = tf.data.Dataset.from_generator(
            self._dataset_generator,
            output_types=(tf.int64, tf.int64),
            output_shapes=((16000,), ())
        )

    def _prepare_dataset(self):
        """
        Augment dataset to produce MFCC features
        """
        self.dataset = self.dataset.map(
            process_samples,
            tf.data.experimental.AUTOTUNE
        )
        # cache dataset in memory
        self.dataset = self.dataset.cache()
        # shuffle dataset
        self.dataset = self.dataset.shuffle(self.sample_count)
        # set batch size
        self.dataset = self.dataset.batch(config.batch_size)
        # prefetch to increase performace
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def get_batches(self):
        """
        Return training batches
        """
        return self.dataset


class TFDS:
    splits = {}
    info = None
    labels = []

    def __init__(self):
        self._download_dataset()
        self._load_dataset()
        self._prepare_dataset()

    def _download_dataset(self):
        """
        Download dataset if it isn't available
        """
        builder = tfds.builder('speech_commands')
        builder.download_and_prepare()

    def _load_dataset(self):
        """
        Read local TFRecord files and load them as Dataset objects
        """
        (train, validation, test), self.info = tfds.load(
            'speech_commands',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True)
        self.splits['train'] = train
        self.splits['validation'] = validation
        self.splits['test'] = test

        self.labels = self.info.features['label'].names

    def _prepare_dataset(self):
        """
        Augment dataset to produce MFCC features
        """
        for split in self.splits:
            # extract mfccs and reshape input
            self.splits[split] = self.splits[split].map(
                process_samples,
                tf.data.experimental.AUTOTUNE)
            # only shuffle files if training
            if split == 'train':
                self.splits[split] = self.splits[split].cache()
                self.splits[split] = self.splits[split].shuffle(
                    self.info.splits[split].num_examples)
            # set batch size
            self.splits[split] = self.splits[split].batch(config.batch_size)
            # cache evaluation batches after batching as batches can be the
            # same between epochs
            if split != 'train':
                self.splits[split] = self.splits[split].cache()
            # prefetch to boost performance
            self.splits[split] = self.splits[split].prefetch(
                tf.data.experimental.AUTOTUNE)

    def get_batches(self):
        """
        Return training batches
        """
        return (self.splits[x] for x in self.splits)



if __name__ == '__main__':
    ds = TFDS()
    print(ds.info.features['label'].names)
