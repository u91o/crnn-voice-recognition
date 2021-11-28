import os

height = 59
width = 13
dropout = 0.5 # dropout rate, percentage of input to drop

batch_size = 100
epochs = 10
validation_split = 0.10
validation_freq = 2 # number of epochs between each validation

# paths
basedir = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = os.path.join(basedir, 'ckpts')
log_dir = os.path.join(basedir, 'logs')
data_dir = os.path.join(basedir, 'data')
test_dir = os.path.join(basedir, 'test')
