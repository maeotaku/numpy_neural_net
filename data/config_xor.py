from data.dataset import *
import numpy as np

#hyperparameters
batch_size = 2
validate_every_no_of_batches = 8
epochs = 2000
input_size = 2
output_size = 2
hidden_shapes = [2]
lr = 0.085
has_dropout=False
dropout_perc=0.5
output_log = "runs/xor_log.txt"
#XOR dataset - baseline to check backprop, update and forward calculations
ds_train = dataset(np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0]), batch_size)
ds_test = dataset(np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0]), batch_size)
ds_val = dataset(np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0]), batch_size)
