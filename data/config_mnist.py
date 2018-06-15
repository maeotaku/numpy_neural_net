from data.dataset import *
import numpy as np
import pickle


#hyperparameters
dataset_path = r"runs/nmist.ds"
batch_size = 32
validate_every_no_of_batches = 600
epochs = 10
input_size = 784
output_size = 10
hidden_shapes = [512, 128]
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/nmist_log.txt"
with open(dataset_path, "rb") as input_file:
    nmist = pickle.load(input_file)
    x = np.array(nmist.data)
    x = x / 255.0
    y = np.array(nmist.target).astype(np.int)
    #print(x)
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.8, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
