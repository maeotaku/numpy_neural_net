from data.dataset import *
import numpy as np
import pickle


#hyperparameters
dataset_path = r"runs/iris.ds"
batch_size = 10
validate_every_no_of_batches = 80
epochs = 1000
input_size = 4
output_size = 3
hidden_shapes = [10]
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/iris_log.txt"
#iris dataset
with open(dataset_path, "rb") as input_file:
    iris_dataset = pickle.load(input_file)
    x = np.array(iris_dataset['data'])
    x = x / np.max(x, axis=0)
    y = np.array(iris_dataset['target'])
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.6, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
