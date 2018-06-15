A Neural Network (nn) from scratch!
Author: Jose Carranza-Rojas

This is a neural network implementation using numpy on Python 3.5

In order to train a model, just run main.py in the directory root, using: python3 main.py. Currently code runs with Iris dataset
but more configurations are present for other datasets (see Data and config section for more).

Training:
The training process present in main.py file will create an entry in the "runs" folders, depending on the selected
settings and dataset (by default is IRIS). As the model trains, validation loss and accuracy are printed on console,
and at the end a plot is shown to the user. Also, a csv file is written to disk with the traing logs.
Test accuracy will be present in the console output after running all epochs.
By default, dropout is activated, so do not be surprised if test is higher with respect to validation (is possible).
The dataset split is not balanced, this was done in porpose since real life is never balanced in terms of classes.

Code is comprised of several modules, explained in next sections:
nn module:      all classes, functions and code for the neural net
data module :   dataset and settings handling
report module:  logging and plotting
tools module:   misc tools, mostly for downloading iris and nmist
runs folder: stores information about the run.

----------------------------------------The neural network (nn) ----------------------------------------
The network is comprised of operations, which are represented by the following classes, inside the "nn" module:

- nn.op:            class op represents the base class for all operations. Dense layers, Cross Entropy layers
                    and future, yet-to-be-implemented layers will inherit from this. All layers need implement
                    the forward and backward functions. op has the parameters (W, b) and knows how to update them
                    according to the gradient calculations.
- nn.dense:         represents a dense layer, or hidden layer in terms of nn. Has logic to calculate forward and backward
                    inherits from op. Contains mask for dropout.
- nn.loss_layer:    represents a cross entropy layer used for multi-class categorizations. Uses softmax to calculate
                    the probability distribution over the classes. Implements both forward, backward (based on
                    log loss derivatives), as well the log loss -log(softmax).
- nn.model:         represents a neural network, with its input, k hidden layers of any size, and log loss layer at the end. Allows to do
                    dropout during training. Receives a function reference on the constructor to be able to add different
                    activation functions and their derivatives. Relu and sigmoid can be used out-of-the-box, but more can be added.
- nn.funcs.py       includes functions used everywhere. Relu, sigmoid, log loss, softmax, among others. Also includes
                    corresponding derivatives.

NOTE: this implementation was designed to be extended with more layer types in the future, by inheriting from op class.




---------------------------------------Data and config----------------------------------------
The module "data" allows to handle datasets and configurations. The following classes are useful for this:

- data.dataset:     represents a dataset with targets/labels, in forms of numpy tensors.
                    Allows to iterate over the data in batches of a given size.

- data.dataset_splitter:    takes a complete dataset and divides it in train, validation and testing datasets. Uses the
                            dataset class to represent each. Shuffles the complete dataset before doing the split.

- data.config:      depending on an enum ConfigEnum imports a different py file that contains per-dataset specific
                    hyperparameters and settings. There are 3 different config files and more can be added if needed
                    in the future by just creating a new config py file and adding an enum.
                    Currently code includes a XOR, IRIS and MNIST datasets:
                    - data.config_iris.py: as per requirement. Dataset is normalized by feature as it is not image data.
                    - data.config_xor.py: used as baseline for calculations to check for correct values. Typical non-linearly
                    solvable problem to test a nn.
                    - data.config_nmist.py: tested for higher dimensionality given it uses images. Dataset is normalized by dividing
                    between 255.0.


--------------------------------------Plots and logging-----------------------------------------
The "report" module contains classes to handle logging of results and also plotting:

- report.historian: keeps information about iteration, loss and accuracy for any phase, training, val or test.

- report.nnlogger:  logs into file a historian class which contains iteration, loss and accuracy information over time. This is
                    written to a cvs file directly for easier future manipulation.

- report.nnplotter: plots iteration, loss and validation for both training and validation, using matplotlib.
