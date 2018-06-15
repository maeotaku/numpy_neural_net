from data.config import *
from data.dataset import *
from report.dumps import *
from nn.model import model
from nn.funcs import *
import numpy as np

def test(ds, verbose=False, phase="Validation"):
    ds.reset()
    hits = 0
    mean_loss = 0
    while not(ds.iter_done()):
        x, y = ds.next()
        o, batch_loss = nn.forward(x, y, train=False)
        print(o)
        hits += batch_hits(o, y)
        mean_loss += np.mean(batch_loss)
        #if verbose:
        #    print("Loss: " + str(mean_loss), " Predicted: " + str(o), " Expected: " + str(y))
    accuracy = float(hits) / float(ds.size)
    mean_loss = float(mean_loss) / float(ds.size)
    if verbose:
        print(phase + " Accuracy: " + str(accuracy) + " Mean Loss " + str(mean_loss))
    return accuracy, mean_loss

def train(nn, hp, val_hist, train_hist, logger):
    cur_epoch = 1
    cur_iter = 1
    for i in range(1, hp.epochs+1):
        train_loss = 0
        hits = 0
        cur_trained = 0
        while not(hp.ds_train.iter_done()):
            x, y = hp.ds_train.next()
            #print(y)
            o, batch_loss = nn.forward(x, y)
            nn.backward(y,o)
            nn.update(hp.lr)

            hits += batch_hits(o, y)
            cur_trained += len(x)
            train_loss += np.mean(batch_loss)

            if cur_iter % hp.validate_every_no_of_batches == 0:

                train_accuracy = float(hits) / float(cur_trained)
                train_loss = float(train_loss) / float(cur_trained)
                train_hist.add(cur_iter, train_loss, train_accuracy)
                logger.write( (cur_epoch, "Training", cur_iter, train_accuracy, train_loss) )
                hits = 0
                train_loss = 0

                val_accuracy, val_loss = test(hp.ds_val, True)
                val_hist.add(cur_iter, val_loss, val_accuracy)
                logger.write( (cur_epoch, "Val", cur_iter, val_accuracy, val_loss) )
            cur_iter+=1
        cur_epoch+=1
        hp.ds_train.reset()
    return val_hist

#load hyperparameters and settings according to dataset enum
hp = hyperparams(ConfigEnum.XOR)
#hp = hyperparams(ConfigEnum.IRIS)
#hp = hyperparams(ConfigEnum.MNIST)

#model has number of inputs, number of outputs, and list with sizes of hidden layers
#requires at least 1 hidden layer, else fails assert
nn = model(hp.input_size, hp.output_size, hp.hidden_shapes, sigmoid, sigmoid_grad, has_dropout=hp.has_dropout, dropout_perc=hp.dropout_perc)

val_hist = historian()
train_hist = historian()
logger = nnlogger(hp.output_log, ("Epoch", "Phase", "Iteration", "Accuracy", "Loss") )
train(nn, hp, val_hist, train_hist, logger)
test(hp.ds_test, verbose=True, phase="Test")
nnplotter.view(val_hist, train_hist) #see results on plot
logger.close()


'''
#this stuff just changes percentages of training and validation to check for best accuracies
logger = nnlogger(hp.output_log, ("Epoch", "Phase", "Iteration", "Accuracy", "Loss") )

maxi = 0
maxi_next_perc_train = 0
maxi_next_perc_val = 0
for next_perc_train in range(10, 100, 10):
    for next_perc_val in range(10, 100, 10):
        val_hist = historian()
        train_hist = historian()
        hp.split_again(next_perc_train / 100.0, next_perc_val / 100.0)
        val_hist = train(nn, hp, val_hist, train_hist, logger)
        if maxi < np.max(np.array(val_hist.acc)):
            maxi = np.max(np.array(val_hist.acc))
            maxi_next_perc_train = next_perc_train
            maxi_next_perc_val = next_perc_val
print("Max Accuracy " + str(maxi) + " Train: " + str(maxi_next_perc_train) + " Val: " + str(maxi_next_perc_val))
logger.close()
'''
