from sklearn.datasets import fetch_mldata
import numpy as np
import pickle

mnist = fetch_mldata('MNIST original')
filename = 'nmist.ds'
outfile = open(filename,'wb')
pickle.dump(mnist, outfile)
outfile.close()
