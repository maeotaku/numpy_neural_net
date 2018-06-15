from sklearn import datasets
import numpy as np
import pickle

iris = datasets.load_iris()
filename = 'iris.ds'
outfile = open(filename,'wb')
pickle.dump(iris, outfile)
outfile.close()
