#coding=utf-8
import pickle
import sys

def dump_pickle_model(outpath, *layers):
    print >> sys.stderr, 'dump model', outpath
    with open(outpath, 'w') as file:
        pickle.dump(layers, file)
   
def load_pickle_model(inpath):
    return pickle.load(open(inpath))
    
