import sklearn
import math
import pdb
import sys
import  os.path
import  numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing, decomposition
import torchvision.models.vgg as models
import json
from PIL import Image
import pickle
from numpy.random import RandomState


class Affordances:
    def __init__(self, train, exclude, batchsz, k_shot, k_qry, dim_out):
        self.rand = RandomState(222)
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_fts.pkl"
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        self.classes = list(sorted(set([k.split("_00")[0] for k in self.inputs.keys()])))
        print(self.classes)
        self.num_classes = len(self.classes)
        self.input_scale = preprocessing.StandardScaler()
        self.input_scale.fit(list(self.inputs.values()))
        self.num_samples_per_class = k_shot + k_qry
        self.batch_size = batchsz
        self.dim_output = dim_out
        self.dim_latent = 2
        self.dim_input = len(list(self.inputs.values())[0])

    def next(self):
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class])
        o = self.rand.choice(len(self.classes), self.batch_size, replace=False)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            obj = self.classes[o[t]]
            obj_keys = list(set([k for k in self.inputs.keys() if k.startswith(obj)]))
            k = self.rand.choice(len(obj_keys), self.num_samples_per_class, replace=False)
            for n in range(self.num_samples_per_class):
                init_inputs[t,n] = self.input_scale.transform(self.inputs[obj_keys[k[n]]].reshape(1,-1))
                outputs[t,n] = o[t]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
