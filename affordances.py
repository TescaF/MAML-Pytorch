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
    def __init__(self, train, batchsz, k_shot, k_qry, dim_out):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.rand = RandomState(222)
        #self.scale = np.array([544,408,1200,2*math.pi])/2.0
        #self.scale = np.array([640,480,1200,2*math.pi])/2.0
        self.dirs = np.array([1,-1,1,1])
        self.affs, vals_max, vals_min = [], [], []
        data_count = 0
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/reduced_fts_0.95.pkl"
        for aff in range(2,7):
            aff_loc = "/home/tesca/data/part-affordance-dataset/features/4d_small_aff/aff_" + str(aff) + "_positions.pkl"
            with open(aff_loc, 'rb') as handle:
                aff_data = pickle.load(handle)      #dict(category) = [img1, img2, ...]
            keys = list(sorted(aff_data.keys()))
            if train:
                valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (not k.split("_00")[0].endswith("01"))]
            else:
                #valid_keys = [k for k in keys if (not None in aff_data[k][-1])]
                valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (k.split("_00")[0].endswith("01"))]
            vals = [aff_data[k][-1]*self.dirs for k in valid_keys]
            val_m = np.matrix(vals) 
            vals_max.append(np.max(val_m,axis=0))
            vals_min.append(np.min(val_m,axis=0))
            self.affs.append([valid_keys, aff_data])
            data_count += len(valid_keys)

        ## Load VGG features for all images
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        print(str(data_count) + " image/aff pairs loaded")
        self.num_samples_per_class = k_shot + k_qry 
        self.batch_size = batchsz
        self.dim_output = dim_out
        self.dim_latent = 4
        self.dim_input = len(list(self.inputs.values())[0])

        all_max = np.max(np.array(vals_max),axis=0)
        all_min = np.min(np.array(vals_min),axis=0)
        max_mul = np.ones((self.dim_latent, self.dim_output))
        max_val = np.max([abs(all_max), abs(all_min)],axis=0)
        upper = np.matmul(max_val, max_mul) + 0.68
        lower = np.matmul(max_val, np.negative(max_mul)) - 0.68
        self.sc = preprocessing.MinMaxScaler()
        self.sc.fit(np.concatenate([upper,lower]))


    def input_only(self, img_names):
        data = []
        for i in img_names:
            data.append(self.inputs[i])
            print(i)
            d = self.affs[1][1][i]
            print(d)
        return np.array(data)

    def next_input(self, aff):
        inputs = np.zeros([self.num_samples_per_class, self.dim_input])
        keys = []
        valid_keys, aff_data = self.affs[aff]
        samples = self.rand.choice(len(valid_keys), self.num_samples_per_class, replace=False)
        for j in range(self.num_samples_per_class):
            key = valid_keys[samples[j]]
            keys.append(key)
            inputs[j] = self.inputs[key]
        return keys, inputs

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for t in range(self.batch_size):
            a = self.rand.choice(len(self.affs))
            valid_keys, aff_data = self.affs[a]
            s = self.rand.uniform(-1.0, 1.0, [self.dim_latent, self.dim_output])
            i = self.rand.normal(0, 0.34, self.dim_output)
            samples = self.rand.choice(len(valid_keys), self.num_samples_per_class, replace=False)
            for j in range(self.num_samples_per_class):
                key = valid_keys[samples[j]]
                init_inputs[t,j] = self.inputs[key]
                #data = np.array([(aff_data[key][-1]-self.scale)/self.scale])*self.dirs
                data = np.array([aff_data[key][-1]]) * self.dirs
                tf_data = np.matmul(data,s) + i
                #tf_data = np.matmul(data,s)# + i
                #if np.min(np.sum(data,axis=0)) < m0:
                outputs[t,j] = self.sc.transform(tf_data)
        return init_inputs, outputs

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
