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
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/polar/reduced_fts_0.95.pkl"
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_polar_fts.pkl"
        ## Load VGG features for all images
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        categories = list(sorted(set([k.split("_")[0] for k in self.inputs.keys()])))
        print("Categories: " + str(categories))
        if train:
            print("Excluding category '" + str(categories[exclude]) + "'")
        else:
            print("Testing on category '" + str(categories[exclude]) + "'")

        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/reduced_fts_0.95.pkl"
        training_keys = []
        all_vals = []
        self.valid_keys = []
        for aff in range(2,7):
            aff_loc = "/home/tesca/data/part-affordance-dataset/features/polar_aff_" + str(aff) + "_positions.pkl"
            with open(aff_loc, 'rb') as handle:
                aff_data = pickle.load(handle)      #dict(category) = [img1, img2, ...]
            keys = list(sorted(aff_data.keys()))
            train_valid_keys = [k for k in keys if (aff_data[k][-1] is not None) and (not k.startswith(categories[exclude]))]
            #train_valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (k.endswith("01")) and (not k.split("_00")[0].endswith("01"))]
            #train_valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (not k.split("_00")[0].endswith("01"))]
            training_keys += train_valid_keys
            test_valid_keys = [k for k in keys if (aff_data[k][-1] is not None) and (k.startswith(categories[exclude]))]
            #test_valid_keys = [k for k in keys if (aff_data[k][-1] is not None) and (k.split("_00")[0].endswith("01"))]
            #test_valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (k.endswith("01")) and (k.split("_00")[0].endswith("01"))]
            #test_valid_keys = [k for k in keys if (not None in aff_data[k][-1]) and (k.split("_00")[0].endswith("01"))]
            if train:
                valid_keys = train_valid_keys
            else:
                valid_keys = test_valid_keys
            self.valid_keys += valid_keys
            vals = [aff_data[k][-1] for k in valid_keys]
            #vals = [aff_data[k][-1]*self.dirs for k in valid_keys]
            val_m = np.matrix(vals) 
            if val_m.shape[1] > 0:
                all_vals.append(val_m)
            #vals_max.append(np.max(val_m,axis=0))
            #vals_min.append(np.min(val_m,axis=0))
            self.affs.append([valid_keys, aff_data])
            data_count += len(valid_keys)

        self.valid_keys = list(sorted(set(self.valid_keys)))
        self.classes = list(sorted(set([k.split("_00")[0] for k in self.inputs.keys()])))
        self.num_classes = len(set([k.split("_00")[0] for k in self.valid_keys]))
        
        inputs = []
        for k in training_keys:
            inputs.append(self.inputs[k])
        inputs = np.array(inputs)
        self.input_scale = preprocessing.StandardScaler()
        self.input_scale.fit(inputs)
        print(str(data_count) + " image/aff pairs loaded")
        self.num_samples_per_class = k_shot + k_qry
        self.batch_size = batchsz
        self.dim_output = dim_out
        self.dim_latent = 2
        self.dim_input = len(list(self.inputs.values())[0])

        #all_max = np.max(np.array(vals_max),axis=0)
        #all_min = np.min(np.array(vals_min),axis=0)
        #max_mul = np.ones((self.dim_latent, self.dim_output))
        #max_val = np.max([abs(all_max), abs(all_min)],axis=0)
        #upper = np.matmul(max_val, max_mul) + 0.68
        #lower = np.matmul(max_val, np.negative(max_mul)) - 0.68
        #self.sc = preprocessing.MinMaxScaler(feature_range=(-1,1))
        #self.sc.fit(np.concatenate([upper,lower]))
        #self.sc1 = preprocessing.StandardScaler() #feature_range=(-1,1))
        self.sc1 = preprocessing.MinMaxScaler(feature_range=(0,1))
        #vals_m = np.concatenate(all_vals)
        #all_vals_max = np.matmul(vals_m,max_mul)
        #all_vals_min = np.matmul(vals_m,np.negative(max_mul))
        tf_data = self.sc1.fit_transform(np.concatenate(all_vals))
        self.output_std = np.std(tf_data,axis=0)
        self.output_mean = np.mean(tf_data,axis=0)
        #self.sc1.fit(np.concatenate([all_vals_max,all_vals_min]))
        #self.sc2 = preprocessing.StandardScaler()
        #self.sc2.fit(np.concatenate([all_vals_max,all_vals_min]))


    def input_only(self, img_names):
        data = []
        for i in img_names:
            data.append(self.input_scale.transform(self.inputs[i].reshape(1,-1)).squeeze(0))
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
            inputs[j] = self.input_scale.transform(self.inputs[key].reshape(1,-1))
        return keys, inputs

    def next(self):
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class])
        objects = list(set([k.split("_00")[0] for k in self.valid_keys]))
        o = self.rand.choice(len(objects), self.batch_size, replace=False)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            obj = objects[o[t]]
            for c in range(len(self.classes)):
                if obj == self.classes[c]:
                    out = c
            obj_keys = list(set([k for k in self.valid_keys if k.startswith(obj)]))
            k = self.rand.choice(len(obj_keys), self.num_samples_per_class, replace=False)
            for n in range(self.num_samples_per_class):
                init_inputs[t,n] = self.input_scale.transform(self.inputs[obj_keys[k[n]]].reshape(1,-1))
                outputs[t,n] = out
        return init_inputs, outputs

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
