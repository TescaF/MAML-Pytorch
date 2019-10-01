from scipy.stats import norm
import sklearn
import math
import pdb
import itertools
import sys
import  os.path
import  numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing, decomposition
import torchvision.models as models
import json
from PIL import Image
import pickle
from numpy.random import RandomState


class Affordances:
    def __init__(self, mode, train, exclude, samples, batchsz, k_shot, k_qry, dim_out):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.train = train
        self.rand = RandomState(222)
        self.affs = []
        self.sample_size = samples
        self.base_loc = "/home/tesca/data/part-affordance-dataset/tools/"
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/" + mode + "_resnet_pool_fts-14D.pkl"
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_fts.pkl"
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_polar_fts.pkl"
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        categories = list(sorted(set([k.split("_")[0] for k in self.inputs.keys()])))
        print("Categories: " + str(categories))
        if exclude >= 0:
            if train:
                print("Excluding category '" + str(categories[exclude]) + "'")
            else:
                print("Testing on category '" + str(categories[exclude]) + "'")

        self.valid_keys, training_keys, all_vals = [],[],[]
        for aff in range(2,7):
            aff_loc = "/home/tesca/data/part-affordance-dataset/features/" + mode + "_aff_" + str(aff) + "_positions.pkl"
            with open(aff_loc, 'rb') as handle:
                aff_data = pickle.load(handle)      #dict(category) = [img1, img2, ...]

            keys = list(sorted(aff_data.keys()))
            if exclude >= 0:
                train_valid_keys = [k for k in keys if (aff_data[k][-1] is not None) and (not k.startswith(categories[exclude]))]
                test_valid_keys = [k for k in keys if (aff_data[k][-1] is not None) and (k.startswith(categories[exclude]))]
            else:
                test_valid_keys = train_valid_keys = [k for k in keys if aff_data[k][-1] is not None]
            training_keys += train_valid_keys

            if train:
                valid_keys = train_valid_keys    
            else:
                valid_keys = test_valid_keys
            self.valid_keys += valid_keys
            vals_m = np.matrix([aff_data[k][-1] for k in valid_keys])
            if vals_m.shape[1] > 0:
                all_vals.append(vals_m)
            self.affs.append([valid_keys, aff_data])

        self.valid_keys = list(sorted(set(self.valid_keys)))
        self.classes = list(sorted(set([k.split("_00")[0] for k in self.valid_keys])))
        self.num_classes = len(set([k.split("_00")[0] for k in self.valid_keys]))
        
        self.num_samples_per_class = k_shot + k_qry
        self.batch_size = batchsz
        self.dim_output = dim_out
        self.dim_input = len(list(self.inputs.values())[0])
        self.output_scale = preprocessing.MinMaxScaler(feature_range=(-1,1))
        val_range = np.matrix([[0,0],[480,640]])
        self.output_scale.fit(val_range)
        #self.output_scale.fit(np.concatenate(all_vals)[:,self.dim_output])
        self.center = np.array([240,320,0])
        all_objs = list(sorted(set([k.split("_00")[0] for k in self.valid_keys])))
        self.categories = list(sorted(set([k1.split("_")[0] for k1 in all_objs if sum([o.startswith(k1.split("_")[0]) for o in all_objs]) >= self.num_samples_per_class])))
        print(self.categories)

    def convert_to_pdf(self, data, dists):
        inputs = []
        tf_data = np.zeros_like(data)
        c = list(itertools.product(*[range(i) for i in data.shape]))
        for idx in c:
            tf_data[idx] = norm.pdf(data[idx], dists[idx[-1]][0], dists[idx[-1]][1])
        return tf_data

    def next(self):
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, 14,14,1024])
        neg_inputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, 14,14,1024])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, self.dim_output])
        selected_keys,pdf_data = [],[]
        c = self.rand.choice(len(self.categories), self.batch_size, replace=True)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            # Get set of negative examples for img classification
            neg_cats = self.rand.choice(len(self.categories), self.num_samples_per_class, replace=True)
            while not (c[t] in neg_cats):
                neg_cats = self.rand.choice(len(self.categories), self.num_samples_per_class, replace=True)

            output_list,input_list,negative_list,sel_keys,cart_out = [],[],[],[],[]
            cat = self.categories[c[t]]
            valid_affs = [a for a in range(len(self.affs)) if any([o.startswith(cat) for o in self.affs[a][0]])]
            aff_num = self.rand.choice(len(valid_affs))
            valid_keys, aff_data = self.affs[valid_affs[aff_num]]
            obj_keys = list(sorted(set([k.split("_00")[0] for k in valid_keys if k.startswith(cat)])))
            tf_a = self.rand.uniform(-np.pi/4.0,np.pi/4.0)
            tf_r = self.rand.uniform(-0.5,0.5)
            tf_z = self.rand.uniform(-0.5,0.5)
            k = self.rand.choice(len(obj_keys), self.num_samples_per_class, replace=False)
            for n in range(self.num_samples_per_class):
                negative_keys = list([key for key in self.valid_keys if key.startswith(self.categories[neg_cats[n]])])
                nk = self.rand.choice(len(negative_keys), self.sample_size, replace=False)
                sample_keys = list([key for key in valid_keys if key.startswith(obj_keys[k[n]])])
                sk = self.rand.choice(len(sample_keys), self.sample_size, replace=False)
                for s in range(self.sample_size):
                    neg_fts = self.inputs[negative_keys[nk[s]]]
                    negative_list.append(neg_fts.reshape((1024,14,14)).transpose())
                    sel_keys.append(sample_keys[sk[s]])
                    fts = self.inputs[sample_keys[sk[s]]]
                    input_list.append(fts.reshape((1024,14,14)).transpose())
                    pt1 = np.array(aff_data[sample_keys[sk[s]]][-1])
                    pt = pt1 - self.center
                    r1 = np.sqrt(pt[0]**2 + pt[1]**2)
                    r = r1 * (1+tf_r)
                    a = math.atan2(pt[1],pt[0]) + tf_a
                    tf_out_x = self.center[0] + (r * math.cos(a))
                    tf_out_y = self.center[1] + (r * math.sin(a))
                    tf_out_z = pt[2] * (1+tf_z)
                    out = self.output_scale.transform(np.array([tf_out_x,tf_out_y,tf_out_z])[:self.dim_output].reshape(1,-1)).squeeze()[:self.dim_output]
                    #if not self.train:
                    #    pdb.set_trace()
                    #out = self.output_scale.transform(np.array([aff_data[sample_keys[sk[s]]][-1][:self.dim_output]]).reshape(1,-1)).squeeze()
                    output_list.append(out)
                    #output_list.append([tf_out_x,tf_out_y])
            init_inputs[t] = np.stack(input_list)
            neg_inputs[t] = np.stack(negative_list)
            outputs[t] = np.stack(output_list)
            selected_keys.append(sel_keys)
        #pdf_data = np.concatenate(pdf_data)
        stats = []
        #for i in range(pdf_data.shape[-1]):
        #    stats.append(norm.fit(pdf_data[:,i]))
        return init_inputs, neg_inputs, outputs, selected_keys, stats

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
