import time
import cv2 as cv
import sklearn.cluster
from sklearn.linear_model import LinearRegression
import scipy
from sklearn.decomposition import PCA
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
import os.path

class Affordances:
    def __init__(self, CLUSTER, inputs, train, exclude, samples, batchsz, k_shot, k_qry, dim_out, grasp):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.CLUSTER = CLUSTER
        self.grasp_pos = grasp
        #self.px_to_cm = 1.0/5.0 #7.6
        #self.cm_to_std = [1.33/42.0,1.0/42.0] # Standardize 480x640 image dims
        self.rand = RandomState(222)
        self.sample_size = samples

        # Load image features
        with open(os.path.expanduser("~") + "/data/cropped/tt_pca_out.pkl", 'rb') as handle:
            self.aff_pts = pickle.load(handle)      #dict(category) = [img1, img2, ...]

        keys = self.aff_pts.keys()
        categories = list(sorted(set([k.split("_")[0] for k in keys])))
        self.all_categories = [c for c in categories if c is not categories[exclude]]
        if train:
            valid_keys = [k for k in keys if not k.startswith(categories[exclude])]
        else:
            valid_keys = [k for k in keys if k.startswith(categories[exclude])]
        self.neg_keys = list(sorted(set([k for k in keys if not k.startswith(categories[exclude])])))
        self.valid_keys = list(sorted(set(valid_keys)))
        self.valid_objs = list(sorted(set([k.split("_00")[0] for k in self.valid_keys])))

        print("Categories: " + str(categories))
        self.exclude = exclude
        if exclude >= 0:
            if train:
                print("Excluding category '" + str(categories[exclude]) + "'")
            else:
                print("Testing on category '" + str(categories[exclude]) + "'")

        self.batch_size = batchsz
        self.dim_output = dim_out

        if k_qry == -1:
            self.num_objs_per_batch = len([o for o in self.valid_objs if o.startswith(categories[exclude])])
        else:
            self.num_objs_per_batch = k_shot + k_qry

        self.large_categories = []
        obj_counts = [o.split("_")[0] for o in self.valid_objs]
        objs = list(sorted(set(obj_counts)))
        self.cat_objs = dict()
        for o in range(len(objs)):
            self.cat_objs[objs[o]] = sorted([k for k in self.valid_objs if k.startswith(objs[o])])
            if obj_counts.count(objs[o]) >= self.num_objs_per_batch:
                self.large_categories.append(objs[o])
        #self.categories = list(sorted(set([k1.split("_")[0] for k1 in all_objs if sum([o.startswith(k1.split("_")[0]) for o in all_objs]) >= self.num_samples_per_class])))
        print(self.large_categories)

        all_vals = self.normalize_outputs()
        self.val_range = [np.min(all_vals,axis=0), np.max(all_vals,axis=0)]
        self.scale = preprocessing.StandardScaler()
        self.scale.fit(all_vals)
        print("Scaler: " + str(self.scale.scale_) + ", " + str(self.scale.mean_))

    def select_keys(self):
        pos_keys,pos_affs,neg_keys = [],[],[]
        #if len(self.categories) == 0:
        c = self.rand.choice(len(self.large_categories), self.batch_size, replace=True)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            # Get set of negative examples for img classification
            p_keys,n_keys = [],[]
            cat = self.large_categories[c[t]]
            pos_objs = self.cat_objs[cat]
            k = self.rand.choice(len(pos_objs), self.num_objs_per_batch, replace=False)
            for n in range(self.num_objs_per_batch):
                sample_keys = list([key for key in self.valid_keys if key.startswith(pos_objs[k[n]])])
                sk = self.rand.choice(len(sample_keys), self.sample_size, replace=False)
                for s in range(self.sample_size):
                    p_keys.append(sample_keys[sk[s]])

            neg_cands = [n for n in self.neg_keys if not n.startswith(cat)]
            neg_cats = self.rand.choice(len(neg_cands), self.sample_size * self.num_objs_per_batch, replace=False)
            for i in range(self.sample_size * self.num_objs_per_batch):
                n_keys.append(neg_cands[neg_cats[i]])
            neg_keys.append(n_keys)
            pos_keys.append(p_keys)
        return pos_keys, neg_keys

    def normalize_outputs(self):
        expanded = []
        for k in self.neg_keys:
                var_ratio, c1, c2 = self.aff_pts[k] #.split("_label")[0]]
                x = np.sqrt((c2[0]-c1[0])**2.0 + (c2[1]-c1[1])**2.0) 
                y = x * var_ratio[1] / var_ratio[0]
                expanded.append(np.array([x,y]))
                expanded.append(np.array([x,-y]))
        return np.stack(expanded)

    def expand_outputs(self):
        expanded = []
        keys = [k for k in list(self.aff_pts.keys()) if self.aff_pts[k] is not None]
        for k in keys:
            for c in self.all_categories:
                if k.startswith(c):
                    var_ratio, c1, c2 = self.aff_pts[k]
                    #align_normal, var_ratio, c1, c2, c3, c4 = self.aff_pts[k]
                    x_len = np.sqrt((c2[0]-c1[0])**2.0 + (c2[1]-c1[1])**2.0) 
                    expanded.append(np.array([x_len,0]))
                    new_r = math.sqrt(2*(450.0**2.0))/4.0
                    for tf_a in [np.pi, np.pi/2.0, 0.0, -np.pi/2.0]:
                        ratio = var_ratio[1] / var_ratio[0]
                        x = x_len + (new_r * math.sin(tf_a))
                        y = (x_len * ratio * np.sign(math.cos(tf_a))) + (new_r * ratio * math.cos(tf_a))
                        expanded.append(np.array([x,y]))
                        expanded.append(np.array([x,-y]))
        return np.stack(expanded)

    def next(self):
        pos_keys, neg_keys = self.select_keys()
        outputs = np.zeros([self.batch_size, self.num_objs_per_batch * self.sample_size, self.dim_output])
        # Each "batch" is an object class
        for t in range(self.batch_size):
            output_list = []
            tf = np.array([self.rand.uniform(-1,1), self.rand.uniform(-1,1)])
            scale = self.rand.uniform(0.5, 2.0)
            # Number of objects per class
            for c in range(self.num_objs_per_batch):
                # Number of images per object
                dims = []
                pos = pos_keys[t][c*self.sample_size:]
                for n in range(self.sample_size):
                    var_ratio, c1, c2 = self.aff_pts[pos[n]] #.split("_label")[0]]
                    x = np.sqrt((c2[0]-c1[0])**2.0 + (c2[1]-c1[1])**2.0) 
                    y = x * var_ratio[1] / var_ratio[0]
                    dims.append(self.scale.transform(np.array([x,y * np.sign(tf[1])]).reshape(1,-1)).squeeze())
                med_dim = np.median(np.stack(dims),axis=0)
                out = med_dim + tf
                for i in range(self.sample_size):
                    output_list.append(out)

            outputs[t] = np.stack(output_list)
        return pos_keys, neg_keys, outputs

    def project_tf(self, name_spt, tf, scale=-1):
        spt_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        qry_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        neg_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        outputs = np.zeros([self.num_samples_per_class * self.sample_size, self.dim_output])
        spt_output_list,qry_output_list,qry_input_list,spt_input_list,negative_list,qry_keys,cart_out = [],[],[],[],[],[],[]

        # Get spt/qry object category name
        cat = name_spt.split("_")[0]

        # Get list of other object categories
        neg_cats = self.rand.choice(len(self.all_categories), self.num_samples_per_class, replace=True)
        while self.exclude in neg_cats:
            neg_cats = self.rand.choice(len(self.all_categories), self.num_samples_per_class, replace=True)

        # Get list of objects within spt/qry category
        obj_keys = list(sorted(set([k.split("_00")[0] for k in self.valid_keys if k.startswith(cat)])))

        # Get negative examples
        negative_keys = list([key for key in self.all_keys if not key.startswith(cat)])
        nk = self.rand.choice(len(negative_keys), self.sample_size, replace=False)

        for n in range(len(obj_keys)):
            # Get positive examples
            sample_keys = list([key for key in self.valid_keys if key.startswith(obj_keys[n])])
            if self.sample_size > len(sample_keys):
                pdb.set_trace()
            sk = self.rand.choice(len(sample_keys), self.sample_size, replace=False)
            for s in range(self.sample_size):
                #neg_fts = self.inputs[negative_keys[nk[s]]]
                #negative_list.append(neg_fts.reshape((1024,14,14)).transpose())
                im = sample_keys[sk[s]]
                #fts = self.inputs[im]
                if im.startswith(name_spt):
                    pose = np.matrix([tf[0]])
                    tf_xy = np.array(np.dot(pose,self.grasp_pos[1])) #.reshape(1,-1)
                    sc_div = np.stack(self.val_range) / tf_xy
                    sc_min = np.max(np.min(sc_div,axis=0))
                    sc_max = np.min(np.max(sc_div,axis=0))
                    sc_ivl = (sc_max - sc_min) / 19.0
                    if scale == -1:
                        out = self.scale.transform((100*tf_xy) / self.px_to_cm) 
                    else:
                        out = self.scale.transform((sc_min + (sc_ivl * scale)) * tf_xy.reshape(1,-1)) 
                    #inv = self.scale.inverse_transform(out) * self.px_to_cm / 100 #self.apply_tf_wrt_grasp(im, tf)
                    #inv_xy = np.array(np.dot(inv, np.linalg.pinv(self.grasp_pos[1]))).reshape(1,-1)

                    spt_output_list.append(out)
                    #spt_output_list.append(out)
                    #spt_input_list.append(fts.reshape((1024,14,14)).transpose())
                    qry_output_list.append(np.matrix([0,0]))
                    #qry_input_list.append(fts.reshape((1024,14,14)).transpose())
                else:
                    qry_output_list.append(np.matrix([0,0]))
                    #qry_input_list.append(fts.reshape((1024,14,14)).transpose())
                qry_keys.append(sample_keys[sk[s]])
        spt_inputs = None #np.stack(spt_input_list)
        qry_inputs = None #np.stack(qry_input_list)
        neg_inputs = np.stack(negative_list)
        spt_outputs = np.stack(spt_output_list)
        qry_outputs = np.stack(qry_output_list)
        return spt_inputs, qry_inputs, neg_inputs, spt_outputs, qry_outputs, qry_keys,(sc_min + (sc_ivl * scale))

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
