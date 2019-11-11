import sklearn.cluster
from sklearn.linear_model import LinearRegression
import scipy
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
        self.px_to_cm = 1.0/7.6
        self.cm_to_std = [1.33/42.0,1.0/42.0] # Standardize 480x640 image dims
        self.train = train
        self.rand = RandomState(222)
        self.affs = []
        self.sample_size = samples
        self.aff_dir = "/home/tesca/data/part-affordance-dataset/tools/"
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/" + mode + "_resnet_pool_fts-14D.pkl"
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_fts.pkl"
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/resnet_polar_fts.pkl"
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        categories = list(sorted(set([k.split("_")[0] for k in self.inputs.keys()])))
        self.all_categories = categories
        self.all_keys = list(sorted(self.inputs.keys()))
        print("Categories: " + str(categories))
        self.exclude = exclude
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

    def project_tf(self, name_spt, tf):
        spt_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        qry_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        neg_inputs = np.zeros([self.num_samples_per_class * self.sample_size, 14,14,1024])
        outputs = np.zeros([self.num_samples_per_class * self.sample_size, self.dim_output])
        spt_output_list,qry_output_list,qry_input_list,spt_input_list,negative_list,sel_keys,cart_out = [],[],[],[],[],[],[]

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
                neg_fts = self.inputs[negative_keys[nk[s]]]
                negative_list.append(neg_fts.reshape((1024,14,14)).transpose())
                sel_keys.append(sample_keys[sk[s]])
                fts = self.inputs[sample_keys[sk[s]]]
                if sample_keys[sk[s]].startswith(name_spt):
                    out = self.apply_tf_wrt_grasp(sample_keys[sk[s]], tf)
                    im = sample_keys[sk[s]]
                    spt_output_list.append(out)
                    spt_input_list.append(fts.reshape((1024,14,14)).transpose())
                else:
                    qry_output_list.append(np.matrix([0,0]))
                    qry_input_list.append(fts.reshape((1024,14,14)).transpose())
        spt_inputs = np.stack(spt_input_list)
        qry_inputs = np.stack(qry_input_list)
        neg_inputs = np.stack(negative_list)
        spt_outputs = np.stack(spt_output_list)
        qry_outputs = np.stack(qry_output_list)
        return spt_inputs, qry_inputs, neg_inputs, spt_outputs, qry_outputs, sel_keys

        # Get support images
        # For each support image, get centroid and direction of grasp
        # Get tf wrt grasp
        # Add to outputs
        # Get negative images
        # Get query images

    def apply_tf_wrt_grasp(self, img_name, tf):
        label = scipy.io.loadmat(self.aff_dir + img_name.split("_00")[0] + "/" + img_name + "_label.mat")
        img_affs = label['gt_label']

        grasp_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] == 1])
        aff_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] > 1])
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=5).fit_predict(aff_pts)

        # Get edge normal
        reg = LinearRegression().fit(grasp_pts[:,0], grasp_pts[:,1])
        normal = math.atan(reg.coef_) #+ math.pi/2.0

        ## Pick TF direction that minimizes distance from aff points mean
        mean_aff = np.multiply(np.mean(aff_pts,axis=0) * self.px_to_cm, self.cm_to_std) - 1.0
        mean_grasp = np.multiply(np.mean(grasp_pts,axis=0) * self.px_to_cm, self.cm_to_std) - 1.0
        comp_ang = math.atan2(mean_aff[0,1]-mean_grasp[0,1],mean_aff[0,0]-mean_grasp[0,0])
        cand_ang = np.array([normal, normal-np.pi, normal+np.pi])
        align_normal = cand_ang[np.argmin(np.absolute(cand_ang - comp_ang))]

        # Project TF x and y
        tf_x = tf.pose.position.x 
        tf_y = tf.pose.position.z
        tf_z = tf.pose.position.y
        tf_r = math.sqrt(tf_x**2.0 + tf_y**2.0)
        tf_ang = math.atan2(tf_y, tf_x)
        a = tf_ang + align_normal
        ee = [math.cos(a) * tf_r * 100.0 * self.cm_to_std[0], math.sin(a) * tf_r * 100.0 * self.cm_to_std[1]]
        return ee

    def inverse_project(self, img_name, pt):
        label = scipy.io.loadmat(self.aff_dir + img_name.split("_00")[0] + "/" + img_name + "_label.mat")
        img_affs = label['gt_label']

        grasp_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] == 1])
        aff_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] > 1])

        # Get edge normal
        reg = LinearRegression().fit(grasp_pts[:,0], grasp_pts[:,1])
        normal = math.atan(reg.coef_) #+ math.pi/2.0
        mean_grasp = np.multiply(np.mean(grasp_pts,axis=0) * self.px_to_cm, self.cm_to_std) - 1.0
        mean_aff = np.multiply(np.mean(aff_pts,axis=0) * self.px_to_cm, self.cm_to_std) - 1.0
        comp_ang = math.atan2(mean_aff[0,1]-mean_grasp[0,1],mean_aff[0,0]-mean_grasp[0,0])
        cand_ang = np.array([normal, normal-np.pi, normal+np.pi])
        align_normal = cand_ang[np.argmin(np.absolute(cand_ang - comp_ang))]

        inv_pt = np.divide(pt, self.cm_to_std)/100.0
        pt_r = np.linalg.norm(inv_pt)
        pt_ang = math.atan2(inv_pt[1], inv_pt[0])
        a = align_normal - pt_ang
        ee = [math.cos(a) * pt_r, math.sin(a) * pt_r]

        return ee

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
