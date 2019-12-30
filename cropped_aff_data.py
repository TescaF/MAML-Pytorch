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
    def __init__(self, CLUSTER, inputs, mode, train, exclude, samples, batchsz, k_shot, k_qry, dim_out, grasp):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.CLUSTER = CLUSTER
        self.grasp_pos = grasp
        self.inputs = inputs
        self.cm_to_std = [1.33/42.0,1.0/42.0] # Standardize 480x640 image dims
        self.rand = RandomState(222)
        self.affs = []
        self.sample_size = samples
        self.aff_dir = os.path.expanduser("~") + "/data/cropped/"
        self.tf = cv.getPerspectiveTransform(np.array([[445,170],[525,375],[125,375],[153,170]],dtype="float32"),np.array([[525,170],[525,425],[125,425],[125,170]],dtype="float32"))
        self.offset = np.array([[ 1 , 0 , -110], [ 0 , 1 , -60], [ 0 , 0 ,    1    ]])

        # Load image features
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/" + mode + "_resnet_pool_fts-14D.pkl"
        if self.inputs is None:
            with open(fts_loc, 'rb') as handle:
                self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        categories = list(sorted(set([k.split("_")[0] for k in self.inputs.keys()])))
        self.all_categories = [c for c in categories if c is not categories[exclude]]
        self.all_keys = list(sorted(self.inputs.keys()))
        print("Categories: " + str(categories))
        self.exclude = exclude
        if exclude >= 0:
            if train:
                print("Excluding category '" + str(categories[exclude]) + "'")
            else:
                print("Testing on category '" + str(categories[exclude]) + "'")

        # Load image affordance locations
        self.valid_keys, training_keys, all_vals = [],[],[]
        with open(os.path.expanduser("~") + "/data/cropped/tt-1.pkl", 'rb') as handle:
            self.aff_pts = pickle.load(handle)      #dict(category) = [img1, img2, ...]
        with open(os.path.expanduser("~") + "/data/cropped_grasp_positions.pkl", 'rb') as handle:
            self.grasps = pickle.load(handle)      #dict(category) = [img1, img2, ...]
        for aff in range(2,7):
            aff_loc = os.path.expanduser("~") + "/data/cropped_aff_" + str(aff) + "_positions.pkl"
            with open(aff_loc, 'rb') as handle:
                aff_data = pickle.load(handle)      #dict(category) = [img1, img2, ...]
            aff_keys = list(aff_data.keys())
            for k in aff_keys:
                if k not in self.all_keys:
                    aff_data.pop(k, None)

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
        self.batch_size = batchsz
        self.dim_output = dim_out
        self.output_scale = preprocessing.MinMaxScaler(feature_range=(-1,1))
        self.output_scale.fit(np.matrix([[0,0],[450,450]]))
        all_objs = list(sorted(set([k.split("_00")[0] for k in self.valid_keys])))
        if k_qry == -1:
            self.num_samples_per_class = len([o for o in all_objs if o.startswith(categories[exclude])])
        else:
            self.num_samples_per_class = k_shot + k_qry
        self.categories = list(sorted(set([k1.split("_")[0] for k1 in all_objs if sum([o.startswith(k1.split("_")[0]) for o in all_objs]) >= self.num_samples_per_class])))
        print(self.categories)

    def select_keys(self):
        pos_keys,pos_affs,neg_keys = [],[],[]
        c = self.rand.choice(len(self.categories), self.batch_size, replace=True)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            # Get set of negative examples for img classification
            p_keys,p_affs,n_keys = [],[],[]
            cat = self.categories[c[t]]
            neg_cands = [n for n in self.all_categories if not n == cat]
            neg_cats = self.rand.choice(len(neg_cands), self.num_samples_per_class, replace=True)
            valid_affs = [a for a in range(len(self.affs)) if any([o.startswith(cat) for o in self.affs[a][0]])]
            valid_keys, aff_data = self.affs[valid_affs[self.rand.choice(len(valid_affs))]]
            obj_keys = list(sorted(set([k.split("_00")[0] for k in valid_keys if k.startswith(cat)])))
            k = self.rand.choice(len(obj_keys), self.num_samples_per_class, replace=False)
            for n in range(self.num_samples_per_class):
                negative_keys = list([key for key in self.all_keys if key.startswith(neg_cands[neg_cats[n]])])
                sample_keys = list([key for key in valid_keys if key.startswith(obj_keys[k[n]])])
                sk = self.rand.choice(len(sample_keys), self.sample_size, replace=False)
                for s in range(self.sample_size):
                    n_keys.append(negative_keys[sk[s]])
                    p_keys.append(sample_keys[sk[s]])
                    p_affs.append(aff_data[p_keys[-1]][-1])
            neg_keys.append(n_keys)
            pos_keys.append(p_keys)
            pos_affs.append(p_affs)
        return pos_keys, pos_affs, neg_keys

    def next(self):
        pos_keys, pos_affs, neg_keys = self.select_keys()
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, 14,14,1024])
        neg_inputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, 14,14,1024])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class * self.sample_size, self.dim_output])
        # Each "batch" is an object class
        for t in range(self.batch_size):
            output_list,input_list,negative_list = [],[],[]
            # use the pca version to calculate angle
            tf_a = self.rand.uniform(-np.pi,np.pi)
            max_r = math.sqrt(2*(450.0**2.0))/2.0
            tf_r = self.rand.uniform(0,0.25)
            for n in range(self.num_samples_per_class * self.sample_size):
                negative_list.append(self.inputs[neg_keys[t][n]].reshape((1024,14,14)).transpose())
                input_list.append(self.inputs[pos_keys[t][n]].reshape((1024,14,14)).transpose())
                #center = self.grasps[pos_keys[t][n]]
                if pos_keys[t][n].split("_label")[0] in self.aff_pts.keys():
                    # Centers: grasp end, grasp absolute center, grasp center, tooltip
                    align_normal, c1, c2, c3, c4 = self.aff_pts[pos_keys[t][n].split("_label")[0]]
                else:
                    align_normal, grasp_diff, center = self.get_grasp_normal(pos_keys[t][n].split("_label")[0], grasp_aff=-1)
                    c1, c2, c3, c4 = center

                #pt = np.array(c4) - np.array(c3)
                #r1 = np.sqrt(pt[0]**2 + pt[1]**2)
                a = tf_a + align_normal
                new_r = max_r * tf_r

                tf_out_x = c4[0] + (new_r * math.cos(a))
                tf_out_y = c4[1] + (new_r * math.sin(a))

                '''name = pos_keys[t][n].split("_label")[0]
                img = cv.imread('/u/tesca/data/cropped/' + name + '_rgb.jpg')
                cv.circle(img,(int(pos_affs[t][n][1]),int(pos_affs[t][n][0])),5,[0,255,0])
                cv.circle(img,(int(center[2][1]),int(center[2][0])),5,[255,0,0])
                cv.circle(img,(int(tf_out_y),int(tf_out_x)),5,[0,0,255])
                cv.imwrite('/u/tesca/data/train_imgs/t' + str(t) + "_im-" + name +'.jpg', img)'''

                out = self.output_scale.transform(np.array([tf_out_x,tf_out_y]).reshape(1,-1)).squeeze()
                output_list.append(out)
            init_inputs[t] = np.stack(input_list)
            neg_inputs[t] = np.stack(negative_list)
            outputs[t] = np.stack(output_list)
        return init_inputs, neg_inputs, outputs, pos_keys, neg_keys

    def project_tf(self, name_spt, tf):
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
                neg_fts = self.inputs[negative_keys[nk[s]]]
                negative_list.append(neg_fts.reshape((1024,14,14)).transpose())
                im = sample_keys[sk[s]]
                fts = self.inputs[im]
                if im.startswith(name_spt):
                    out = self.apply_tf_wrt_grasp(im, tf)
                    spt_output_list.append(out)
                    spt_input_list.append(fts.reshape((1024,14,14)).transpose())
                    qry_output_list.append(np.matrix([0,0]))
                    qry_input_list.append(fts.reshape((1024,14,14)).transpose())
                else:
                    qry_output_list.append(np.matrix([0,0]))
                    qry_input_list.append(fts.reshape((1024,14,14)).transpose())
                qry_keys.append(sample_keys[sk[s]])
        spt_inputs = np.stack(spt_input_list)
        qry_inputs = np.stack(qry_input_list)
        neg_inputs = np.stack(negative_list)
        spt_outputs = np.stack(spt_output_list)
        qry_outputs = np.stack(qry_output_list)
        return spt_inputs, qry_inputs, neg_inputs, spt_outputs, qry_outputs, qry_keys

        # Get support images
        # For each support image, get centroid and direction of grasp
        # Get tf wrt grasp
        # Add to outputs
        # Get negative images
        # Get query images

    def get_grasp_normal(self, img_name, aff=-1, grasp_aff=1):
        if self.CLUSTER:
            label = scipy.io.loadmat(self.aff_dir + img_name + "_label.mat")
        else:
            label = scipy.io.loadmat(self.aff_dir + img_name.split("_00")[0] + "/" + img_name + "_label.mat")
        img_affs = cv.warpPerspective(label['gt_label'], np.dot(self.offset,self.tf), (450,450))

        if aff == -1:
            aff_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] > 1 and img_affs[i,j] < 7])
        else:
            aff_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] == aff])
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=10).fit_predict(aff_pts)
        aff_clust = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
        ay, ax = np.median(aff_clust,axis=0).squeeze(0)

        if grasp_aff == -1:
            grasp_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] >= 1])
        else:
            grasp_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] == grasp_aff])
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
        grasp_clust = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
        cy, cx = np.median(grasp_clust,axis=0).squeeze(0)

        # Get edge normal
        pca = PCA(n_components=1)
        pca_tf = pca.fit_transform(np.stack(grasp_clust))
        normal = math.atan2(pca.components_[0,1],pca.components_[0,0])
        grasp_reduced = pca.inverse_transform(pca_tf)
        comp_ang = math.atan2(ax-cx,ay-cy)
        cand_ang = np.array([normal, normal-np.pi, normal+np.pi])
        align_normal = cand_ang[np.argmin(np.absolute(cand_ang - comp_ang))]
        #align_normal = comp_ang

        # Get grasp end along normal dim
        e_dists = [math.sqrt((ay - p[0])**2.0 + (ax - p[1])**2.0) for p in grasp_reduced]
        e_idx = np.argmax(np.array(e_dists))
        ey, ex = np.array(grasp_reduced[e_idx])

        # Get grasp center along normal dim
        c_dists = [math.sqrt((cy - p[0])**2.0 + (cx - p[1])**2.0) for p in grasp_reduced]
        c_idx = np.argmin(np.array(c_dists))
        ncy, ncx = np.array(grasp_reduced[c_idx])

        # Get tooltip end along normal dim
        t_dists = [math.sqrt((ey - p[0])**2.0 + (ex - p[1])**2.0) for p in grasp_reduced]
        t_idx = np.argmax(np.array(t_dists))
        ty, tx = np.array(grasp_reduced[t_idx])

        grasp_diff = np.array(self.output_scale.transform(np.array([ncy,ncx]).reshape(1,-1)) - self.output_scale.transform(np.array([ey,ex]).reshape(1,-1))).squeeze(0)
        # Centers: grasp end, grasp absolute center, grasp center, tooltip
        return align_normal, grasp_diff, [(ey,ex),(cy,cx),(ncy,ncx),(ty,tx)]

    def apply_tf_wrt_grasp(self, img_name, tf):
        align_normal, grasp_diff, center = self.get_grasp_normal(img_name)

        # Project TF x and y
        '''tf_x = tf.pose.position.x 
        tf_y = -tf.pose.position.z
        tf_z = tf.pose.position.y'''
        if isinstance(tf, list):
            pose = np.matrix([tf[0]])
        else:
            pose = np.matrix([[tf.pose.position.x,tf.pose.position.y,tf.pose.position.z]])
        tf_x, tf_y = np.array(np.dot(pose,self.grasp_pos[1])).squeeze(0).tolist()
        tf_r = math.sqrt(tf_x**2.0 + tf_y**2.0)
        tf_ang = math.atan2(tf_y, tf_x)
        a = tf_ang + align_normal
        ee_std = [math.cos(a) * tf_r * 100.0 * self.cm_to_std[0], math.sin(a) * tf_r * 100.0 * self.cm_to_std[1]]
        if self.grasp_pos[0] == "end":
            ee_std -= grasp_diff

        # Testing
        '''for pt in aff_clust:
            img[pt[0,0],pt[0,1]] = (0,0,0)
        for pt in grasp_clust:
            img[pt[0,0],pt[0,1]] = (255,255,255)'''
        ee_inv = self.inverse_project(img_name, ee_std)
        #print(img_name + ": " + str([tf_x,tf_y]) + " -> " + str(ee_inv))
        '''img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + img_name + '_center.jpg')
        img2 = cv.imread('/home/tesca/data/part-affordance-dataset/tools/' + img_name.split("_00")[0] + '/' + img_name + '_rgb.jpg')
        ee = self.output_scale.inverse_transform(np.array(ee_std-centering).reshape(1,-1)).squeeze(0)
        ee1 = self.output_scale.inverse_transform(np.array(ee_std+grasp_diff).reshape(1,-1)).squeeze(0)
        cv.circle(img,(int(ee[1]),int(ee[0])),5,[255,255,0])
        #cv.circle(img,(int(ee1[1]),int(ee1[0])),5,[255,0,0])
        cv.imshow("img",img)
        cv.waitKey(0)'''
        '''ee = self.output_scale.inverse_transform(np.array(ee_std+centering).reshape(1,-1)).squeeze(0)
        ee1 = self.output_scale.inverse_transform(np.array(ee_std+grasp_diff+centering).reshape(1,-1)).squeeze(0)
        cv.circle(img2,(int(ee[1]),int(ee[0])),5,[255,255,0])
        cv.circle(img2,(int(ee1[1]),int(ee1[0])),5,[255,0,0])
        cv.imshow("img",img2)
        cv.waitKey(0)'''
        return ee_std

    def inverse_project(self, img_name, pt):
        align_normal, grasp_diff, center = self.get_grasp_normal(img_name)

        if self.grasp_pos[0] == "end":
            pt1 = pt + grasp_diff
        #centering = self.output_scale.transform(np.array(center).reshape(1,-1)).squeeze()
        #pt += centering

        inv_pt = np.divide(pt1, self.cm_to_std)/100.0
        #ee_px = self.output_scale.inverse_transform(np.array(pt).reshape(1,-1)).squeeze(0)
        #inv_pt = (np.array(ee_px)-np.array(self.center[:2])) * self.px_to_cm / 100.0
        pt_r = math.sqrt(inv_pt[0]**2.0 + inv_pt[1]**2.0)
        pt_ang = math.atan2(inv_pt[1], inv_pt[0])
        a = pt_ang - align_normal  #- pt_ang
        ee = [math.cos(a) * pt_r, math.sin(a) * pt_r]

        # Testing
        #img2 = cv.imread('/home/tesca/data/part-affordance-dataset/tools/' + img_name.split("_00")[0] + '/' + img_name + '_rgb.jpg')
        '''img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + img_name + '_center.jpg')
        ee = self.output_scale.inverse_transform(np.array(pt).reshape(1,-1)).squeeze(0)
        cv.circle(img,(int(ee[1]),int(ee[0])),5,[255,255,0])
        cv.imshow("img",img)
        cv.waitKey(0)'''

        return ee

if __name__ == '__main__':
    IN = Affordances(5,3,3,2)
    data = IN.next()
    pdb.set_trace()
