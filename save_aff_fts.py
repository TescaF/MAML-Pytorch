import ast
import pdb
import sklearn.cluster
from sklearn.linear_model import LinearRegression
import scipy.io
import numpy as np
import sys
import  os.path
import math
from os import path
import cv2 as cv
import pickle 
from itertools import product
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models


class ImageProc:

    def __init__(self):
        self.base_dir = "/u/tesca/data/"
        self.im_dir = "/u/tesca/data/cropped/"
        self.tf = cv.getPerspectiveTransform(np.array([[445,170],[525,375],[125,375],[153,170]],dtype="float32"),np.array([[525,170],[525,425],[125,425],[125,170]],dtype="float32"))
        self.offset = np.array([[ 1 , 0 , -110], [ 0 , 1 , -60], [ 0 , 0 ,    1    ]])

    def get_img_data(self, data):
        label_tf = data
        grasp_pts_1 = [(i,j) for i in range(label_tf.shape[0]) for j in range(label_tf.shape[1]) if label_tf[i,j] == 1]
        grasp_pts_2 = [(i,j) for i in range(label_tf.shape[0]) for j in range(label_tf.shape[1]) if label_tf[i,j] == 7]
        if len(grasp_pts_1) == 0:
            grasp = grasp_pts_2
        else:
            grasp = grasp_pts_1
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=5).fit_predict(grasp)
        grasp_pts = [grasp[i] for i in range(len(grasp)) if clusters[i] > -1]
        cy, cx = [int(x) for x in np.median(grasp_pts,axis=0)]
        return (cy,cx)
        
    def save_features(self):
        grasps = dict()
        c1 = 0
        files = sorted([f for f in os.listdir(self.im_dir) if f.endswith(".mat")])#[4900:]
        for f in files:
            sys.stdout.write("\rFile %i of %i" %(c1, len(files)))
            sys.stdout.flush()
            label = scipy.io.loadmat(self.im_dir + f)['gt_label']
            l = cv.warpPerspective(label, np.dot(self.offset,self.tf), (450,450))
            key = f.split("_rgb")[0]
            g = self.get_img_data(l)
            if g is not None:
                grasps[key] = g
            c1+=1
        with open(self.base_dir + "cropped_grasp_positions.pkl", 'wb') as handle:
            pickle.dump(grasps, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
if __name__ == '__main__':
    proc = ImageProc()
    proc.save_features()
