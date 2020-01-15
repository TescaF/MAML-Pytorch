from copy import deepcopy
from sklearn import preprocessing, decomposition
from sklearn.decomposition import PCA
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
        self.base_dir = "/u/tesca/data/part-affordance-dataset/"
        self.im_dir = "/u/tesca/data/cropped/"
        self.tf = cv.getPerspectiveTransform(np.array([[445,170],[525,375],[125,375],[153,170]],dtype="float32"),np.array([[525,170],[525,425],[125,425],[125,170]],dtype="float32"))
        self.offset = np.array([[ 1 , 0 , -110], [ 0 , 1 , -60], [ 0 , 0 ,    1    ]])

    def get_img_data(self, img, data):
        feats = self.features_from_img(img)
        label_tf = data

        # Get affordance point
        aff_data = []
        for a in range(2,7):
            aff_pts = [(i,j) for i in range(label_tf.shape[0]) for j in range(label_tf.shape[1]) if label_tf[i,j] == a]
            if len(aff_pts) == 0:
                aff_data.append(None)
            else:
                clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=5).fit_predict(aff_pts)
                disp_pts = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
                if len(disp_pts) == 0:
                    aff_data.append(None)
                else:
                    cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
                    aff_data.append([cy,cx])
        return aff_data, feats
        
    def get_grasp_normal(self, img_name):
        output_scale = preprocessing.MinMaxScaler(feature_range=(-1,1))
        output_scale.fit(np.matrix([[0,0],[450,450]]))

        label = scipy.io.loadmat(self.im_dir + img_name + "_label.mat")
        img_affs = cv.warpPerspective(label['gt_label'], np.dot(self.offset,self.tf), (450,450))

        aff_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] > 1 and img_affs[i,j] < 7])
        if aff_pts.shape[1] == 0:
            return None
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=10).fit_predict(aff_pts)
        aff_clust = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
        ay, ax = np.median(aff_clust,axis=0).squeeze(0)

        grasp_pts = np.matrix([(i,j) for i in range(img_affs.shape[0]) for j in range(img_affs.shape[1]) if img_affs[i,j] >= 1])
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
        grasp_clust = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
        cy, cx = np.median(grasp_clust,axis=0).squeeze(0)

        # Get edge normal
        pca = PCA(n_components=2)
        pca_tf = pca.fit_transform(np.stack(grasp_clust))
        normal = math.atan2(pca.components_[0,1],pca.components_[0,0])
        var_ratio = deepcopy(pca.explained_variance_ratio_)
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

        grasp_diff = np.array(output_scale.transform(np.array([ncy,ncx]).reshape(1,-1)) - output_scale.transform(np.array([ey,ex]).reshape(1,-1))).squeeze(0)
        return [align_normal,var_ratio,(ey,ex),(cy,cx),(ncy,ncx),(ty,tx)]

    def save_features(self):

        labels, images = [], []
        pos_dict = []
        features = dict()
        for i in range(2,7):
            pos_dict.append(dict())
        c1 = 0
        with open("/u/tesca/data/keys.txt",'rb') as f:
            k = f.readlines()[0].decode()
            keys = ast.literal_eval(k)
        files = sorted([f for f in os.listdir(self.im_dir) if f.endswith(".mat") and not f in keys])#[4900:]
        for f in files:
            sys.stdout.write("\rFile %i of %i" %(c1, len(files)))
            sys.stdout.flush()
            obj_name = f.split("_00")[0]
            mat = self.im_dir + f #.split("rgb")[0] + "label.mat"
            label = scipy.io.loadmat(mat)
            label = label['gt_label']
            im = cv.imread(self.im_dir + f.split("label")[0] + "rgb.jpg") #,-1)
            images.append(im)
            l = cv.warpPerspective(label, np.dot(self.offset,self.tf), (450,450))
            labels.append([f.split("_rgb")[0],l])
            aff_data, feats = self.get_img_data(images[-1], labels[-1][1])

            if aff_data is not None:
                features[labels[-1][0]] = feats
                for a in range(2,7):
                    pos = [a, aff_data[a-2]]
                    pos_dict[a-2][labels[-1][0]] = pos
            c1+=1
            if c1 == len(files) or (c1 % 100 == 0):
                with open(self.base_dir + str(c1) + "_rem_cropped_resnet_pool_fts-14D.pkl", 'wb') as handle:
                    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
                features = dict()
        '''for a in range(2,7):
            with open(self.base_dir + "features/cropped_aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                pickle.dump(pos_dict[a-2], handle, protocol=pickle.HIGHEST_PROTOCOL)'''
                        
    def features_from_img(self, img):
        try:
            img_in = Image.fromarray(np.uint8(img)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            model = models.resnet50(pretrained=True)
            layer = model._modules.get("layer3") #[-2]
            embedding = torch.zeros(1024,14,14) #self.dim_input)
        except:
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        model(img_var)
        hook.remove()
        return embedding.numpy()

if __name__ == '__main__':
    start = 9000
    proc = ImageProc()
    #proc.save_features()
    with open(proc.im_dir + "tt_w_var.pkl", 'rb') as handle:
        pt_dict = pickle.load(handle)
    #pt_dict = dict()
    files = sorted([f for f in os.listdir(proc.im_dir) if f.endswith(".mat")])
    #files = files[start:min(len(files),start+1000)]
    c1 = 0
    for f in files:
        sys.stdout.write("\rFile %i of %i" %(c1, len(files)))
        sys.stdout.flush()
        name = f.split("_label")[0]
        if name not in pt_dict.keys():
            data = proc.get_grasp_normal(name)
            pt_dict[name] = data
        c1 += 1
        if c1 % 10 == 0:
            with open(proc.im_dir + "tt_w_var_out.pkl", 'wb') as handle:
                pickle.dump(pt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(proc.im_dir + "tt_w_var_out.pkl", 'wb') as handle:
        pickle.dump(pt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
