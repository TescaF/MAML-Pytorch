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
from matplotlib import pyplot as plt
from itertools import product
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models


class ImageProc:

    def __init__(self):
        self.base_dir = "/home/tesca/data/part-affordance-dataset/"

    def get_img_tf(self, img, data, mode):
        # Get centerpoint
        grasp_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == 1]
        if len(grasp_pts) == 0:
            return None
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
        disp_pts = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
        if len(disp_pts) == 0:
            return None
        cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
        #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #plt.scatter([cx],[cy])
        #plt.show()
        if mode == "center":
            x_shift = (img.shape[1]/2.0)-cx
            y_shift = (img.shape[0]/2.0)-cy
            M = np.float32([[1,0,x_shift],[0,1,y_shift]])
            img_tf = cv.warpAffine(img, M,(img.shape[1],img.shape[0]))
        elif mode == "polar":    
            value = img.shape[1]/2.0 #min([cy, cx, img.shape[0] - cy, img.shape[1] - cx])
            img_tf = cv.linearPolar(img, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
        else:
            img_tf = img
        return img_tf

    def get_img_data(self, img, depth, data, mode):
        if mode == "polar" or mode == "center":
            # Get centerpoint
            grasp_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == 1]
            if len(grasp_pts) == 0:
                return None,None
            clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
            disp_pts = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
            if len(disp_pts) == 0:
                return None,None
            cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
            #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            #plt.scatter([cx],[cy])
            #plt.show()
            if mode == "polar":
                value = min([cy, cx, img.shape[0] - cy, img.shape[1] - cx])
                img_tf = cv.linearPolar(img, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
                feats = self.features_from_img(img_tf)
                label_tf = cv.linearPolar(data, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
                depth_tf = cv.linearPolar(depth, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
            if mode == "center":
                x_shift = (img.shape[1]/2.0)-cx
                y_shift = (img.shape[0]/2.0)-cy
                M = np.float32([[1,0,x_shift],[0,1,y_shift]])
                label_tf = cv.warpAffine(data, M,(data.shape[1], data.shape[0]))
                depth_tf = cv.warpAffine(depth, M,(depth.shape[1], depth.shape[0]))
                img_tf = cv.warpAffine(img, M,(img.shape[1],img.shape[0]))
                feats = self.features_from_img(img_tf)
            grasp = [cy,cx]
        else:
            feats = self.features_from_img(img)
            depth_tf = depth
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
                    if mode == "polar":
                        dists = [p[1] for p in disp_pts]
                        i = np.argmax(np.array(dists)) #dists.index(max(dists))
                        cy, cx = disp_pts[i]
                    else:
                        #dists = [math.sqrt((grasp[1]-p[1])**2 + (grasp[0]-p[0])**2) for p in disp_pts]
                        #i = np.argmax(np.array(dists)) #dists.index(max(dists))
                        cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
                        #cy, cx = disp_pts[i]
                    cz = depth_tf[cy,cx]
                    aff_data.append([cy,cx,cz])
                    #print(aff_data[-1])
                    #plt.imshow(cv.cvtColor(img_tf, cv.COLOR_BGR2RGB))
                    #plt.scatter([aff_data[-1][1]],[aff_data[-1][0]])
                    #plt.show()
        return aff_data, feats
        
    def save_features(self, mode):
        depths, labels, images = [], [], []
        dirs = os.listdir(self.base_dir + "tools/")
        pos_dict = []
        features = dict()
        for i in range(2,7):
            pos_dict.append(dict())
        c2 = 0
        for d in dirs:
            print("Directory " + str(c2) + " of " + str(len(dirs)))
            if os.path.isdir(self.base_dir + "tools/" + d):
                files = os.listdir(self.base_dir + "tools/" + d)
                objs = [i for i in files if i.endswith('label.mat') and (int(i.split(d+"_")[1].split("_label")[0])-1)%3==0]
                c1 = 0
                for o in objs:
                    sys.stdout.write("\rFile %i of %i" %(c1, len(objs)))
                    sys.stdout.flush()
                    label = scipy.io.loadmat(self.base_dir + "tools/" + d + "/" + o)
                    label = label['gt_label']
                    labels.append([o.split("_label")[0],label])
                    depths.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'depth.png',-1))
                    images.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'rgb.jpg',-1))
                    img_tf = self.get_img_tf(images[-1], labels[-1][1], mode)
                    if mode=="polar" and (img_tf is not None):
                        cv.imwrite(self.base_dir + 'polar_tools/' + o.split("_label")[0] + '_polar.jpg', img_tf)
                    if mode=="center" and (img_tf is not None):
                        cv.imwrite(self.base_dir + 'center_tools/' + o.split("_label")[0] + '_center.jpg', img_tf)

                    aff_data, feats = self.get_img_data(images[-1], depths[-1], labels[-1][1], mode)
                    if aff_data is not None:
                        features[labels[-1][0]] = feats
                        for a in range(2,7):
                            pos = [a, aff_data[a-2]]
                            pos_dict[a-2][labels[-1][0]] = pos
                    c1+=1
            c2+=1
            if mode=="polar":
                fname = "polar"
            elif mode=="center":
                fname = "center"
            else:
                fname = "cart"
            with open(self.base_dir + "features/shifted/" + fname + "_resnet_pool_fts-14D.pkl", 'wb') as handle:
                pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for a in range(2,7):
                with open(self.base_dir + "features/shifted/" + fname + "_aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                    pickle.dump(pos_dict[a-2], handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
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

    def reduce_features(self):
        var = 0.95
        with open(self.base_dir + "features/polar/polar_fts.pkl", 'rb') as handle:
            fts = pickle.load(handle)
        keys = list(fts.keys())
        scaler = sklearn.preprocessing.StandardScaler()
        vals = list(fts.values())
        print("Fitting data...")
        scaler.fit(vals)
        sc_data = scaler.transform(vals)
        pca = sklearn.decomposition.PCA(var)
        pca_fit = pca.fit(sc_data)
        print("Transforming data...")
        for k in keys:
            a = scaler.transform(fts[k].reshape(1,-1))
            fts[k] = pca_fit.transform(a).squeeze(0)
        print("Writing data...")
        with open(self.base_dir + "features/polar/reduced_fts_" + str(var) + ".pkl", 'wb') as handle:
            pickle.dump(fts, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    proc = ImageProc()
    #proc.process_images()
    #proc.show_transforms("mug_01-2",4)
    #proc.show_transforms("mug_01",3)
    #proc.save_transforms(transform=False)
    #proc.process_all_images()
    #proc.convert_to_polar()
    #proc.reduce_features()
    #proc.proc_features()
    proc.save_features(mode="center")
