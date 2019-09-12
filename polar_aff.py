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
        self.temp_dir = "templates/"

    def get_centered_img(self, img, depth, data):
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
        value = min([cy, cx, img.shape[0] - cy, img.shape[1] - cx])
        img_tf = cv.linearPolar(img, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
        depth_tf = cv.linearPolar(depth, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
        feats = self.features_from_img(img_tf)
        plt.imshow(cv.cvtColor(img_tf, cv.COLOR_BGR2RGB))
        plt.show()
        return None, None

        # Get affordance point
        label_tf = cv.linearPolar(data, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
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
                    dists = [p[1] for p in disp_pts]
                    #dists = [math.sqrt((cy - p[1])**2.0 + (cx - p[0])**2.0) for p in disp_pts]
                    i = np.argmax(np.array(dists)) #dists.index(max(dists))
                    aff_data.append(list(disp_pts[i]) + [depth_tf[disp_pts[i][0],disp_pts[i][1]]])
                    #print(disp_pts[i])
                    plt.imshow(cv.cvtColor(img_tf, cv.COLOR_BGR2RGB))
                    plt.scatter([aff_data[-1][1]],[aff_data[-1][0]])
                    plt.show()
        return aff_data, feats
        
    def convert_to_polar(self):
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
                objs = [i for i in files if i.endswith('label.mat') and int(i.split(d+"_")[1].split("_label")[0])%3==0]
                c1 = 0
                for o in objs:
                    sys.stdout.write("\rFile %i of %i" %(c1, len(objs)))
                    sys.stdout.flush()
                    label = scipy.io.loadmat(self.base_dir + "tools/" + d + "/" + o)
                    label = label['gt_label']
                    labels.append([o.split("_label")[0],label])
                    depths.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'depth.png',-1))
                    images.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'rgb.jpg',-1))
                    aff_data, feats = self.get_centered_img(images[-1], depths[-1], labels[-1][1])
                    if aff_data is not None:
                        features[labels[-1][0]] = feats
                        for a in range(2,7):
                            pos = [a, aff_data[a-2]]
                            pos_dict[a-2][labels[-1][0]] = pos
                            #print(pos)#
                    c1+=1
                #for a in range(2,7):
                #    with open(self.base_dir + "features/polar_aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                #        pickle.dump(pos_dict[a-2], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.base_dir + "features/resnet_polar_fts.pkl", 'wb') as handle:
                    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            c2+=1
                            
    def features_from_img(self, img):
        try:
            img_in = Image.fromarray(np.uint8(img)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            model = models.resnet50(pretrained=True)
            #model = models.vgg16(pretrained=True)
            #pdb.set_trace()
            #layer = model._modules.get("classifier")[-2]
            pdb.set_trace()
            layer = model._modules.get("avgpool")
            layer2 = model._modules.get("layer4")
            embedding = torch.zeros(2048) #self.dim_input)
            #embedding = torch.zeros(25088) #self.dim_input)
        except:
            return None

        def copy_data(m, i, o):
            pdb.set_trace()
            embedding.copy_(o.data.squeeze().flatten())
        hook = layer2.register_forward_hook(copy_data)
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
    proc.convert_to_polar()
    #proc.reduce_features()
    #proc.proc_features()
