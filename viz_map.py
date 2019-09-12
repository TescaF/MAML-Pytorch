import pickle
import pdb
from learner import Learner
import sklearn.cluster
from sklearn.linear_model import LinearRegression
from    torch.nn import functional as F
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
        self.mod, self.ft_maps = self.load_fc_weights()

    def features_from_img(self, filename):
        img = filename + "_rgb.jpg"
        data = scipy.io.loadmat(filename + "_label.mat")['gt_label']
        try:
            img_in = Image.open(img) #fromarray(np.uint8(img)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))

            grasp_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == 1]
            if len(grasp_pts) == 0:
                return None,None
            clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
            disp_pts = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
            if len(disp_pts) == 0:
                return None,None
            cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
            value = min([cy, cx, img_in.size[1] - cy, img_in.size[0] - cx])
            img_tf = cv.linearPolar(np.array(img_in), (cx,cy), value, cv.WARP_FILL_OUTLIERS)
            #plt.imshow(img_tf)
            #plt.imshow(cv.cvtColor(img_tf, cv.COLOR_BGR2RGB))
            #plt.show()

            model = models.resnet50(pretrained=True) #.to(self.device)
            layer = model._modules.get("layer4")
            embedding = torch.cuda.FloatTensor(2048,7,7)
        except:
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.squeeze())
        hook = layer.register_forward_hook(copy_data)
        output = model(img_var)
        hook.remove()
        return embedding, output.cuda(), img_tf

    def get_cam(self, fts, fc_weights):
        cam = fc_weights.dot(fts)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

    def load_fc_weights(self):
        load_path = os.getcwd() + '/data/tfs/model_batchsz20_stepsz0.1_exclude-1_epoch200_al.pt'
        print(load_path)
        config = [
            ('linear', [128,2048]),
            ('relu', [True]),
            ('linear', [87,128]),
        ]

        self.device = torch.device('cuda')
        maml = Learner(config).to(self.device)
        model = torch.load(load_path)
        keys = list(model.keys())
        for k in keys:
            model[k.split("net.")[1]] = model.pop(k)
        maml.load_state_dict(model)
        maml.eval()
        ft_maps = torch.mm(maml.parameters()[-2],maml.parameters()[0])
        return maml, ft_maps

    def get_grads(self, filename, class_num):
        embedding, output, polar_img = self.features_from_img(filename)
        pool = torch.nn.AvgPool2d(7,7,0)
        mean = pool(embedding).squeeze()
        res = self.mod(mean,bn_training=False)
        c = F.softmax(res,dim=0).argmax()
        embedding = embedding.reshape((2048,49))
        
        #pooled = torch.mean(embedding.reshape((2048,49)),dim=0).unsqueeze(0)
        #m = torch.mean(res, dim=1)
        #m.backward()
        #g = self.mod.parameters()[0].grad.transpose(0,1)
        #mean = torch.mean(g,dim=1)
        #outs = torch.cuda.FloatTensor(2048,7,7)
        dot = torch.mm(self.ft_maps, embedding)
        outs = dot[c].reshape((7,7))
        #for i in range(2048):
        #    outs[i,:,:] = embedding[i,:,:] * self.ft_maps[class_num,i]
            #outs[i,:,:] = embedding[i,:,:] * mean[i]
        #outs = torch.mean(outs,dim=0)
        #outs = torch.max(torch.zeros_like(outs), outs)
        #outs = torch.abs(1/outs)
        outs = outs - torch.min(outs)
        outs = outs / torch.max(outs)
        outs = np.uint8((outs * 255).cpu().detach().numpy())
        layer = cv.resize(outs, (640,480))
        heatmap = cv.applyColorMap(layer, cv.COLORMAP_JET)
        #inal_im = heatmap  0.3 + polar_img * 0.5
        cv.addWeighted(heatmap, 0.3, polar_img, 0.7, 0, heatmap)
        cv.imshow("im", heatmap)
        cv.waitKey()
        cv.destroyAllWindows()
        return layer

if __name__ == '__main__':
    proc = ImageProc()
    fts_loc = proc.base_dir + "/features/resnet_polar_fts.pkl"
    with open(fts_loc, 'rb') as handle:
        inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
    classes = list(sorted(set([k.split("_00")[0] for k in inputs.keys()])))

    objs = ["knife_01", "ladle_01", "mallet_03"]
    for img in objs:
        proc.get_grads(proc.base_dir + "/tools/" + img + "/" + img + "_00000120", classes.index(img))
