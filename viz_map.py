from sklearn import preprocessing
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

    def standard_features_from_img(self, filename):
        img = filename + "_rgb.jpg"
        try:
            img_in = Image.open(img) #fromarray(np.uint8(img)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))

            model = models.resnet50(pretrained=True) #.to(self.device)
            layer = model._modules.get("layer4")
            embedding = torch.cuda.FloatTensor(2048,7,7)
            output = torch.cuda.FloatTensor(2048)
        except:
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.squeeze())
        def copy_output(m, i, o):
            output.copy_(o.squeeze())
        hook = layer.register_forward_hook(copy_data)
        out_hook = model._modules.get("avgpool").register_forward_hook(copy_output)
        with torch.no_grad():
            model(img_var)
        hook.remove()
        return embedding, output, np.array(img_in)

    def polar_features_from_img(self, filename):
        img = filename + "_rgb.jpg"
        data = scipy.io.loadmat(filename + "_label.mat")['gt_label']
        pdb.set_trace()
        try:
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

            self.model = models.resnet50(pretrained=True) #.to(self.device)
            layer = self.model._modules.get("layer4")
            embedding = torch.cuda.FloatTensor(2048,7,7)
        except:
            pdb.set_trace()
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.squeeze())
        hook = layer.register_forward_hook(copy_data)
        with torch.no_grad():
            output = self.model(img_var)
        hook.remove()
        return embedding, output.cuda(), img_tf

    def load_fc_weights(self):
        load_path = os.getcwd() + '/data/tfs/model_batchsz20_stepsz0.1_exclude-1_epoch0_al.pt'
        #load_path = os.getcwd() + '/data/tfs/model_batchsz1_stepsz0.1_exclude-1_epoch0_al.pt'
        print(load_path)
        config = [ ('linear', [58,2048])]
        #config = [
        #    ('linear', [128,2048]),
        #    ('relu', [True]),
        #    ('linear', [87,128]),
        #]

        self.device = torch.device('cuda')
        maml = Learner(config).to(self.device)
        model = torch.load(load_path)
        keys = list(model.keys())
        for k in keys:
            model[k.split("net.")[1]] = model.pop(k)
        maml.load_state_dict(model)
        maml.eval()
        ft_maps = maml.parameters()[0] #torch.mm(maml.parameters()[-2],maml.parameters()[0])
        return maml, ft_maps

    def get_grads(self, filename, scaler, img_fts, class_num):
        embedding, output, img = self.standard_features_from_img(filename)
        #embedding, output, polar_img = self.standard_features_from_img(filename)
        #out = self.mod(torch.from_numpy(scaler.transform(output.detach().cpu().reshape(1,-1))).float().cuda())
        out = self.mod(torch.from_numpy(scaler.transform(img_fts.reshape(1,-1))).float().cuda())
        c = F.softmax(out,dim=1).argmax()
        scaled = embedding.reshape((2048,49)).detach().cpu().numpy()
        
        scaled_rows = []
        for i in range(49):
            sc = scaler.transform(scaled[:,i].reshape(1,-1))
            scaled_rows.append(sc)
        emb = torch.from_numpy(np.concatenate(scaled_rows).transpose()).cuda()
        cam = torch.mm(self.ft_maps[c].unsqueeze(0), embedding.reshape((2048,49)))
        outs = cam.reshape((7,7))
        outs = outs - torch.min(outs)
        outs = outs / torch.max(outs)
        outs = np.uint8(outs.cpu().detach().numpy() * 255)
        layer = cv.resize(outs, (640,480))
        heatmap = cv.applyColorMap(layer, cv.COLORMAP_JET)
        cv.addWeighted(heatmap, 0.3, img, 0.7, 0, heatmap)
        cv.imshow("im", heatmap)
        cv.waitKey()
        cv.destroyAllWindows()
        return layer

if __name__ == '__main__':
    proc = ImageProc()
    fts_loc = proc.base_dir + "/features/resnet_fts.pkl"
    #fts_loc = proc.base_dir + "/features/resnet_polar_fts.pkl"
    with open(fts_loc, 'rb') as handle:
        inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
    input_scale = preprocessing.StandardScaler()
    input_scale.fit(np.array(list(inputs.values())))

    classes = list(sorted(set([k.split("_00")[0] for k in inputs.keys()])))
    objs = ['cup_02'] #"knife_03", "ladle_02", "mallet_02", "mallet_01", "saw_01"]
    for img in classes:
        proc.get_grads(proc.base_dir + "/tools/" + img + "/" + img + "_00000120", input_scale, inputs[img + "_00000120"], classes.index(img))
