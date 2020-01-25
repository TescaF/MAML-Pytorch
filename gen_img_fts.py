import pdb
import sklearn.cluster
import numpy as np
import sys
import  os.path
import math
from os import path
import cv2 as cv
import pickle 
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models


class ImageProc:

    def __init__(self):
        self.base_dir = "/u/tesca/data/cropped/"

    def save_features(self):
        images = []
        features = dict()
        files = os.listdir(self.base_dir)
        objs = [i for i in files if i.endswith('label.mat') and (int(i.split("_00")[1].split("_label")[0])-1)%3==0]
        c1 = 0
        for o in objs[:100]:
            sys.stdout.write("\rFile %i of %i" %(c1, len(objs)))
            sys.stdout.flush()
            images.append(cv.imread(self.base_dir + o.split("label")[0] + 'rgb.jpg',-1))
            c1+=1
        feats2 = self.features_from_img(images)
        feats1 = self.features_from_img([images[-1]])
        pdb.set_trace()
        #with open(self.base_dir + "features/shifted/" + fname + "_resnet_pool_fts-14D.pkl", 'wb') as handle:
        #    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
    def features_from_img(self, img):
        imgs = []
        #try:
        for i in img:
            img_in = Image.fromarray(np.uint8(i)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            imgs.append(img_var)
        model = models.resnet50(pretrained=True)
        layer = model._modules.get("layer3") #[-2]
        pdb.set_trace()
        data_in = torch.cat(imgs)
        embedding = torch.zeros(data_in.shape[0],1024,14,14) #self.dim_input)
        #except:
        #    return None

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        print("Running...")
        model(data_in)
        print("Finished")
        hook.remove()
        return embedding.numpy()

if __name__ == '__main__':
    proc = ImageProc()
    proc.save_features()
