import pdb
import sys
import  os.path
import math
import  numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as models
from PIL import Image
from os import path
from itertools import islice
import pickle 

class ImageNetData:

    def __init__(self, training_percent):
        self.base_dir = "/home/tesca/data/cornell_grasps/"
        self.dim_input = 4096
        self.ex_count = 2
        self.trp = training_percent

    def train_test_split(self, cats, objs):
        train_cat = dict() #training set of categories across all objects
        test_cat = dict()  #test set of categories across all objects
        train_obj = dict() #training set of objects across all categories
        test_obj = dict()  #test set of objects across all categories
        train_obj_count = 0
        test_obj_count = 0

        # Split data by category
        k = list(cats.keys())
        tr_idx = int(math.ceil(self.trp * float(len(k))))
        for tr_k in k[:tr_idx]:
            train_cat[tr_k] = cats[tr_k]
        for ts_k in k[tr_idx:]:
            test_cat[ts_k] = cats[ts_k]

        # Split data by object id
        for ki in k:
            unique_objs = np.unique(objs[ki])
            tr_idx = int(math.ceil(self.trp * float(len(unique_objs))))
            if len(unique_objs) - tr_idx > 0:
                train_obj_count += tr_idx
                test_obj_count += (len(unique_objs) - tr_idx)
                train_obj[ki] = [] 
                test_obj[ki] = [] 
                for j in range(len(cats[ki])):
                    if objs[ki][j] in unique_objs[:tr_idx]:
                        train_obj[ki].append(cats[ki][j])
                    else:
                        test_obj[ki].append(cats[ki][j])
        print("Categories train/test: " + str(len(train_cat.keys())) + "/" + str(len(test_cat.keys())))
        print("Objects train/test: " + str(train_obj_count) + "/" + str(test_obj_count))
        return train_cat, test_cat, train_obj, test_obj

    def load_categories(self):
        zfile = self.base_dir + "/processed/z.txt"
        cats = dict()
        objs = dict()
        with open(zfile) as f:
            lines = [l.rstrip('\n').split() for l in f]
            prev_img = None
            for l in lines: 
                img = l[0] #image id
                oid = l[1] #object id
                key = l[2] #category desc
                if img == prev_img: # or not (img[:2] in self.prefixes):
                    continue
                prev_img = img
                if key in cats:
                    cats[key] += [img]
                    objs[key] += [oid]
                else:
                    cats[key] = [img]
                    objs[key] = [oid]
        return cats, objs

    def load_grasps(self, cats):
        dims = dict()
        for c in cats.keys():
            imgs = cats[c]
            if len(imgs) < self.ex_count:
                continue
            for i in imgs:
                p = i[:2]
                img_fn = self.base_dir + p + "/cropped/pcd" + str(i) + "r.png"
                grasp_fn = self.base_dir + p + "/pcd" + str(i) + "cpos.txt"
                if not (path.exists(img_fn) and path.exists(grasp_fn)):
                    continue
                with open(grasp_fn) as f:
                    grasps = []
                    while True:
                        rect = list(islice(f, 4))
                        if rect:
                            coords = [[float(c.strip().split(" ")[0])-150.0, float(c.strip().split(" ")[1])-140.0] for c in rect] 
                            angle = math.degrees(math.atan2(coords[0][1] - coords[1][1], coords[0][0] - coords[1][0]))
                            x = sum([c[0] for c in coords])/4.0
                            y = sum([c[1] for c in coords])/4.0
                            grasps.append([angle, x, y])
                        else:
                            dims[i] = grasps
                            break
        return dims

    def proc_features(self, img_nums):
        dataset = dict()
        i = 0 
        for k in img_nums: #i in range(len(self.dims)):
            sys.stdout.write("\rLoading image %i of %i" %(i, len(img_nums)))
            sys.stdout.flush()
            inputs = self.features(k) #self.dims[i][0], self.dims[i][1])
            if inputs is None:
                print("\nError reading image " + k)
                pdb.set_trace()
            else:
                dataset[k] = inputs
            i += 1
        return dataset

    def features(self, img_num): #prefix, img_num):
        try:
            prefix = img_num[:2]
            img_loc = self.base_dir + prefix + "/cropped/pcd" + str(img_num) + "r.png"
            img_in = Image.open(img_loc)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Scale((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            model = models.vgg16(pretrained=True)
            layer = model._modules.get("classifier")[-2]
            embedding = torch.zeros(self.dim_input)
        except:
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        model(img_var)
        hook.remove()
        return embedding.numpy()

if __name__ == '__main__':
    split = 0.7
    img_data = ImageNetData(split)
    cats,objs = img_data.load_categories()
    tr1, ts1, tr2, ts2 = img_data.train_test_split(cats, objs)
    outputs = img_data.load_grasps(cats)
    dataset = img_data.proc_features(outputs.keys())
    with open(img_data.base_dir + "all_fts.pkl", 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(img_data.base_dir + "all_outs.pkl", 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(img_data.base_dir + "train_cat_categories-" + str(split) + ".pkl", 'wb') as handle:
        pickle.dump(tr1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(img_data.base_dir + "test_cat_categories-" + str(split) + ".pkl", 'wb') as handle:
        pickle.dump(ts1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(img_data.base_dir + "train_obj_categories-" + str(split) + ".pkl", 'wb') as handle:
        pickle.dump(tr2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(img_data.base_dir + "test_obj_categories-" + str(split) + ".pkl", 'wb') as handle:
        pickle.dump(ts2, handle, protocol=pickle.HIGHEST_PROTOCOL)
