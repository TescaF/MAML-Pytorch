import math
import pdb
import sys
import  os.path
import  numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as models
import json
from PIL import Image
import pickle

class CategorizedGrasps:

    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        fts_loc = "/home/tesca/data/cornell_grasps/all_fts.pkl"
        out_loc = "/home/tesca/data/cornell_grasps/all_outs.pkl"
        if train:
            if split_cat == 1:
                cat_loc = "/home/tesca/data/cornell_grasps/train_cat_categories-" + str(split) + ".pkl"
            else:
                cat_loc = "/home/tesca/data/cornell_grasps/train_obj_categories-" + str(split) + ".pkl"
        else:
            if split_cat == 1:
                cat_loc = "/home/tesca/data/cornell_grasps/test_cat_categories-" + str(split) + ".pkl"
            else:
                cat_loc = "/home/tesca/data/cornell_grasps/test_obj_categories-" + str(split) + ".pkl"

        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        with open(out_loc, 'rb') as handle:
            self.grasps = pickle.load(handle)       #dict(img) = [[angle, x, y], ... ]
        with open(cat_loc, 'rb') as handle:
            self.categories = pickle.load(handle)   #dict(category) = [img1, img2, ...]

        self.batch_size = batchsz
        self.num_grasps_per_sample = num_grasps
        self.dim_input = 4096
        self.dim_output = 2#self.outputs[0].shape[0]-1 #self.dataset[1].shape[0]
        self.dim_params = 1
        print(str(len(self.categories.keys())) + " categories loaded")
        print(str(len(np.unique(sum(self.categories.values(),[])))) + " objects loaded")
        print("Dims " + str(self.dim_input) + "x" + str(self.dim_output))
        #print(str(len(self.inputs.keys())) + " samples loaded (Dims " + str(self.dim_input) + "x" + str(self.dim_output) + ")")

        self.num_samples_per_class = k_shot + k_qry 

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class*self.num_grasps_per_sample, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class*self.num_grasps_per_sample, self.dim_input + self.dim_params])
        valid_cats = []
        for c in list(self.categories.keys()):
            valid = False
            for i in self.categories[c]:
                valid = valid or (i in self.inputs)
            if valid:
                valid_cats.append(c)
        tasks = np.random.randint(0, len(valid_cats), self.batch_size)
        for i in range(self.batch_size):
            objs_in_category = [c for c in self.categories[valid_cats[tasks[i]]] if c in self.inputs]
            samples = np.random.randint(0, len(objs_in_category), self.num_samples_per_class)
            for j in range(self.num_samples_per_class):
                sample = objs_in_category[samples[j]]
                self.grasps[sample] = [i for i in self.grasps[sample] if not all([math.isnan(x) for x in i])]
                grasps = np.random.randint(0, len(self.grasps[sample]), self.num_grasps_per_sample)
                base = j * self.num_grasps_per_sample
                for g in range(self.num_grasps_per_sample):
                    grasp = self.grasps[sample][grasps[g]]
                    val = np.tanh(math.radians(grasp[0]))
                    if math.isnan(val):
                        pdb.set_trace()
                    init_inputs[i, base + g] = np.concatenate((self.inputs[sample], np.array([np.tanh(math.radians(grasp[0]))])))
                    outputs[i, base + g] = grasp[1:]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = CornellGrasps(100,5,8)
    data = IN.next()
