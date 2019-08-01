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
from numpy.random import RandomState

class LeaveoutGrasps:

    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.rand = RandomState(222)
        fts_loc = "/home/tesca/data/cornell_grasps/all_fts.pkl"
        out_loc = "/home/tesca/data/cornell_grasps/all_outs.pkl"
        cat_range = list(range(93))
        if train:
            if split_cat == 1:
                self.categories = dict()
                for i in cat_range:
                    if not i == split:
                        cat_loc = "/home/tesca/data/cornell_grasps/category_" + str(i) + ".pkl"
                        with open(cat_loc, 'rb') as handle:
                            self.categories[str(i)] = pickle.load(handle)
                cat_loc = "/home/tesca/data/cornell_grasps/train_obj_categories-" + str(split) + ".pkl"
        else:
            if split_cat == 1:
                cat_loc = "/home/tesca/data/cornell_grasps/category_" + str(split) + ".pkl"
                with open(cat_loc, 'rb') as handle:
                    self.categories[str(split)] = pickle.load(handle)
            else:
                cat_loc = "/home/tesca/data/cornell_grasps/test_obj_categories-" + str(split) + ".pkl"

        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        with open(out_loc, 'rb') as handle:
            self.grasps = pickle.load(handle)       #dict(img) = [[angle, x, y], ... ]

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
        #outtxt = "Objs per category:\n"
        #for c in self.categories.keys():
        #    outtxt += str(len(self.categories[c])) + ", "
        #print(outtxt)

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class*self.num_grasps_per_sample, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class*self.num_grasps_per_sample, self.dim_input + self.dim_params])
        valid_cats = [c for c in sorted(self.categories.keys()) if len(self.categories[c]) >= self.num_samples_per_class]
        if len(valid_cats) < self.batch_size:
            pdb.set_trace()
        tasks = self.rand.choice(len(valid_cats), self.batch_size, replace=False)
        tasks = [self.categories[valid_cats[t]] for t in tasks]
        #tasks = self.rand.randint(0, len(valid_cats), self.batch_size)
        for i in range(self.batch_size):
            samples = self.rand.choice(len(tasks[i]), self.num_samples_per_class, replace=False)
            #samples = self.rand.randint(0, len(objs_in_category), self.num_samples_per_class)
            for j in range(self.num_samples_per_class):
                sample = tasks[i][samples[j]]
                self.grasps[sample] = [i for i in self.grasps[sample] if not all([math.isnan(x) for x in i])]
                grasps = self.rand.choice(len(self.grasps[sample]), self.num_grasps_per_sample, replace=True)
                #grasps = self.rand.randint(0, len(self.grasps[sample]), self.num_grasps_per_sample)
                base = j * self.num_grasps_per_sample
                for g in range(self.num_grasps_per_sample):
                    grasp = self.grasps[sample][grasps[g]]

                    # Assuming grasps are symmetrical, transform to be within range [-pi/2, pi/2]
                    angle = grasp[0]
                    if angle < -90:
                        angle += 180
                    if angle > 90:
                        angle -= 180
                    angle = angle / 90 #Scale to [-1, 1]
                    init_inputs[i, base + g] = np.concatenate((self.inputs[sample], np.array([angle])))
                    outputs[i, base + g] = grasp[1:]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = CornellGrasps(100,5,8)
    data = IN.next()
