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

class Affordances:
    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat):
        split_task = split_cat
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.rand = RandomState(222)
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/all_fts.pkl"
        self.train = train
        if train:
            self.aff_range = list(range(2,7))
            self.aff_range.remove(split)
            if split_task == 1:
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                #obj_dir = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                obj_loc = []
                for a in self.aff_range:
                    obj_loc.append(obj_dir + "aff_" + str(a) + "_transforms.pkl")
            else:
                obj_loc = ["/home/tesca/data/part-affordance-dataset/features/small_aff/obj_split_1_transforms.pkl"]
        else:
            if split_task == 1:
                #obj_loc = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                #obj_loc = [obj_loc + "aff_3_transforms.pkl", obj_loc + "aff_5_transforms.pkl"]
                #self.aff_range = [3, 5]
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                #obj_dir = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                obj_loc = [obj_dir + "aff_" + str(split) + "_transforms.pkl"]
                self.aff_range = [split]
            else:
                obj_loc = ["/home/tesca/data/part-affordance-dataset/features/small_aff/obj_split_2_transforms.pkl"]
                self.aff_range = range(2,7)

        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        self.objects = dict()
        for loc in obj_loc:
            with open(loc, 'rb') as handle:
                obj_data = pickle.load(handle)   #dict(category) = [img1, img2, ...]
                obj_keys = list(sorted(obj_data.keys()))
                for k in obj_keys:
                    if k in self.objects.keys():
                        self.objects[k].append(obj_data[k])
                    elif len(obj_loc) > 1:
                        self.objects[k] = [obj_data[k]]
                    else:
                        self.objects[k] = [obj_data[k]]
                        #self.objects[k] = obj_data[k]
        self.affordances = dict()
        obj_keys = list(sorted(self.objects.keys()))
        for o in obj_keys:
            for a in self.objects[o]:
                if not np.isin(a[1],None).any():
                    aff = str(a[0])
                    if aff in self.affordances.keys():
                        self.affordances[aff].append(o)
                    else:
                        self.affordances[aff] = [o]
        self.dim_input = 4096
        self.dim_output = 3#self.outputs[0].shape[0]-1 #self.dataset[1].shape[0]
        self.dim_params = 0
        self.num_samples_per_class = k_shot + k_qry 
        #if train:
        self.task_pairs = self.aff_cat_pairs()
        #elif split_task == 1:
        #if split_task == 1:
        #    self.task_pairs = self.new_task_pairs()
        #elif split_task == 0:
        #    self.task_pairs = self.new_cat_pairs()
        print("Task pairs: " + str(len(self.task_pairs)))
        if batchsz < 0:
            self.batch_size = len(self.task_pairs)
        else:
            self.batch_size = batchsz

    def new_task_pairs(self):
        pairs = []
        for i in self.aff_range:
            #aff_idx = self.aff_range.index(int(i))
            # Get samples in that are valid for the current affordance
            #samples = [o for o in obj_keys if not np.isin(self.objects[o][aff_idx][1], None).any()] 
            samples = self.affordances[str(i)]
            pairs.append([[str(i), s] for s in samples])
        return pairs

    def new_cat_pairs(self):
        pairs = []
        categories = [c.split("_")[0] for c in self.objects.keys()]
        counts = [categories.count(c) for c in list(set(categories))]
        categories = list(sorted(set([c for c in categories if categories.count(c) >= self.num_samples_per_class])))
        for cat in categories:
            valid_objs = []
            cat_pairs = []
            for aff in self.aff_range: 
                # Get all objects in this category for this affordance
                objs = [c for c in self.affordances[str(aff)] if c.startswith(cat)]
                for obj in objs:
                    cat_pairs.append([str(aff),obj]) 
            pairs.append(cat_pairs)
        return pairs

    def aff_cat_pairs(self):
        pairs = []
        for i in self.aff_range:
            #categories = [c.split("_")[0] for c in self.affordances[str(i)]]
            #categories = list(sorted(set([c for c in categories if categories.count(c) >= self.num_samples_per_class])))
            #categories = list(sorted(set([c for c in self.objects.keys() if categories.count(c.split("_")[0]) >= self.num_samples_per_class])))
            new_pairs = []
            for j in self.affordances[str(i)]: #categories:
                new_pairs.append([str(i),j])
            pairs.append(new_pairs)
        return pairs

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input + self.dim_params])
        obj_keys = list(sorted(self.objects.keys()))

        # Select tasks
        if self.train: #choose tasks at random
            tasks = self.rand.choice(len(self.task_pairs), self.batch_size, replace=False)
        else: #keep order of tasks
            tasks = range(self.batch_size)
        #tasks = self.rand.choice(len(self.task_pairs), self.batch_size, replace=False)
        for i in range(self.batch_size):
            # Get all objects with valid affordance for current task
            task_data = self.task_pairs[tasks[i]]   
            samples = self.rand.choice(len(task_data), self.num_samples_per_class, replace=False)
            for j in range(self.num_samples_per_class):
                sample = task_data[samples[j]]
                aff = sample[0]
                obj = [i for i in self.objects[sample[1]] if i[0] == int(aff)][0]
                # Assuming grasps are symmetrical, transform to be within range [-pi/2, pi/2]
                angle = obj[1][2]
                if angle is None:
                    pdb.set_trace()
                if angle < -math.pi/2.0:
                    angle += math.pi
                if angle > math.pi/2.0:
                    angle -= math.pi
                angle = angle / (math.pi / 2.0) #Scale to [-1, 1]
                init_inputs[i,j] = self.inputs[sample[1]]
                outputs[i, j] = [obj[1][0],obj[1][1], angle]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = Affordances(5,3,3,0.5,1,1)
    data = IN.next()
    pdb.set_trace()
