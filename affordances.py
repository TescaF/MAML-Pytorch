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
    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat, include_angles=True, grasp_params=False):
        self.grasp_params = grasp_params
        self.split_task = split_cat
        self.include_angles = include_angles
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.rand = RandomState(222)
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/all_fts.pkl"
        self.cat_names = [[4,'bowl'],[4,'cup'],[2,'knife'],[3,'ladle'],[4,'mug'],[2,'scissors'],[3,'spoon'],[3,'trowel'],[6,'turner']]
        self.train = train
        if self.include_angles:
            self.weights = np.array([1.0, 1.0, 100.0])
        else:
            self.weights = np.array([1.0, 1.0])
        self.ignored_objects = "None"
        self.aff_range = list(range(2,7))
        if train:
            if self.split_task == 1:
                self.aff_range.remove(split)
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                #obj_dir = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                obj_loc = []
                for a in self.aff_range:
                    if self.grasp_params:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_positions.pkl")
                    else:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_transforms.pkl")
            else:
                self.ignored_objects = self.cat_names[split]
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                obj_loc = []
                for a in self.aff_range:
                    if self.grasp_params:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_positions.pkl")
                    else:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_transforms.pkl")
        else:
            if self.split_task == 1:
                #obj_loc = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                #obj_loc = [obj_loc + "aff_3_transforms.pkl", obj_loc + "aff_5_transforms.pkl"]
                #self.aff_range = [3, 5]
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                #obj_dir = "/home/tesca/data/part-affordance-dataset/features/small_aff/"
                if self.grasp_params:
                    obj_loc = [obj_dir + "aff_" + str(split) + "_positions.pkl"]
                else:
                    obj_loc = [obj_dir + "aff_" + str(split) + "_transforms.pkl"]
                self.aff_range = [split]
            else:
                self.ignored_objects = self.cat_names[split]
                obj_dir = "/home/tesca/data/part-affordance-dataset/features/large_aff/"
                obj_loc = []
                for a in self.aff_range:
                    if self.grasp_params:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_positions.pkl")
                    else:
                        obj_loc.append(obj_dir + "aff_" + str(a) + "_transforms.pkl")

        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        self.objects = dict()
        for loc in obj_loc:
            with open(loc, 'rb') as handle:
                obj_data = pickle.load(handle)   #dict(category) = [img1, img2, ...]
                obj_keys = list(sorted(obj_data.keys()))
                if self.split_task == 0:
                    if train:
                        obj_keys = [k for k in obj_keys if not k.startswith(self.ignored_objects[1])]
                    else:
                        obj_keys = [k for k in obj_keys if k.startswith(self.ignored_objects[1])]
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
                if not np.isin(a[-1],None).any():
                    aff = str(a[0])
                    if aff in self.affordances.keys():
                        self.affordances[aff].append(o)
                    else:
                        self.affordances[aff] = [o]
        self.dim_input = 4096
        if self.include_angles:
            self.dim_output = 3#self.outputs[0].shape[0]-1 #self.dataset[1].shape[0]
        else:  
            self.dim_output = 2
        if grasp_params:
            self.dim_params = 2
        else:
            self.dim_params = 0
        self.num_samples_per_class = k_shot + k_qry 
        #if train:
        self.task_pairs = self.aff_cat_pairs()
        #elif split_task == 1:
        #if split_task == 1:
        #    self.task_pairs = self.new_task_pairs()
        #elif split_task == 0:
        #    self.task_pairs = self.new_cat_pairs()
        #print("Task pairs: " + str(len(self.task_pairs)))
        if batchsz < 0:
            self.batch_size = len(self.task_pairs)
        else:
            self.batch_size = batchsz

    def aff_cat_pairs(self):
        pairs = []
        if self.split_task == 0 and not self.train:
            affs_list = [self.ignored_objects[0]]
        else:
            affs_list = [str(a) for a in self.aff_range if str(a) in self.affordances.keys()]
        for i in affs_list:
            new_pairs = []
            for j in self.affordances[str(i)]: #categories:
                new_pairs.append([str(i),j])
            pairs.append(new_pairs)
        return pairs

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input + self.dim_params])
        obj_keys = list(sorted(self.objects.keys()))
        if self.train: #choose tasks at random
            tasks = self.rand.choice(len(self.task_pairs), self.batch_size, replace=False)
        else: #keep order of tasks
            tasks = range(self.batch_size)
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
                if self.grasp_params:
                    if self.include_angles:
                        init_inputs[i,j] = np.concatenate((self.inputs[sample[1]], np.squeeze(np.array([obj[1]]))))
                    else:
                        init_inputs[i,j] = np.concatenate((self.inputs[sample[1]], np.squeeze(np.array([obj[1][:2]]))))
                else:
                    init_inputs[i,j] = self.inputs[sample[1]]
                if self.include_angles:
                    outputs[i, j] = np.array([obj[-1][0],obj[-1][1], angle])*self.weights
                else:
                    outputs[i, j] = np.array([obj[-1][0],obj[-1][1]])*self.weights
        return init_inputs, outputs

    def all_samples(self, aff):
        obj_keys = list(sorted(self.affordances[aff]))
        outputs = np.zeros([len(obj_keys), self.dim_output])
        init_inputs = np.zeros([len(obj_keys), self.dim_input + self.dim_params])
        for j in range(len(obj_keys)):
            obj = [i for i in self.objects[obj_keys[j]] if i[0] == int(aff)][0]
            if self.grasp_params:
                init_inputs[j] = np.concatenate((self.inputs[obj_keys[j]], np.squeeze(np.array([self.objects[obj_keys[j]][1]]))))
            else:
                init_inputs[j] = self.inputs[obj_keys[j]]
            #outputs[0,j,0] = obj[1][0]
            #outputs[0,j,1] = obj[1][1]
            outputs[j] = np.array([obj[-1][0],obj[-1][1]])
        return init_inputs, outputs

class Affordances2D(Affordances):
    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat):
        super(Affordances2D, self).__init__(batchsz, k_shot, k_qry, num_grasps, split, train, split_cat, include_angles=False)

class Affordances2DTT(Affordances):
    def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat):
        super(Affordances2DTT, self).__init__(batchsz, k_shot, k_qry, num_grasps, split, train, split_cat, include_angles=False,grasp_params=True)

if __name__ == '__main__':
    IN = Affordances(5,3,3,0.5,1,1)
    data = IN.next()
    pdb.set_trace()
