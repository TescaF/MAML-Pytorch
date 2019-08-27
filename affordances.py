import sklearn
import math
import pdb
import sys
import  os.path
import  numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing, decomposition
import torchvision.models.vgg as models
import json
from PIL import Image
import pickle
from numpy.random import RandomState


class Affordances:
    #def __init__(self, batchsz, k_shot, k_qry, num_grasps, split, train, split_cat, include_angles=True, grasp_params=False):
    def __init__(self, batchsz, k_shot, k_qry, train, new_aff, exclude):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.rand = RandomState(222)
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/reduced_fts_0.95.pkl"
        #fts_loc = "/home/tesca/data/part-affordance-dataset/features/all_fts.pkl"
        self.new_aff = (new_aff == 1)
        self.categories = [[4,'bowl'],[4,'cup'],[2,'knife'],[4,'ladle'],[4,'mug'],[2,'scissors'],[3,'spoon'],[3,'trowel'],[6,'turner']]
        self.train = train
        self.ignored_objects = "None"
        self.aff_range = list(range(2,7))
        obj_dir = "/home/tesca/data/part-affordance-dataset/features/all_small_aff/"
        obj_loc = []
        self.img_size = [640,480]

        ## Configure data directories for train/test split
        if self.new_aff:
            if train:
                self.aff_range.remove(exclude)
                for a in self.aff_range:
                    obj_loc.append(obj_dir + "aff_" + str(a) + "_positions.pkl")
            else:
                self.aff_range = [exclude]
                obj_loc = [obj_dir + "aff_" + str(exclude) + "_positions.pkl"]
        else:
            self.ignored_objects = self.categories[exclude]
            for a in self.aff_range:
                obj_loc.append(obj_dir + "aff_" + str(a) + "_positions.pkl")

        print("Loading data...")
        ## Load VGG features for all images
        with open(fts_loc, 'rb') as handle:
            self.inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]

        ## Load object categories (linking categories to image names)
        self.objects = dict()
        for loc in obj_loc:
            with open(loc, 'rb') as handle:
                obj_data = pickle.load(handle)      #dict(category) = [img1, img2, ...]
                obj_keys = list(sorted(obj_data.keys()))

                ## Set list of valid object categories for this data split
                if not self.new_aff:
                    if train:
                        obj_keys = [k for k in obj_keys if not k.startswith(self.ignored_objects[1])]
                    else:
                        obj_keys = [k for k in obj_keys if k.startswith(self.ignored_objects[1])]

                ## Create dictionary of valid object names 
                for k in obj_keys:
                    if k in self.objects.keys():
                        self.objects[k].append(obj_data[k])
                    else:
                        self.objects[k] = [obj_data[k]]

        ## Store object affordances (linking affordances to image names)
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

        self.num_samples_per_class = k_shot + k_qry 
        self.task_pairs = self.aff_cat_pairs()
        if batchsz < 0:
            self.batch_size = len(self.task_pairs)
        else:
            self.batch_size = batchsz
        self.dim_output = 2
        self.dim_input = len(list(self.inputs.values())[0])
        '''if pca_fit is None:
            self.pca_fit, _, _, = self.all_samples()
        else:
            self.pca_fit = pca_fit
        self.dim_input = self.pca_fit.n_components_
        self.reduce_inputs()'''

    def reduce_inputs(self):
        print("Reducing inputs...")
        keys = list(sorted(self.inputs.keys()))
        for k in keys:
            self.inputs[k] = self.pca_fit.transform(self.inputs[k].reshape(1,-1)).squeeze(0)

    def aff_cat_pairs(self):
        pairs = []
        if self.new_aff or self.train:
            affs_list = [str(a) for a in self.aff_range if str(a) in self.affordances.keys()]
        else:
            affs_list = [self.ignored_objects[0]]
        for i in affs_list:
            categories = list(set([j.split("_")[0] for j in self.affordances[str(i)]]))
            obj_cat = []
            for c in categories:
                new_pairs = []
                for j in self.affordances[str(i)]: 
                    if j.startswith(c):
                        new_pairs.append([str(i),j])
                if len(new_pairs) >= self.num_samples_per_class:
                    pairs.append(new_pairs)
                    #obj_cat.append(new_pairs)
            #pairs.append(obj_cat)
        return pairs

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
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
                angle = obj[-1][-1]
                if angle < -math.pi/2.0:
                    angle += math.pi
                if angle > math.pi/2.0:
                    angle -= math.pi
                angle = angle / (math.pi / 2.0) #Scale to [-1,1]
                init_inputs[i,j] = self.inputs[sample[1]]
                #init_inputs[i,j] = self.pca_fit.transform(self.inputs[sample[1]].reshape(1,-1)).squeeze(0)
                #init_inputs[i,j] = np.concatenate((self.inputs[sample[1]], np.squeeze(np.array([obj[1][:2]]))))
                outputs[i, j] = np.array([obj[-1][0],obj[-1][1]])
                a = outputs[i,j]
                #outputs[i, j] = np.array([(obj[-1][0]-(self.img_size[0]/2.0)),(obj[-1][1]-(self.img_size[1]/2.0))]) #,angle])
        return init_inputs, outputs

    def all_samples(self):
        obj_keys = list(sorted(self.objects.keys()))
        ins,outs = [],[]
        for j in range(len(obj_keys)):
            obj = [i for i in self.objects[obj_keys[j]] if not None in i[-1]]
            for o in obj:
                ins.append(self.inputs[obj_keys[j]])
                outs.append(np.array([o[-1][0],o[-1][1]]))
        pca_fit, red_data = self.dim_reduce(ins)
        init_inputs = np.zeros([len(ins), len(red_data[0])])
        outputs = np.zeros([len(outs), self.dim_output])
        for i in range(len(outs)):
            init_inputs[i] = red_data[i]
            outputs[i] = outs[i]
        return pca_fit, init_inputs, outputs

    def dim_reduce(self, data):
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(data)
        sc_data = scaler.transform(data)
        pca = sklearn.decomposition.PCA(0.95)
        pca_fit = pca.fit(sc_data)
        tf_data = pca_fit.transform(sc_data)
        return pca_fit, tf_data
        

if __name__ == '__main__':
    IN = Affordances(5,3,3,0.5,1,1)
    data = IN.next()
    pdb.set_trace()
