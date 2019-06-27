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

class CornellGrasps:

    def __init__(self, batchsz, k_shot, k_qry):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        fts_loc = "/home/tesca/data/cornell_grasps/all_fts.npy"
        out_loc = "/home/tesca/data/cornell_grasps/all_outputs.npy"
        self.dataset = np.load(fts_loc)
        self.outputs = np.load(out_loc) #image, grasp, angle/x/y
        self.batch_size = batchsz
        self.dim_input = self.dataset[0].shape[0]
        self.dim_output = 2#self.outputs[0].shape[0]-1 #self.dataset[1].shape[0]
        print(str(self.dataset.shape[0]) + " samples loaded (Dims " + str(self.dim_input) + "x" + str(self.dim_output) + ")")

        self.num_samples_per_class = k_shot + k_qry 
        self.task_angles = range(-90, 90, 5) #i*60 for i in range(-1, 2)]
        #self.task_angles = [i*10 for i in range(-9, 10)]
        self.binned_data = None
        self.bin_samples()

    def bin_samples(self):
        data = dict()
        for a in range(len(self.task_angles)-1):
            data[a] = []
        for img in range(self.dataset.shape[0]):
            for grasp in range(len(self.outputs[0,img])):
                clip_angle = self.outputs[0,img][grasp][0]
                if clip_angle <= -90:
                    clip_angle += 180
                if clip_angle >= 90:
                    clip_angle -= 180
                for a in range(len(self.task_angles)-1):
                    if self.task_angles[a] < clip_angle < self.task_angles[a+1]:
                        data[a].append([np.squeeze(self.dataset[img]), self.outputs[0,img][grasp][1:]])
        self.binned_data = data 

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        tasks = np.random.randint(0, len(self.task_angles)-1, self.batch_size)
        for i in range(self.batch_size):
            t = tasks[i]
            samples = np.random.randint(0, len(self.binned_data[t]), self.num_samples_per_class)
            for j in range(self.num_samples_per_class):
                s = samples[j]
                init_inputs[i, j] = self.binned_data[t][s][0]
                outputs[i, j] = self.binned_data[t][s][1]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = CornellGrasps(100,5,8)
    data = IN.next()
