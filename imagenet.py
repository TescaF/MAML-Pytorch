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

class ImageNet:

    def __init__(self, batchsz, k_shot, k_qry):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        prefix = "n03481172"
        fts_loc = "/home/tesca/data/imagenet/" + prefix + "/" + prefix + "_fts.npy"
        self.dataset = np.load(fts_loc)
        self.batch_size = batchsz
        self.dim_input = self.dataset[0,0].shape[0]
        self.dim_output = self.dataset[0,1].shape[0]
        print(str(self.dataset.shape[0]) + " samples loaded (Dims " + str(self.dim_input) + "x" + str(self.dim_output) + ")")

        self.num_samples_per_class = k_shot + k_qry 

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            k = np.random.randint(0, self.dataset.shape[0], self.num_samples_per_class)
            for s in k: #range(self.num_samples_per_class):
                init_inputs[func,s] = np.squeeze(self.dataset[s,0])
                outputs[func,s] = np.squeeze(self.dataset[s,1])
        return init_inputs, outputs

if __name__ == '__main__':
    IN = ImageNet(100,5,8)
    data = IN.next()
    pdb.set_trace()
