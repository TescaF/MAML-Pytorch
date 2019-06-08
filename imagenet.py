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
        self.img_dir = "/home/tesca/data/imagenet/"
        self.prefix = "n03481172"
        self.box_dir = self.img_dir + self.prefix + "/Annotation/" + self.prefix
        self.img_set = None
        with open(self.box_dir + "/obj_dims.json", 'r') as fp:
            self.img_set = json.load(fp)
        self.dataset = [] 
        self.batch_size = batchsz
        self.dim_input = 4096
        self.dim_output = 2

        self.resize = 1 
        self.num_samples_per_class = k_shot + k_qry 

        keys = list(self.img_set.keys())
        for i in range(len(keys)):
            sys.stdout.write("\rLoading image %i of %i" %(i, len(keys)))
            sys.stdout.flush()
            img_key = keys[i]
            inputs = self.features(img_key)
            outputs = self.img_set[img_key]
            self.dataset.append([inputs,outputs])
        print("\n" + str(len(keys)) + " images loaded.")

    def features(self, img_num):
        img_loc = self.img_dir + self.prefix + "/" + self.prefix + "_" + str(img_num) + ".JPEG"
        img_in = Image.open(img_loc)
        #img_in = Image.open("/home/tesca/data/imagenet/n03481172/n03481172_3.JPEG")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        scaler = transforms.Scale((224, 224))
        img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
        model = models.vgg16(pretrained=True)
        layer = model._modules.get("classifier")[-2]
        #layer = model._modules.get("avgpool")
        embedding = torch.zeros(self.dim_input)

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        model(img_var)
        hook.remove()
        return embedding

    def next(self):
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            k = np.random.randint(0, len(self.dataset), self.num_samples_per_class)
            for s in range(self.num_samples_per_class):
                init_inputs[func,s], outputs[func,s] = self.dataset[s]
        return init_inputs, outputs

if __name__ == '__main__':
    IN = ImageNet(1,2,3)
    IN.img_test()
