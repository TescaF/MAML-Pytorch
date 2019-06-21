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

class ImageNetData:

    def __init__(self):
        self.prefixes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        self.base_dir = "/home/tesca/data/cornell_grasps/"
        self.dim_input = 4096
        self.dims = self.load_data(self.prefixes)

    def load_data(self, prefixes):
        dims = []
        for p in prefixes:
            img_nums = [i.split(".png")[0].split("pcd")[1] for i in os.listdir(self.base_dir + p + "/") if i.endswith('.png')]
            for i in img_nums:
                img_fn = self.base_dir + p + "/cropped/pcd" + str(i) + ".png"
                grasp_fn = self.base_dir + p + "/pcd" + str(i)[:-1] + "cpos.txt"
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
                            dims.append([p, i, grasps])
                            break
        return dims

    def get_features(self):
        dataset = []
        outputs = []
        for i in range(len(self.dims)):
            sys.stdout.write("\rLoading image %i of %i" %(i, len(self.dims)))
            sys.stdout.flush()
            inputs = self.features(self.dims[i][0], self.dims[i][1])
            if inputs is None:
                print("\nError reading image " + img_key)
            else:
                dataset.append(inputs)
                outputs.append(self.dims[i][2])
        np.save(self.base_dir + "all_fts.npy", np.matrix(dataset))
        np.save(self.base_dir + "all_outputs.npy", np.matrix(outputs))

    def features(self, prefix, img_num):
        try:
            img_loc = self.base_dir + prefix + "/cropped/pcd" + str(img_num) + ".png"
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
    img_data = ImageNetData()
    data = img_data.get_features()
