# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
import pickle
from sklearn import preprocessing
import os
import torch
import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from learner import Learner

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet50(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs, pooled_features = [],[]
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def hook_pooled_feature(module, input, output):
    pooled_features.append(output.squeeze().unsqueeze(0).cuda())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)
net._modules.get("avgpool").register_forward_hook(hook_pooled_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def load_fc_weights():
    load_path = os.getcwd() + '/data/tfs/model_batchsz20_stepsz0.1_exclude-1_epoch0_reduced.pt'
    print(load_path)
    #config = [ ('linear', [104,2048])]
    config = [
        ('linear', [128,2048]),
        ('relu', [True]),
        ('linear', [105,128]),
    ]

    device = torch.device('cuda')
    maml = Learner(config).to(device)
    model = torch.load(load_path)
    keys = list(model.keys())
    for k in keys:
        model[k.split("net.")[1]] = model.pop(k)
    maml.load_state_dict(model)
    maml.eval()
    #ft_maps = torch.mm(maml.parameters()[-2],maml.parameters()[0])
    ft_maps = maml.parameters()[0] #torch.mm(maml.parameters()[-2],maml.parameters()[0])
    return maml, ft_maps

def returnCAM(feature_conv, weight_softmax, scaler):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    #for idx in class_idx:
    cam = np.zeros(49)
    for idx in range(128):
        cam += scaler.inverse_transform(weight_softmax[idx]).dot(feature_conv.reshape((nc, h*w)))
    cam = cam / 128
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

im = "mug_01"

img_pil = Image.open('/home/tesca/data/part-affordance-dataset/tools/' + im + '/' + im + '_00000001_rgb.jpg')
img_pil.save('test.jpg')

fts_loc = '/home/tesca/data/part-affordance-dataset/features/resnet_fts.pkl'
with open(fts_loc, 'rb') as handle:
    inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
input_scale = preprocessing.StandardScaler()
input_scale.fit(np.array(list(inputs.values())))

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
with torch.no_grad():
    logit = net(img_variable)
m, w = load_fc_weights()
#logit = m(torch.from_numpy(input_scale.transform(pooled_features[0].detach().cpu().numpy().reshape(1,-1))).float().cuda())
logit = m(pooled_features[0])

#h_x = F.softmax(logit, dim=1).data.squeeze()
#probs, idx = h_x.sort(0, True)
#idx = idx.detach().cpu().numpy()

# output the prediction

# generate class activation mapping for the top1 prediction
weight_softmax = w.detach().cpu().numpy()
CAMs = returnCAM(features_blobs[0], weight_softmax, input_scale)

# render the CAM and output
img = cv2.imread('/home/tesca/data/part-affordance-dataset/tools/' + im + '/' + im + '_00000001_rgb.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)


