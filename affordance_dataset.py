import pdb
import sklearn.cluster
from sklearn.linear_model import LinearRegression
import scipy.io
import numpy as np
import sys
import  os.path
import math
from os import path
import cv2 as cv
import pickle 
from matplotlib import pyplot as plt
from itertools import product
import quaternion
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as models


class ImageProc:

    def __init__(self):
        self.base_dir = "/home/tesca/data/part-affordance-dataset/"
        self.temp_dir = "templates/"

    def process_images(self):
        temp_ll = cv.imread(self.base_dir + self.temp_dir + "checker_lower_left.png",1)
        temp_ur = cv.imread(self.base_dir + self.temp_dir + "checker_upper_right.png",1)
        temp_lr = cv.imread(self.base_dir + self.temp_dir + "checker_lower_right.png",1)
        temp_ul = cv.imread(self.base_dir + self.temp_dir + "checker_upper_left.png",1)
        temps = [temp_ul,temp_ur,temp_lr,temp_ll]
        categories = [x for x in os.listdir(self.base_dir + "tools") if os.path.isdir(self.base_dir + "tools/" + x)]
        #categories = ["cup_02", "knife_06", "saw_03", "scissors_02", "trowel_01"]
        '''fixed_corners =  [[[164,151],[477,162],[542,385],[79,370]],
                    [[150,137],[460,137],[519,345],[81,345]],
                    [[167,128],[474,149],[500,363],[68,322]],
                    [[151,135],[457,136],[518,344],[77,340]],
                    [[148,140],[453,133],[522,336],[87,348]]]'''
        i = 0
        for cat in categories:
            # Load images
            img = cv.imread(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_rgb.jpg',1)
            label = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_label.mat')
            label = label['gt_label']
            rank = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_label_rank.mat')
            rank = rank['gt_label']

            # Warp RGB image
            corners = self.get_corners(img, temps)
            #corners = fixed_corners[i]
            warped = self.warp_image(img, corners)
            rsz = self.crop_image(warped)
            cv.imwrite(self.base_dir + "cropped/" + cat + "_rgb.jpg", rsz)

            # Similarly transform ranked affordance data
            w_label = self.warp_image(label, corners)
            rsz_label = self.crop_image(w_label)
            #cv.imwrite(self.base_dir + "cropped/" + cat + "_label.jpg", rsz_label)
            np.save(self.base_dir + "cropped/" + cat + "_label.npy", np.array(rsz_label))
            w_rank = self.warp_image(rank, corners)
            rsz_rank = self.crop_image(w_rank)
            np.save(self.base_dir + "cropped/" + cat + "_rank.npy", np.array(rsz_rank))
            #i+=1

    def crop_image(self, image):
            x, y = image.shape[::-1][-2:]
            w = 440
            h = 210
            if x < w:
                m = w/x
                image = cv.resize(image, (int(m*x),int(m*y)))
                x, y = image.shape[::-1][-2:]
            if y < h:
                m = h/y
                image = cv.resize(image, (int(m*x),int(m*y)))
                x, y = image.shape[::-1][-2:]
            sx = x//2-(w//2)
            sy = y//2-(h//2)
            return image[sy:sy+h, sx:sx+w]

    def get_corners(self, image, temps):
        dims = []
        pos = []
        for t in temps:
            dims.append(t.shape[::-1]) #[w,h]
            match = cv.matchTemplate(image, t, cv.TM_CCOEFF)
            pos.append([np.float32(cv.minMaxLoc(match)[-1][0]), np.float32(cv.minMaxLoc(match)[-1][1])])
        pos[0][0] += 5
        pos[0][1] += 5
        pos[1][0] += dims[1][1] - 10
        pos[1][1] += 5
        pos[2][0] += dims[2][1] - 10
        pos[2][1] += dims[2][2] - 5
        pos[3][0] += 10
        pos[3][1] += dims[3][2] - 10
        #loc_upper = (loc_upper[0] loc_upper[1])
        #loc_lower = (loc_lower[0], loc_upper[1] + 220)
        return pos

    def warp_image(self, image, corners):
        (tl, tr, br, bl) = corners

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv.getPerspectiveTransform(np.float32(corners), dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def get_normal(self, obj_name, aff, interest_pt="centroid", base = None, disp_img=None, arrow_color="blue"):
        data = np.load(self.base_dir + "cropped/" + obj_name + "_label.npy")
        if disp_img is None:
            disp_img = np.zeros(data.shape)
        aff_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == aff]
        if len(aff_pts) == 0:
            return None, disp_img
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(aff_pts)
        disp_pts = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
        if len(disp_pts) == 0:
            return None, disp_img
        #for p in disp_pts:
        #    disp_img[p] = 1

        # Get centerpoint
        if interest_pt is "centroid":
            cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
        if interest_pt is "farthest":
            dists = [math.sqrt((base[0] - p[0])**2.0 + (base[1] - p[1])**2.0) for p in disp_pts]
            i = dists.index(max(dists))
            cy, cx = disp_pts[i]

        # Get edge normal
        aff_mat = np.matrix(disp_pts)
        reg = LinearRegression().fit(aff_mat[:,0], aff_mat[:,1])
        normal = math.atan(reg.coef_) + math.pi/2.0
        dx = int(math.cos(normal) * 10)
        dy = int(math.sin(normal) * 10)
        #plt.arrow(cx, cy, dx, dy, head_width=5, color=arrow_color)
        if len(disp_img.shape) > 2:
            cvt = cv.cvtColor(disp_img, cv.COLOR_BGR2RGB)
        else:
            cvt = disp_img
        #plt.imshow(cvt)
        return [cx, cy, normal], disp_img

    def tf_result(self, obj_name, aff):
        grasp, _ = self.get_normal(obj_name, 1)
        if grasp is None:
            return np.array([None]*3)
        goal, _ = self.get_normal(obj_name, aff, interest_pt="farthest", base = grasp[:2])
        if goal is None:
            return np.array([None]*3)
        tf = self.get_tf(grasp, goal)
        return grasp, goal, tf

    def get_tf(self, pt_a, pt_b):
        return np.array([pt_b[0]-pt_a[0], pt_b[1]-pt_a[1], pt_b[2]-pt_a[2]])
        
    def show_transforms(self, obj_name, aff):
        grasp, goal, tf = self.tf_result(obj_name, aff)
        try:
            img_loc = self.base_dir + "cropped/" + obj_name + "_rgb.jpg"
            img_in = Image.open(img_loc)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))))
            img_show = (img_var.permute(1,2,0) + 2.5)/5.0
            plt.imshow(img_show)
            grasp_x = grasp[0]*(224/img_in.size[0])
            grasp_y = grasp[1]*(224/img_in.size[1])
            aff_x = goal[0]*(224/img_in.size[0])
            aff_y = goal[1]*(224/img_in.size[1])
            plt.scatter([grasp_x],[grasp_y],color='r')
            plt.scatter([aff_x],[aff_y],color='g')
            plt.show()
        except:
            return None    
        

    def save_transforms(self, transform=True):
        obj_splits = [line.rstrip('\n') for line in open(self.base_dir + "novel_split.txt")]
        splits = dict()
        affs = range(2,7)
        for a in affs:
            aff_dict = dict()
            pos_dict = dict()
            print("Affordance " + str(a))
            for obj in obj_splits:
                obj_name = obj.split()[1]
                split_num = obj.split()[0]
                grasp, goal, tf = self.tf_result(obj_name, a)
                tf = [a, tf]
                pos = [a, grasp, goal]
                print(obj_name + ": " + str(tf))
                aff_dict[obj_name] = tf
                pos_dict[obj_name] = pos
                if not split_num in splits.keys():
                    splits[split_num] = dict()
                if obj_name in splits[split_num].keys():
                    splits[split_num][obj_name] += [tf]
                else:
                    splits[split_num][obj_name] = [tf] 
            if transform:
                with open(self.base_dir + "features/aff_" + str(a) + "_transforms.pkl", 'wb') as handle:
                    pickle.dump(aff_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.base_dir + "features/aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                    pickle.dump(pos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #for s in splits.keys():
        #    with open(proc.base_dir + "features/obj_split_" + s + "_transforms.pkl", 'wb') as handle:
        #        pickle.dump(splits[s], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def proc_features(self):
        objs = [i.split("_rgb")[0] for i in os.listdir(self.base_dir + "cropped") if i.endswith('.jpg')]
        dataset = dict()
        i = 0
        for k in objs: #i in range(len(self.dims)):
            sys.stdout.write("\rLoading image %i of %i" %(i, len(objs)))
            sys.stdout.flush()
            inputs = self.features(k) #self.dims[i][0], self.dims[i][1])
            if inputs is None:
                print("\nError reading image " + k)
                pdb.set_trace()
            else:
                dataset[k] = inputs
            i += 1
        with open(self.base_dir + "features/all_fts.pkl", 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dataset

    def features(self, img_name): #prefix, img_num):
        try:
            img_loc = self.base_dir + "cropped/" + img_name + "_rgb.jpg"
            img_in = Image.open(img_loc)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            model = models.vgg16(pretrained=True)
            layer = model._modules.get("classifier")[-2]
            embedding = torch.zeros(4096) #self.dim_input)
        except:
            return None

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        model(img_var)
        hook.remove()
        return embedding.numpy()


if __name__ == '__main__':
    proc = ImageProc()
    #proc.show_transforms("mug_01",5)
    #proc.show_transforms("mug_01",3)
    proc.save_transforms(transform=False)
    #proc.proc_features()
