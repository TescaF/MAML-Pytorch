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

    def get_centered_img(self, img, data):
        # Get centerpoint
        grasp_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == 1]
        if len(grasp_pts) == 0:
            return None,None
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=20).fit_predict(grasp_pts)
        disp_pts = [grasp_pts[i] for i in range(len(grasp_pts)) if clusters[i] > -1]
        if len(disp_pts) == 0:
            return None,None
        cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
        #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        #plt.scatter([cx],[cy])
        #plt.show()
        value = min([cy, cx, img.shape[0] - cy, img.shape[1] - cx])
        img_tf = cv.linearPolar(img, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
        feats = self.features_from_img(img_tf)

        # Get affordance point
        label_tf = cv.linearPolar(data, (cx,cy), value, cv.WARP_FILL_OUTLIERS)
        aff_data = []
        for a in range(2,7):
            aff_pts = [(i,j) for i in range(label_tf.shape[0]) for j in range(label_tf.shape[1]) if label_tf[i,j] == a]
            if len(aff_pts) == 0:
                aff_data.append(None)
            else:
                clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=5).fit_predict(aff_pts)
                disp_pts = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
                if len(disp_pts) == 0:
                    aff_data.append(None)
                else:
                    dists = [p[1] for p in disp_pts]
                    #dists = [math.sqrt((cy - p[1])**2.0 + (cx - p[0])**2.0) for p in disp_pts]
                    i = np.argmax(np.array(dists)) #dists.index(max(dists))
                    aff_data.append(disp_pts[i])
                    #plt.imshow(cv.cvtColor(img_tf, cv.COLOR_BGR2RGB))
                    #plt.scatter([aff_data[-1][1]],[aff_data[-1][0]])
                    #plt.show()
        return aff_data, feats
        
    def convert_to_polar(self):
        depths, labels, images = [], [], []
        dirs = os.listdir(self.base_dir + "tools/")
        pos_dict = []
        features = dict()
        for i in range(2,7):
            pos_dict.append(dict())
        c2 = 0
        for d in dirs:
            print("Directory " + str(c2) + " of " + str(len(dirs)))
            if os.path.isdir(self.base_dir + "tools/" + d):
                files = os.listdir(self.base_dir + "tools/" + d)
                objs = [i for i in files if i.endswith('label.mat') and int(i.split(d+"_")[1].split("_label")[0])%3==0]
                c1 = 0
                for o in objs:
                    sys.stdout.write("\rFile %i of %i" %(c1, len(objs)))
                    sys.stdout.flush()
                    label = scipy.io.loadmat(self.base_dir + "tools/" + d + "/" + o)
                    label = label['gt_label']
                    labels.append([o.split("_label")[0],label])
                    depths.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'depth.png',-1))
                    images.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'rgb.jpg',-1))
                    aff_data, feats = self.get_centered_img(images[-1], labels[-1][1])
                    if aff_data is not None:
                        features[labels[-1][0]] = feats
                        for a in range(2,7):
                            pos = [a, aff_data[a-2]]
                            pos_dict[a-2][labels[-1][0]] = pos
                    c1+=1
            c2+=1
        print("Saving poses...")
        for a in range(2,7):
            with open(self.base_dir + "features/polar_aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                pickle.dump(pos_dict[a-2], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.base_dir + "features/polar_fts.pkl", 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
    def process_all_images(self):
        depths, labels = [], []
        dirs = os.listdir(self.base_dir + "tools/")
        for d in dirs:
            if os.path.isdir(self.base_dir + "tools/" + d):
                files = os.listdir(self.base_dir + "tools/" + d)
                objs = [i for i in files if i.endswith('label.mat') and int(i.split(d+"_")[1].split("_label")[0])%3==0]
                for o in objs:
                    label = scipy.io.loadmat(self.base_dir + "tools/" + d + "/" + o)
                    label = label['gt_label']
                    labels.append([o.split("_label")[0],label])
                    depths.append(cv.imread(self.base_dir + "tools/" + d + '/' + o.split("label")[0] + 'depth.png',-1))
        affs = range(2,7)
        for a in affs:
            pos_dict = dict()
            for l in range(len(labels)):
                label = labels[l]
                grasp, _ = self.get_normal_from_data(label[1], 1, depth_img=depths[l])
                if grasp is None:
                    grasp = np.array([None]*3)
                    goal, _ = self.get_normal_from_data(label[1], a, depth_img=depths[l])
                else:
                    goal, _ = self.get_normal_from_data(label[1], a, interest_pt="farthest", base = grasp[:2], depth_img=depths[l])
                if goal is None:
                    goal = np.array([None]*3)
                pos = [a, grasp, goal]
                print(label[0] + ": " + str(pos))
                pos_dict[label[0]] = pos
            with open(self.base_dir + "features/aff_" + str(a) + "_positions.pkl", 'wb') as handle:
                pickle.dump(pos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            
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
            img = cv.imread(self.base_dir + "tools/" + cat + '/' + cat + '_00000072_rgb.jpg',1)
            #img = cv.imread(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_rgb.jpg',1)
            label = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000072_label.mat')
            #label = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_label.mat')
            label = label['gt_label']
            rank = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000072_label_rank.mat')
            #rank = scipy.io.loadmat(self.base_dir + "tools/" + cat + '/' + cat + '_00000001_label_rank.mat')
            rank = rank['gt_label']

            # Warp RGB image
            corners = self.get_corners(img, temps)
            #corners = fixed_corners[i]
            warped = self.warp_image(img, corners)
            rsz = self.crop_image(warped)
            cv.imwrite(self.base_dir + "cropped/" + cat + "_rgb-1.jpg", rsz)

            # Save rotated version
            r1_img = cv.flip(rsz,1)
            cv.imwrite(self.base_dir + "cropped/" + cat + "_rgb-2.jpg", r1_img)

            # Similarly transform ranked affordance data
            w_label = self.warp_image(label, corners)
            rsz_label = self.crop_image(w_label)
            #cv.imwrite(self.base_dir + "cropped/" + cat + "_label.jpg", rsz_label)
            np.save(self.base_dir + "cropped/" + cat + "_label-1.npy", np.array(rsz_label))
            w_rank = self.warp_image(rank, corners)
            rsz_rank = self.crop_image(w_rank)
            np.save(self.base_dir + "cropped/" + cat + "_rank-2.npy", np.array(rsz_rank))

            r1_label = cv.flip(rsz_label,1)
            np.save(self.base_dir + "cropped/" + cat + "_label-1.npy", np.array(r1_label))
            r1_rank = cv.flip(rsz_rank,1)
            np.save(self.base_dir + "cropped/" + cat + "_rank-2.npy", np.array(r1_rank))
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
        obj_name, num = obj_name.split("-")
        data = np.load(self.base_dir + "cropped/" + obj_name + "_label-" + str(num) + ".npy")
        return self.get_normal_from_data(data, aff, interest_pt, base, disp_img, arrow_color)

    def get_normal_from_data(self, data, aff, interest_pt="centroid", base = None, disp_img=None, arrow_color="blue",depth_img=None):
        if disp_img is None:
            disp_img = np.zeros(data.shape)
        aff_pts = [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1]) if data[i,j] == aff]
        if len(aff_pts) == 0:
            return None, disp_img
        clusters = sklearn.cluster.DBSCAN(eps=3, min_samples=5).fit_predict(aff_pts)
        disp_pts = [aff_pts[i] for i in range(len(aff_pts)) if clusters[i] > -1]
        if len(disp_pts) == 0:
            return None, disp_img
        #for p in disp_pts:
        #    disp_img[p] = 1

        # Get centerpoint
        if interest_pt is "centroid":
            cy, cx = [int(x) for x in np.median(disp_pts,axis=0)]
        if interest_pt is "farthest":
            dists = [math.sqrt((base[0] - p[1])**2.0 + (base[1] - p[0])**2.0) for p in disp_pts]
            i = np.argmax(np.array(dists)) #dists.index(max(dists))
            cy, cx = disp_pts[i]
        if not depth_img is None:
            cz = depth_img[cy,cx]

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
        #plt.show()
        return [cx, cy, cz, normal], disp_img

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
        obj_name, num = obj_name.split("-")
        try:
            img_loc = self.base_dir + "cropped/" + obj_name + "_rgb-" + str(num) + ".jpg"
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
            pdb.set_trace()
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
                for num in range(2):
                    obj_name = obj.split()[1] + "-" + str(num+1)
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
        objs = []
        dirs = os.listdir(self.base_dir + "tools/")
        for d in dirs:
            if os.path.isdir(self.base_dir + "tools/" + d):
                files = os.listdir(self.base_dir + "tools/" + d)
                objs += [d + "/" + i for i in files if i.endswith('.jpg') and int(i.split(d+"_")[1].split("_rgb")[0])%3==0]
        #objs = [i.split("_rgb")[0] for i in os.listdir(self.base_dir + "cropped") if i.endswith('.jpg')]
        dataset = dict()
        i = 0
        for k in objs: #i in range(len(self.dims)):
            sys.stdout.write("\rLoading image %i of %i" %(i, len(objs)))
            sys.stdout.flush()
            inputs1 = self.features(k,1) #self.dims[i][0], self.dims[i][1])
            #inputs1 = self.features(k,2) #self.dims[i][0], self.dims[i][1])
            if inputs1 is None:
                print("\nError reading image " + k)
                pdb.set_trace()
            else:
                dataset[k.split("/")[1].split("_rgb")[0]] = inputs1
            #if inputs2 is None:
            #    print("\nError reading image " + k)
            #    pdb.set_trace()
            #else:
            #    dataset[k+"-2"] = inputs2
            i += 1
        with open(self.base_dir + "features/all_fts.pkl", 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dataset

    def features(self, img_name,img_num): #prefix, img_num):
        img_loc = self.base_dir + "tools/" + img_name
        #img_loc = self.base_dir + "cropped/" + img_name + "_rgb-" + str(img_num) + ".jpg"
        img_in = Image.open(img_loc)
        return self.features_from_img(img_in)

    def features_from_img(self, img):
        try:
            img_in = Image.fromarray(np.uint8(img)*255)
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

    def reduce_features(self):
        var = 0.9
        with open(self.base_dir + "features/all_fts.pkl", 'rb') as handle:
            fts = pickle.load(handle)
        keys = list(fts.keys())
        scaler = sklearn.preprocessing.StandardScaler()
        vals = list(fts.values())
        scaler.fit(vals)
        sc_data = scaler.transform(vals)
        pca = sklearn.decomposition.PCA(var)
        pca_fit = pca.fit(sc_data)
        for k in keys:
            a = scaler.transform(fts[k].reshape(1,-1))
            fts[k] = pca_fit.transform(a).squeeze(0)
        with open(self.base_dir + "features/reduced_fts_" + str(var) + ".pkl", 'wb') as handle:
            pickle.dump(fts, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    proc = ImageProc()
    #proc.process_images()
    #proc.show_transforms("mug_01-2",4)
    #proc.show_transforms("mug_01",3)
    #proc.save_transforms(transform=False)
    #proc.process_all_images()
    proc.convert_to_polar()
    #proc.reduce_features()
    #proc.proc_features()
