import rosbag
import math
import pdb
#from torch.utils.tensorboard import SummaryWriter
import  torch, os
import cv2 as cv
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
from scipy.stats import norm
import torch.nn.functional as F
from meta import Meta
from aff_data import Affordances

def get_CAM(cam):
    cam = cam.reshape(14,14)
    cam = torch.abs(cam)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    cam_img = np.uint8(255 * cam_img.cpu().detach().numpy())
    return cam_img

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    print(args)

    fts_loc = "/home/tesca/data/part-affordance-dataset/features/center_resnet_pool_fts-14D.pkl"
    with open(fts_loc, 'rb') as handle:
        inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
    categories = list(sorted(set([k.split("_")[0] for k in inputs.keys()])))
    exclude_idx=categories.index(args.name.split("_")[0])

    dim_output = 2
    sample_size = args.sample_size # number of images per object

    db_tune = Affordances(
                       mode="center",
                       train=False,
                       batchsz=1,
                       exclude=exclude_idx,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    print(str(db_tune.dim_input) + "-D input")
    device = torch.device('cuda')
    dim = db_tune.dim_input
    config = [
        ('linear', [dim,dim,True]),
        ('leakyrelu', [0.01, True]),
        ('linear', [1,dim,True]),
        ('leakyrelu', [0.01, True]),
        ('reshape',[196]),
        ('linear', [196,196,True]),
        ('leakyrelu', [0.01, True]),
        ('linear', [1,196,True])
    ]
    maml = Meta(args, config, dim_output, None, None).to(device)
    maml.loss_fn = maml.avg_loss

    ## Load model
    load_path = os.path.expanduser("~") + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(exclude_idx) + '_epoch0_meta1_polar0.pt'
    maml.load_state_dict(torch.load(load_path))
    maml.eval()
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    ## Load TFs for y_spt
    demo_num = 1
    tf_path = os.path.expanduser("~") + '/data/bags/' + args.name + "_" + str(demo_num) + ".bag"
    while os.path.exists(tf_path):
        print("-----Loading source demo " + str(demo_num) + "-----")
        tf_bag = rosbag.Bag(tf_path)
        tf_list = []
        for topic, msg, t in tf_bag.read_messages(topics=['eef_pose_relative']):
            tf_list.append(msg)
        cam_path = os.path.expanduser("~") + '/data/cam/ex' + str(exclude_idx) + '/demo' + str(demo_num) + '/'
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)
        ##tf_list = [tf_list[-1]]
        tf_i = 0 
        out_txt,tf_pred,obj_names = [],[],[]
        for tf in tf_list:
            x_spt,x_qry,n_spt,y_spt,y_qry,names = db_tune.project_tf(args.name, tf)
            x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                                 torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)
            names_spt = names[:x_spt.shape[0]]
            names_qry = names[x_spt.shape[0]:]
            loss,w,acc,res = maml.class_tune2(n_spt, x_spt,y_spt,x_qry,y_qry)
            print("KF " + str(tf_i) + " training loss: " + str(np.array([loss])))
            for i in range(len(names_qry)):
                pred = res[i].cpu().detach().numpy()
                inv_tf = db_tune.inverse_project(names_qry[i],res[i].cpu().detach().numpy())
                #print(names_qry[i] + ": " + str(inv_tf))
                obj_idx = int(np.floor(i/args.sample_size))
                if len(out_txt) <= obj_idx:
                    out_txt.append([])
                    tf_pred.append([])
                    obj_names.append(names_qry[i].split("_00")[0])
                if len(out_txt[obj_idx]) <= tf_i:
                    out_txt[obj_idx].append("")
                    tf_pred[obj_idx].append([])
                out_txt[obj_idx][tf_i] += names_qry[i] + ',' + ('%s;' % ','.join(map(str,inv_tf)))
                tf_pred[obj_idx][tf_i].append(inv_tf)
                cam1 = get_CAM(w[0][i])
                pos = [(pred[0] + 1) / (db_tune.px_to_cm * db_tune.cm_to_std[0]), (pred[1] + 1) / (db_tune.px_to_cm * db_tune.cm_to_std[1])]

                img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + names_qry[i] + '_center.jpg')
                if img is not None:
                    height, width, _ = img.shape
                    heatmap1 = cv.applyColorMap(cv.resize(cam1.transpose(),(width, height)), cv.COLORMAP_JET)
                    result1 = heatmap1 * 0.3 + img * 0.5
                    cv.circle(result1,(int(pos[1]),int(pos[0])),5,[0,255,255])
                    cv.imwrite(cam_path + names_qry[i] + '-kf_' + str(tf_i) + '.jpg', result1)
            print("Loss/Median/Mean/Variance")
            for o in range(len(tf_pred)):
                med = np.median(np.array(tf_pred[o][tf_i]),axis=0)
                mean = np.mean(np.array(tf_pred[o][tf_i]),axis=0)
                var = np.var(np.array(tf_pred[o][tf_i]),axis=0)
                stats = obj_names[o] + ": " + str(np.array([loss])) + " " + str(np.array(med)) + " " + str(np.array(mean)) + " " + str(np.array(var))
                print(stats)
                out_txt[o][tf_i] = "KF " + str(tf_i) + "(loss/median/mean/variance)" + '\n' + stats + '\n' + out_txt[o][tf_i] # + ('%s;' % ','.join(map(str,out_txt[0][tf_i][-1])))
            tf_i += 1
        print("Writing transforms to file")
        for tgt_i in range(len(out_txt)):
            tgt = out_txt[tgt_i]
            name = obj_names[tgt_i]
            file_out = os.path.expanduser("~") + '/data/output/src_' + args.name + "-demo_" + str(demo_num) + "-tgt_" + name + ".txt"
            f = open(file_out, "w")
            for kf_i in range(len(tgt)):
                kf = tgt[kf_i]
                out = ""
                for img in kf:
                   out += img #'%s;' % ','.join(map(str,img))
                f.write(out + '\n')
                #median_tf = np.median(np.array(tf_pred[tgt_i][kf_i]),axis=0)
                #mean_tf = np.mean(np.array(tf_pred[tgt_i][kf_i]),axis=0)
                #var_tf = np.var(np.array(tf_pred[tgt_i][kf_i]),axis=0)
                #f.write('%s' % ','.join(map(str,median_tf)) + '\n')
            f.close()
        demo_num += 1
        tf_path = os.path.expanduser("~") + '/data/bags/' + args.name + "_" + str(demo_num) + ".bag"

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, help='epoch number', default="")
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
