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
from copy import deepcopy

def get_CAM(cam):
    cam = cam.reshape(14,14)
    #cam = torch.abs(cam)
    cam = torch.max(cam.cpu(),torch.tensor(0).float())
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    cam_img = np.uint8(255 * cam_img.cpu().detach().numpy())
    return cam_img

def quaternion_matrix(quaternion):
    ## From ROS TF transformations.py
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def quaternion_from_matrix(matrix):
    ## From ROS TF transformations.py
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def inverse_tf_pose(p, goal_tf):
    trans = p.pose.position
    rot = p.pose.orientation
    trans_i, rot_i = inverse_tf([trans.x,trans.y,trans.z],[rot.x,rot.y,rot.z,rot.w],goal_tf)
    pi = deepcopy(p)
    pi.pose.position.x = trans_i[0]
    pi.pose.position.y = trans_i[1]
    pi.pose.position.z = trans_i[2]
    pi.pose.orientation.x = rot_i[0]
    pi.pose.orientation.y = rot_i[1]
    pi.pose.orientation.z = rot_i[2]
    pi.pose.orientation.w = rot_i[3]
    return pi

def inverse_tf(trans, rot, goal_tf):
    trans_m = np.identity(4)
    # Put into goal space
    trans_m[:3,3] = [trans[0]-goal_tf[0], trans[1]-goal_tf[1], trans[2]-goal_tf[2]]

    rot_m = quaternion_matrix([rot[0],rot[1],rot[2],rot[3]])

    G = np.identity(4)
    G = np.dot(G, trans_m)
    G = np.dot(G, rot_m)
    # Inverse tf into ee space
    inv = np.linalg.inv(G) #tr.inverse_matrix(inv)
    trans_i = np.array(inv, copy=False)[:3, 3].copy() # tr.translation_from_matrix(inv)
    rot_i = quaternion_from_matrix(inv)
    return [trans_i, rot_i]


def main():

    CLUSTER = True

    if CLUSTER:
        import bz2
    else:
        import rosbag
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    print(args)

    if args.name.startswith("ladle"):
        if args.t == 1:
            goal_tf = [0.039,0.442,0.04]
            grasp = ["end", np.matrix([[1,0],[0,0],[0,-1]])]
        elif args.t == 2:
            goal_tf = [-0.132,0.74,0.38]
            grasp = ["center", np.matrix([[0,0],[1,0],[0,-1]])]
        else:
            goal_tf = [0.039,0.442,0.04]
            grasp = ["end", np.matrix([[1,0],[0,0],[0,-1]])]
    if args.name.startswith("saw"):
        if args.t == 1:
            goal_tf = [-0.182,0.645,-0.009]
            grasp = ["end", np.matrix([[1,0],[0,-1],[0,0]])]
        elif args.t == 2:
            goal_tf = [0.005,-0.451,0.38]
            grasp = ["end", np.matrix([[1,0],[0,-1],[0,0]])]
    if args.name.startswith("turner"):
        if args.t == 1:
            goal_tf = [-0.184,0.63,-0.025]
            grasp = ["end", np.matrix([[1,0],[0,0],[0,-1]])]

    if CLUSTER:
        device = torch.device('cuda:0')
        fts_loc = os.path.expanduser("~") + "/data/fts.pbz2"
        print("Loading input files...")
        with bz2.open(fts_loc, 'rb') as handle:
            inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        print("Done")
    else:
        device = torch.device('cuda')
        fts_loc = "/home/tesca/data/part-affordance-dataset/features/center_resnet_pool_fts-14D.pkl"
        with open(fts_loc, 'rb') as handle:
            inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
    categories = list(sorted(set([k.split("_")[0] for k in inputs.keys()])))
    exclude_idx=categories.index(args.name.split("_")[0])

    dim_output = 2
    sample_size = args.sample_size # number of images per object

    db_tune = Affordances(
                       CLUSTER=CLUSTER,
                       inputs=inputs,
                       mode="center",
                       train=False,
                       batchsz=1,
                       exclude=exclude_idx,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output,
                       grasp=grasp)

    print(str(db_tune.dim_input) + "-D input")
    dim = db_tune.dim_input
    config = [
            ('linear', [128,1024,True]),
            ('relu', [True]),
            ('linear', [1,128,True]),
            ('relu', [True]),
            ('reshape', [196]),
            ('bn', [196]),
            ('linear', [196,196,True])
    ]
    '''('linear', [dim,dim,True]),
    ('leakyrelu', [0.01, True]),
    ('linear', [1,dim,True]),
    ('leakyrelu', [0.01, True]),
    ('reshape',[196]),
    ('linear', [196,196,True]),
    ('leakyrelu', [0.01, True]),
    ('linear', [1,196,True])'''
    maml = Meta(args, config, dim_output, None, None).to(device)
    maml.loss_fn = maml.avg_loss

    ## Load model
    load_path = os.path.expanduser("~") + '/data/models/model_tasksz' + str(args.task_num) + '_batchsz' + str(args.k_spt) + '_lr' + str(args.update_lr) + '_mr' + str(args.meta_lr) + '_lambda' + str(args.lmb) + '_exclude' + str(exclude_idx) + '_epoch0_meta1_polar0-rv.pt'
    print("Loading model: ")
    print(load_path)
    maml.load_state_dict(torch.load(load_path))
    maml.eval()
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    ## Load TFs for y_spt
    demo_num = 1
    if CLUSTER:
        with open(os.path.expanduser("~") + "/data/kfs.txt") as f:
            kf_file = f.readlines()
        kf_file = [x.strip() for x in kf_file]
        k_line = 0
    else:
        tf_path = os.path.expanduser("~") + '/test_bagfiles/' + args.name + '_t' + str(args.t) + "_" + str(demo_num) + ".bag"
    #tf_path = os.path.expanduser("~") + '/data/bags/' + args.name + "_" + str(demo_num) + ".bag"
    condition = True
    while condition:
        if CLUSTER:
            name = kf_file[k_line]
            while k_line < len(kf_file) and not name.startswith(args.name + "_t" + str(args.t)):
                k_line += 1
                name = kf_file[k_line]
            condition = k_line < len(kf_file) 
        else:
            condition = os.path.exists(tf_path)
        if condition:
            print("-----Loading source demo " + str(demo_num) + "-----")
            tf_list = []
            if CLUSTER:
                k_line += 2
                line = kf_file[k_line].split(",")
                while len(line) > 1:
                    line = [float(l) for l in line[4:]]
                    tf_list.append(inverse_tf(line[:3],line[3:],goal_tf))
                    k_line += 1
                    if k_line < len(kf_file):
                        line = kf_file[k_line].split(",")
                    else:
                        line = []
                cam_path = os.path.expanduser("~") + '/data/cam/ex' + str(exclude_idx) + '/demo' + str(demo_num) + '/'
            else:
                tf_bag = rosbag.Bag(tf_path)
                for topic, msg, t in tf_bag.read_messages(topics=['eef_pose_j2s7s300_link_base']):
                    tf_list.append(inverse_tf_pose(msg,goal_tf))
                cam_path = os.path.expanduser("~") + '/data/cam/ex' + str(exclude_idx) + '/demo' + str(demo_num) + '/'
                if not os.path.exists(cam_path):
                    os.makedirs(cam_path)
            #tf_list = [tf_list[-1]]
            tf_i = 0
            out_txt,tf_pred,obj_names = [],[],[]
            for tf in tf_list:
                x_spt,x_qry,n_spt,y_spt,y_qry,names_qry = db_tune.project_tf(args.name, tf)
                #x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                #                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)
                for i in range(len(names_qry[10:20])):
                    pos = db_tune.output_scale.inverse_transform(np.array(y_spt[i]).reshape(1,-1)).squeeze(0)
                    img = cv.imread('/u/tesca/data/center_tools/' + names_qry[10 + i] + '_center.jpg')
                    cv.circle(img,(int(pos[1]),int(pos[0])),5,[0,255,255])
                    cv.imwrite(cam_path + names_qry[i] + '-kf_' + str(tf_i) + '.jpg', img)
                '''
                loss,w,all_losses,res = maml.class_tune4(n_spt, x_spt,y_spt,x_qry,y_qry)
                #print(all_losses[1])
                #print(all_losses[2])
                invs = []
                print("KF " + str(tf_i) + " training loss: " + str(np.array(loss)))
                #inv_spt = db_tune.inverse_project(names_qry[21],res[21].cpu().detach().numpy())
                #print(str([tf.pose.position.x,tf.pose.position.z]) + " vs " + str(inv_spt))
                for i in range(len(names_qry)):
                    pred = res[i].cpu().detach().numpy()
                    inv_tf = db_tune.inverse_project(names_qry[i], res[i].cpu().detach().numpy())
                    invs.append(inv_tf)
                    #print(names_qry[i] + " - " + str(pred) + " - " + str(inv_tf))
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
                    pos1 = [(pred[0] + 1) / (db_tune.px_to_cm * db_tune.cm_to_std[0]), (pred[1] + 1) / (db_tune.px_to_cm * db_tune.cm_to_std[1])]
                    pos = db_tune.output_scale.inverse_transform(np.array(pred).reshape(1,-1)).squeeze(0)

                    #print(names_qry[i] + ": " + str(pos1) + " " + str(pos))

                    if CLUSTER:
                        img = cv.imread('/u/tesca/data/center_tools/' + names_qry[i] + '_center.jpg')
                    else:
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
                    out_txt[o][tf_i] = "KF " + str(tf_i) + "(loss/median/mean/variance)" + '\n' + stats + '\n' + out_txt[o][tf_i] # + ('%s;' % ','.join(map(str,out_txt[0][tf_i][-1])))'''
                #print(invs[10:20])
                tf_i += 1
            print("Writing transforms to file")
            for tgt_i in range(len(out_txt)):
                tgt = out_txt[tgt_i]
                name = obj_names[tgt_i]
                file_out = os.path.expanduser("~") + '/data/output/src_' + args.name + "-demo_" + str(demo_num) + "-tgt_" + name + "-t_" + str(args.t) + ".txt"
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
        #tf_path = os.path.expanduser("~") + '/data/bags/' + args.name + "_" + str(demo_num) + ".bag"
        tf_path = os.path.expanduser("~") + '/test_bagfiles/' + args.name + '_t' + str(args.t) + "_" + str(demo_num) + ".bag"

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, help='epoch number', default="")
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--task_num', type=int, help='epoch number', default=5)
    argparser.add_argument('--t', type=int, help='epoch number', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--lmb', type=float, help='task-level inner update learning rate', default=1.0)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
