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

def img_polar_tf(img):
    d = int(img.shape[0]/2)
    w = int(img.shape[0])
    tf = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = j/2
            a = 2*np.pi*i/w
            x = int(np.floor(r * math.cos(a))+d)
            y = int(np.floor(r * math.sin(a))+d)
            if x in range(w) and y in range(w):
                tf[j,i] = img[x,y]
    return tf

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
                       batchsz=args.task_num,
                       exclude=exclude_idx,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    print(str(db_tune.dim_input) + "-D input")
    device = torch.device('cuda')
    dim = db_tune.dim_input
    if args.polar == 1 and args.meta == 1:
        config = [
            ('linear', [dim,dim,True]),
            ('leakyrelu', [0.01, True]),
            ('linear', [1,dim,True]),
            ('leakyrelu', [0.01, True]),
            ('reshape',[14,14]),
            ('polar',[14,14]),
            ('reshape',[196]),
            ('linear', [196,196,True]),
            ('leakyrelu', [0.01, True]),
            ('linear', [1,196,True])
        ]
        maml = Meta(args, config, dim_output, None, None).to(device)
        maml.loss_fn = maml.polar_loss
    if args.polar == 0 and args.meta == 1:
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
    if args.meta == 0:
        config = [
            ('linear', [dim,dim,True]),
            ('leakyrelu', [0.01, True]),
            ('linear', [1,dim,True]),
            ('leakyrelu', [0.01, True]),
            ('reshape',[196]),
            ('linear', [196,196,True]),
            ('leakyrelu', [0.01, True]),
            ('linear', [2,196,True])
        ]
        maml = Meta(args, config, dim_output, F.mse_loss, None).to(device)

    ## Load model
    load_path = os.path.expanduser() + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(exclude_idx) + '_epoch0.pt'
    maml.load_state_dict(torch.load(load_path))
    maml.eval()
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    ## Load TF for y_spt
    tf_path = os.path.expanduser() + '/data/' + args.name + ".bag"
    tf_bag = rosbag.Bag(tf_path)
    for topic, msg, t in bag.read_messages(topics=['eef_pose_relative']):
        tf = msg
    
    x_spt,n_spt,y_spt,names,dist = db_tune.project_tf(args.name, tf)
    x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt.squeeze(0)).float().to(device), torch.from_numpy(y_spt.squeeze(0)).float().to(device), \
                                         torch.from_numpy(x_qry.squeeze(0)).float().to(device), torch.from_numpy(y_qry.squeeze(0)).float().to(device), torch.from_numpy(n_spt.squeeze(0)).float().to(device)

    loss,w,res = maml.class_finetuning(n_spt, x_spt,y_spt,x_qry,y_qry)
    for i in range(names.shape[0]):
        cam1 = get_CAM(w[0][i])
        if len(w) > 1:
            cam2 = get_CAM(w[1][i])
        else:
            cam2 = None
        pos = db_train.output_scale.inverse_transform(y_spt_one[i].cpu().numpy().reshape(1,-1)).squeeze()
        img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + names[i] + '_center.jpg')
        # Scale target point to position in 224x224 img
        mult = [(pos[0] * 224/img.shape[0])-112, (pos[1] * 224/img.shape[1])-112]
        r = 224*np.sqrt(mult[0]**2 + mult[1]**2)/(0.5*np.sqrt(2*(224**2)))
        a = (224/(2*np.pi)) * math.atan2(mult[1],mult[0]) % 224
        tf_pos = [r,a]
        tf_img = img_polar_tf(cv.resize(img, (224,224)))
        if img is not None:
            height, width, _ = img.shape
            heatmap1 = cv.applyColorMap(cv.resize(cam1.transpose(),(width, height)), cv.COLORMAP_JET)
            result1 = heatmap1 * 0.3 + img * 0.5
            cv.circle(result1,(int(pos[1]),int(pos[0])),5,[255,255,0])
            cv.imwrite('data/cam/polar' + str(args.polar) + 'meta' + str(args.meta) +'/ex' + str(exclude_idx) + '/' + name + '_t' + str(t) + '.jpg', result1)
            if polar and (cam2 is not None):
                height, width, _ = tf_img.shape
                heatmap2 = cv.applyColorMap(cv.resize(cam2,(width, height)), cv.COLORMAP_JET)
                result2 = heatmap2 * 0.3 + tf_img * 0.5
                cv.circle(result2,(int(tf_pos[1]),int(tf_pos[0])),5,[255,255,0])
                cv.imwrite('data/cam/' + name + 'ex' + str(exclude_idx) + '_CAM_polar.jpg', result2)

    #torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_meta" + str(args.meta) + "_polar" + str(args.polar) + ".pt")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=int, help='epoch number', default="")
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
