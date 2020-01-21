import time
import bz2
import gzip
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
from meta_class import Meta
from cropped_aff_data import Affordances
from os.path import expanduser
from torch import nn

def get_CAM(cam):
    cam = cam.reshape(14,14)
    #cam = torch.abs(cam)
    cam = torch.max(cam, torch.tensor(0).cuda().float())
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

    CLUSTER = True

    home = expanduser("~")
    if CLUSTER:
        fts_loc = home + "/data/fts.pgz"
        print("Loading input files...")
        #with bz2.open(fts_loc, 'rb') as handle:
        with gzip.open(fts_loc, 'rb') as handle:
            inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        print("Done")
    else:
        inputs = None

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    #logger = SummaryWriter()
    print(args)

    mode = "center"
    polar = (args.polar == 1)
    dim_output = 2
    sample_size = args.sample_size # number of images per object

    db_train = Affordances(
                       CLUSTER=CLUSTER,
                       inputs = inputs,
                       mode="center",
                       train=True,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output,
                       grasp=None)

    db_test = Affordances(
                       CLUSTER=CLUSTER,
                       inputs = inputs,
                       mode="center",
                       train=False,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=-1,
                       dim_out=dim_output,
                       grasp=None)

    db_test.output_scale = db_train.output_scale
    if CLUSTER:
        device = torch.device('cuda:0')
        save_path = home + '/data/models/model_tasksz' + str(args.task_num) + '_batchsz' + str(args.k_spt) + '_lr' + str(args.update_lr) + '_mr' + str(args.meta_lr) + '_exclude' + str(args.exclude) + '_epoch'
    else:
        device = torch.device('cuda')
        save_path = os.getcwd() + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(args.exclude) + '_epoch'
    #print(str(db_train.dim_input) + "-D input")
    if args.meta == 1:
        config = [
            ('linear', [128,1024,True]),
            ('leakyrelu', [0.01,False]),
            ('linear', [1,128,True]),
            ('reshape', [196]),
            ('bn', [196]),
            ('leakyrelu', [0.01,False]),
            ('linear', [196,196,True]),
            ('bn', [196]),
            ('leakyrelu', [0.01,False]),
            ('linear', [4,196,True])
            #('relu', [True]),
            #('bn', [196]),
            #('linear', [1,196,True])
        ]
        maml = Meta(args, config, None).to(device)
        maml.loss_fn = maml.avg_loss
        if args.load == 1:
            print("Loading model: ")
            load_path = save_path + "0_meta" + str(args.meta) + "_polar" + str(args.polar) + ".pt"
            print(load_path)
            m = torch.load(load_path)
            maml.load_state_dict(m)
            maml.eval()
    elif args.meta == 0:
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
        maml = Meta(args, config, dim_output, F.mse_loss).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    test_losses,ft_training,pt_training = [],[],[]
    k_spt = args.k_spt * sample_size
    for epoch in range(args.epoch):
        batch_x, n_spt, batch_y,pos_keys,neg_keys = db_train.next()
        x_spt = batch_x[:,:k_spt,:]
        y_spt = batch_y[:,:k_spt,:]
        x_qry = batch_x[:,k_spt:,:]
        y_qry = batch_y[:,k_spt:,:]
        x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)

        if args.meta == 1:
            test_loss, loss_report = maml.forward(n_spt, x_spt, y_spt, x_qry, y_qry)
            test_losses.append(loss_report[0])
            pt_training.append(loss_report[1])
            ft_training.append(loss_report[2])
        else:
            train_loss,test_loss,_,grad = maml(x_spt, y_spt,x_qry,y_qry)
            test_losses.append(test_loss)
            pt_training.append(train_loss)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttesting loss:', np.array(test_losses).mean(axis=0))
            print('step:', epoch, '\tft loss:', np.array(ft_training).mean(axis=0))
            print('step:', epoch, '\tpt loss:', np.array(pt_training).mean(axis=0))
            test_losses,ft_training,pt_training = [],[],[]

        if epoch % 500 == 0:  # evaluation
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_meta" + str(args.meta) + "_polar" + str(args.polar) + "-v2.pt")
            test_losses,ft_training,pt_training = [],[],[]
            batch_x,n_spt,batch_y,pos_keys,neg_keys = db_test.next()
            x_spt = batch_x[:,:k_spt,:]
            y_spt = batch_y[:,:k_spt,:]
            x_qry = batch_x[:,k_spt:,:]
            y_qry = batch_y[:,k_spt:,:]
            x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                         torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)

            t = 0
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one, n_spt_one, pos_one, neg_one in zip(x_spt,y_spt,x_qry,y_qry,n_spt,pos_keys,neg_keys):
                if args.meta == 1:
                    loss_report, pred = maml.tune(n_spt_one, x_spt_one,y_spt_one,x_qry_one,y_qry_one)
                    test_losses.append(loss_report[0])
                    pt_training.append(loss_report[1])
                    ft_training.append(loss_report[2])
                else:
                    train_loss,test_loss,w,_ = maml.finetuning(x_spt_one, y_spt_one,x_qry_one,y_qry_one)
                    test_losses.append(test_loss)
                '''if epoch % 2000 == 0: 
                    for i in range(x_spt_one.shape[0]):
                        name = pos_one[i].split("_label")[0]
                        cam_im = get_CAM(cam[0][i])
                        out_im = get_CAM(cam[1][i])
                        pos = db_train.output_scale.inverse_transform(y_spt_one[i].cpu().numpy().reshape(1,-1)).squeeze()
                        img = cv.imread(home + '/data/cropped/' + name + '_rgb.jpg')
                        # Scale target point to position in 224x224 img
                        if img is not None:
                            height, width, _ = img.shape
                            heatmap1 = cv.applyColorMap(cv.resize(cam_im.transpose(),(width, height)), cv.COLORMAP_JET)
                            heatmap_out = cv.applyColorMap(cv.resize(out_im.transpose(),(width, height)), cv.COLORMAP_JET)
                            result1 = heatmap1 * 0.3 + img * 0.5
                            result3 = heatmap_out * 0.3 + img * 0.5
                            g = "g4"
                            cv.circle(result1,(int(pos[1]),int(pos[0])),5,[255,255,0])
                            cv.imwrite('/u/tesca/data/' + g + '_imgs/min_t' + str(t) + '_ep' + str(epoch) + '_ex' + str(args.exclude) + "_" + name + '-cam.jpg', result1)
                            cv.imwrite('/u/tesca/data/' + g + '_imgs/min_t' + str(t) + '_ep' + str(epoch) + '_ex' + str(args.exclude) + "_" + name + '-out.jpg', result3)'''
                t+=1

            print('Test Loss:', np.array(test_losses).mean(axis=0))
            print('Ft Loss:', np.array(ft_training).mean(axis=0))
            print('Pt Loss:', np.array(pt_training).mean(axis=0))
            test_losses,ft_training,pt_training = [],[],[]
            


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pos_only', type=int, help='epoch number', default=0)
    argparser.add_argument('--load', type=int, help='epoch number', default=0)
    argparser.add_argument('--meta', type=int, help='epoch number', default=1)
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
    argparser.add_argument('--polar', type=int, help='epoch number', default=0)
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--feat_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
