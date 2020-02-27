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
from meta_basic import Meta
from grid_aff_data import Affordances
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

def main():

    home = expanduser("~")

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    #logger = SummaryWriter()
    print(args)

    dim_output = 14
    sample_size = args.sample_size # number of images per object

    db_train = Affordances(
                       inputs = None, #inputs,
                       train=True,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    db_test = Affordances(
                       inputs = None, # inputs,
                       train=False,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    device = torch.device('cuda:0')
    save_path = home + '/data/models/model_tasksz' + str(args.task_num) + '_batchsz' + str(args.k_spt) + '_lr' + str(args.update_lr) + '_mr' + str(args.meta_lr) + '_exclude' + str(args.exclude) + '_epoch'
    base_config = [
        ('conv2d', [64, 4, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [1, 64, 1, 1, 1, 0])
    ]
    maml = Meta(device, args, base_config).to(device)
    if args.load == 1:
        print("Loading model: ")
        load_path = save_path + "0-v2.pt"
        print(load_path)
        m = torch.load(load_path)
        maml.load_state_dict(m)
        maml.eval()
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    test_losses,ft_training,pt_training = [],[],[]
    k_spt = args.k_spt * sample_size
    for epoch in range(args.epoch):
        pos_keys,batch_y = db_train.next()
        spt_keys, qry_keys = [], []
        for k in pos_keys:
            spt_keys.append(k[:k_spt])
            qry_keys.append(k[k_spt:])
        y_spt = batch_y[:,:k_spt,:]
        y_qry = batch_y[:,k_spt:,:]
        y_spt, y_qry = torch.from_numpy(y_spt).float().to(device), torch.from_numpy(y_qry).float().to(device)

        test_loss, loss_report = maml.forward(spt_keys, y_spt, qry_keys, y_qry)
        test_losses.append(loss_report[0])
        pt_training.append(loss_report[1])
        ft_training.append(loss_report[2])

        if epoch % 30 == 0:
            print('step:', epoch, '\ttesting loss:', np.array(test_losses).mean(axis=0))
            print('step:', epoch, '\tft loss:', np.array(ft_training).mean(axis=0))
            print('step:', epoch, '\tpt loss:', np.array(pt_training).mean(axis=0))
            test_losses,ft_training,pt_training = [],[],[]

        if epoch % 500 == 0:  # evaluation
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "-v2.pt")
            test_losses,ft_training,pt_training = [],[],[]
            pos_keys,batch_y = db_test.next()
            y_spt = batch_y[:,:k_spt,:]
            y_qry = batch_y[:,k_spt:,:]
            y_spt, y_qry = torch.from_numpy(y_spt).float().to(device), torch.from_numpy(y_qry).float().to(device)

            t = 0
            for y_spt_one, y_qry_one, pos_one in zip(y_spt,y_qry,pos_keys):
                spt_keys = pos_one[:k_spt]
                qry_keys = pos_one[k_spt:]
                loss_report, pred = maml.tune(spt_keys,y_spt_one,qry_keys,y_qry_one)
                #loss_report, pred = maml.tune(n_spt_one, x_spt_one,y_spt_one,x_qry_one,y_qry_one)
                test_losses.append(loss_report[0])
                pt_training.append(loss_report[1])
                ft_training.append(loss_report[2])
                t+=1

            print('Test Loss:', np.array(test_losses).mean(axis=0))
            print('Ft Loss:', np.array(ft_training).mean(axis=0))
            print('Pt Loss:', np.array(pt_training).mean(axis=0))
            test_losses,ft_training,pt_training = [],[],[]
            


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load', type=int, help='epoch number', default=0)
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
