import gzip
import bz2
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

def main():

    CLUSTER = True

    home = expanduser("~")
    if CLUSTER:
        #fts_loc = home + "/data/fts.pbz2"
        fts_loc = home + "/data/fts.pgz"
        print("Loading input files...")
        #with bz2.open(fts_loc, 'rb') as handle:
        with gzip.open(fts_loc, 'rb') as handle:
            inputs = pickle.load(handle)       #dict(img) = [[4096x1], ... ]
        ex_list = ('saw', 'ladle', 'turner')
        rm_keys = [k for k in inputs.keys() if k.startswith(ex_list)]
        for k in rm_keys:
            inputs.pop(k, None)
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
                       k_qry=args.k_qry,
                       dim_out=dim_output,
                       grasp=None)

    db_test.output_scale = db_train.output_scale
    if CLUSTER:
        device = torch.device('cuda:0')
        save_path = home + '/data/models/model_batchsz' + str(args.k_spt) + '_lr' + str(args.update_lr) + '_mr' + str(args.meta_lr) + '_lambda' + str(args.lmb) + '_exclude' + str(args.exclude) + '_epoch'
    else:
        device = torch.device('cuda')
        save_path = os.getcwd() + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(args.exclude) + '_epoch'
    if True:
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
            ('linear', [2,196,True])
        ]
        maml = Meta(args, config, None).to(device)
        maml.loss_fn = maml.avg_loss
    elif args.polar == 1 and args.meta == 1:
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
    elif args.polar == 0 and args.meta == 1:
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
        maml = Meta(args, config, dim_output, F.mse_loss, None).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    results_txt,test_losses,ft_training,pt_training = [],[],[],[]
    k_spt = args.k_spt * sample_size
    max_grad = 0
    for epoch in range(1): #args.epoch):
        batch_x, n_spt, batch_y,_,_ = db_train.next()
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
            losses.append(test_loss)
            training.append(train_loss)
            max_grad = max(max_grad, grad)

        #if epoch == 10000:  # evaluation
        if epoch % 1000 == 0:  # evaluation
            test_losses,ft_training,pt_training = [],[],[]
            for _ in range(10):
                batch_x,n_spt,batch_y,pos_keys,neg_keys = db_test.next()
                x_spt = batch_x[:,:k_spt,:]
                y_spt = batch_y[:,:k_spt,:]
                x_qry = batch_x[:,k_spt:,:]
                y_qry = batch_y[:,k_spt:,:]
                x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                             torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)

                t = 0
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one, n_spt_one, pos_one, neg_ones in zip(x_spt,y_spt,x_qry,y_qry,n_spt,pos_keys,neg_keys):
                    if args.meta == 1: 
                        losses, cam, loss_report, pred = maml.tune(n_spt_one, x_spt_one,y_spt_one,x_qry_one,y_qry_one) 
                        test_losses.append(loss_report[0]) 
                        pt_training.append(loss_report[1]) 
                        ft_training.append(loss_report[2])
                    else:
                        loss,w,_ = maml.finetuning(x_spt_one, y_spt_one,x_qry_one,y_qry_one)
                        test_losses.append(loss)
                    t+=1

            print('Test Loss:', np.array(test_losses).mean(axis=0))
            print('Ft Loss:', np.array(ft_training).mean(axis=0))
            print('Pt Loss:', np.array(pt_training).mean(axis=0))
            #results_txt.append("%0.6f" % (np.array(test_losses).mean(axis=0)))
            results_txt.append("%0.6f" % (np.array(test_losses).mean(axis=0)[-1]))
            test_losses,ft_training,pt_training = [],[],[]
    results_file = home + "/data/cross_val/meta" + str(args.meta) + "_ex" + str(args.exclude) + "_rv" + ".txt"
    out_file = open(results_file, "a+")
    out_file.write(("%0.3f" % args.update_lr) + ", " + ("%0.5f" % args.meta_lr) + ", " + str(results_txt) + '\n')
    out_file.close()
    


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--meta', type=int, help='epoch number', default=1)
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
    argparser.add_argument('--polar', type=int, help='epoch number', default=1)
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--lmb', type=float, help='task-level inner update learning rate', default=3.0)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
