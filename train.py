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
from meta import Meta
from aff_data import Affordances
from os.path import expanduser
from torch import nn

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

    CLUSTER = True

    home = expanduser("~")
    if CLUSTER:
        fts_loc = home + "/data/fts.pbz2"
        print("Loading input files...")
        with bz2.open(fts_loc, 'rb') as handle:
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
                       dim_out=dim_output)

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
                       dim_out=dim_output)

    db_test.output_scale = db_train.output_scale
    if CLUSTER:
        device = torch.device('cuda:0')
        save_path = home + '/data/models/model_tasksz' + str(args.task_num) + '_batchsz' + str(args.k_spt) + '_lr' + str(args.update_lr) + '_mr' + str(args.meta_lr) + '_lambda' + str(args.lmb) + '_exclude' + str(args.exclude) + '_epoch'
    else:
        device = torch.device('cuda')
        save_path = os.getcwd() + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(args.exclude) + '_epoch'
    print(str(db_train.dim_input) + "-D input")
    dim = db_train.dim_input
    if True:
        config = [
            ('linear', [dim,dim,True]),
            ('relu', [True]),
            ('linear', [1,dim,True]),
            ('relu', [True]),
            ('reshape',[196]),
            ('linear', [196,196,True]),
        ]
        maml = Meta(args, config, dim_output, None, None).to(device)
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

    losses,training = [],[]
    k_spt = args.k_spt * sample_size
    max_grad = 0
    for epoch in range(args.epoch):
        batch_x, n_spt, batch_y,_,dist = db_train.next()
        x_spt = batch_x[:,:k_spt,:]
        y_spt = batch_y[:,:k_spt,:]
        x_qry = batch_x[:,k_spt:,:]
        y_qry = batch_y[:,k_spt:,:]
        x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)

        if args.meta == 1:
            acc, loss, train_acc, grad = maml.class_test(n_spt, x_spt, y_spt, x_qry, y_qry,debug=epoch>25)
            #acc, loss, train_acc, grad = maml.class_forward(n_spt, x_spt, y_spt, x_qry, y_qry,debug=epoch>25)
            losses.append(acc)
            training.append(train_acc)
            max_grad = max(max_grad, grad)
        else:
            train_loss,test_loss,_,grad = maml(x_spt, y_spt,x_qry,y_qry)
            losses.append(test_loss)
            training.append(train_loss)
            max_grad = max(max_grad, grad)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttesting  loss:', np.array(losses).mean(axis=0))
            print('step:', epoch, '\ttraining loss:', np.array(training).mean(axis=0))
            print('max grad: ' + str(max_grad))
            losses,training = [],[]
            max_grad = 0

        if epoch % 1000 == 0:  # evaluation
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_meta" + str(args.meta) + "_polar" + str(args.polar) + "-rv.pt")
            test_losses = []
            batch_x,n_spt,batch_y,names,dist = db_test.next()
            x_spt = batch_x[:,:k_spt,:]
            y_spt = batch_y[:,:k_spt,:]
            x_qry = batch_x[:,k_spt:,:]
            y_qry = batch_y[:,k_spt:,:]
            x_spt, y_spt, x_qry, y_qry, n_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                         torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device), torch.from_numpy(n_spt).float().to(device)

            t = 0
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one, n_spt_one, names_one in zip(x_spt,y_spt,x_qry,y_qry,n_spt,names):
                n_spt = names_one[:k_spt]
                n_qry = names_one[k_spt:]
                if args.meta == 1:
                    _,w,loss,_ = maml.class_tune4(n_spt_one, x_spt_one,y_spt_one,x_qry_one,y_qry_one)
                    test_losses.append(loss)
                else:
                    train_loss,test_loss,w,_ = maml.finetuning(x_spt_one, y_spt_one,x_qry_one,y_qry_one)
                    test_losses.append(test_loss)
                #if not CLUSTER:
                if epoch % 10000 == 0: 
                    for i in range(x_spt_one.shape[0]):
                        name = n_spt[i]
                        cam1 = get_CAM(w[0][i])
                        if len(w) > 1:
                            cam2 = get_CAM(w[1][i])
                        else:
                            cam2 = None
                        pref = name.split("_00")[0]
                        pos = db_train.output_scale.inverse_transform(y_spt_one[i].cpu().numpy().reshape(1,-1)).squeeze()
                        if CLUSTER:
                            img = cv.imread('/u/tesca/data/center_tools/' + name + '_center.jpg')
                        else:
                            img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + name + '_center.jpg')
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
                            if CLUSTER:
                                cv.imwrite('/u/tesca/data/cam/ex' + str(args.exclude) + '/' + name + '_t' + str(t) + '-rv.jpg', result1)
                            else:
                                cv.imwrite('data/cam/polar' + str(args.polar) + 'meta' + str(args.meta) +'/ex' + str(args.exclude) + '/' + name + '_t' + str(t) + '.jpg', result1)
                            if polar and (cam2 is not None):
                                height, width, _ = tf_img.shape
                                heatmap2 = cv.applyColorMap(cv.resize(cam2,(width, height)), cv.COLORMAP_JET)
                                result2 = heatmap2 * 0.3 + tf_img * 0.5
                                cv.circle(result2,(int(tf_pos[1]),int(tf_pos[0])),5,[255,255,0])
                                cv.imwrite('data/cam/' + name + 'ex' + str(args.exclude) + '_CAM_polar.jpg', result2)
                t+=1

            print('Test Loss:', np.array(test_losses).mean(axis=0))
            pdb.set_trace()
            


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--meta', type=int, help='epoch number', default=1)
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
    argparser.add_argument('--polar', type=int, help='epoch number', default=0)
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=30)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--lmb', type=float, help='task-level inner update learning rate', default=10.0)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
