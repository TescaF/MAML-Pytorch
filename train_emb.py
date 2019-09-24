import math
import pdb
#from torch.utils.tensorboard import SummaryWriter
import  torch, os
import cv2 as cv
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
import torch.nn.functional as F
from reg_meta import Meta
from reg_aff import Affordances

def get_CAM(cam):
    '''img = np.zeros((640,480))
    dims = [480,640]
    tf_range = tf.transform(np.array([dims])) - tf.transform(np.array([[0,0]]))
    for x in range(640):
        for y in range(480):
            src = np.round((w/tf_range.squeeze()) * tf.transform(np.array([[y,x]])).squeeze()).astype(int)
            img[x,y] = cam[src[0],src[1]].item()'''
    cam = cam.reshape(14,14)
    cam = torch.abs(cam)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    cam_img = np.uint8(255 * cam_img.cpu().detach().numpy())
    return cam_img

def img_polar_tf(img):
    dim = img.shape[0]
    d = int(dim/2)
    urange = np.linspace(0, 0.5 * np.sqrt(2*(dim**2)), dim)
    #urange = np.linspace(0, np.log(dim), dim)
    vrange = np.linspace(0, 2*np.pi, dim)
    vs, us = np.meshgrid(vrange, urange)
    rs = us
    #rs = np.exp(us)
    xs = rs * np.cos(vs)
    ys = rs * np.sin(vs)
    polar_map = np.clip(np.floor(np.stack([xs,ys],2)), -int(dim/2), int(dim/2)-1)
    tf_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sx,sy = polar_map[i,j]
            tf_img[i,j] = img[int(sx)+d,int(sy)+d]
    return tf_img, polar_map

def cart_to_polar(pos, polar_map):
    dist = np.inf
    for i in range(polar_map.shape[0]):
        for j in range(polar_map.shape[1]):
            c = polar_map[i,j]
            d = ((pos[0]-c[0])**2) + ((pos[1]-c[1])**2)
            if d < dist:
                dist = d
                pt = [i,j]
    return pt

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    #logger = SummaryWriter()
    print(args)

    mode = "center"
    dim_output = 2
    sample_size = args.sample_size # number of images per object

    db_train = Affordances(
                       mode=mode,
                       train=True,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    '''db_test = Affordances(
                       train=False,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=5,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)'''

    save_path = os.getcwd() + '/data/models/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(args.exclude) + '_epoch'
    print(str(db_train.dim_input) + "-D input")
    dim = db_train.dim_input
    config = [
        ('linear', [1,dim,True]),
        ('leakyrelu', [0.01,True]),
        ('reshape',[14,14]),
        ('polar',[14,14]),
        ('reshape',[196]),
        ('linear', [196,196,True]),
        ('relu', [True]),
        ('linear', [dim_output,196,True])
    ]


    #device = torch.device('cpu')
    device = torch.device('cuda')
    if mode == "polar":
        maml = Meta(args, config, None, None).to(device)
        maml.loss_fn = maml.wrap_mse_loss
    elif mode == "center":
        maml = Meta(args, config, None, None).to(device)
        maml.loss_fn = maml.polar_loss
    else:
        maml = Meta(args, config, F.mse_loss, None).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    losses,training = [],[]
    k_spt = args.k_spt * sample_size
    max_grad = 0
    for epoch in range(args.epoch):
        batch_x, batch_y,_ = db_train.next()
        x_spt = batch_x[:,:k_spt,:]
        y_spt = batch_y[:,:k_spt,:]
        x_qry = batch_x[:,k_spt:,:]
        y_qry = batch_y[:,k_spt:,:]
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

        acc, loss, train_acc, grad = maml(x_spt, y_spt, x_qry, y_qry)
        losses.append(acc)
        training.append(train_acc)
        max_grad = max(max_grad, grad)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttesting  loss:', np.array(losses).mean(axis=0))
            print('step:', epoch, '\ttraining loss:', np.array(training).mean(axis=0))
            print('max grad: ' + str(max_grad))
            losses,training = [],[]
            max_grad = 0

        if epoch % 500 == 0:  # evaluation
            
            batch_x,batch_y,names = db_train.next()
            x_spt = batch_x[:,:k_spt,:]
            y_spt = batch_y[:,:k_spt,:]
            x_qry = batch_x[:,k_spt:,:]
            y_qry = batch_y[:,k_spt:,:]
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                         torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

            for x_spt_one, y_spt_one, x_qry_one, y_qry_one, names_one in zip(x_spt,y_spt,x_qry,y_qry,names):
                n_spt = names_one[:k_spt]
                n_qry = names_one[k_spt:]
                loss,w,res = maml.finetuning(x_spt_one,y_spt_one,x_qry_one,y_qry_one)
                for i in range(x_spt_one.shape[0]):
                    name = n_spt[i]
                    cam1 = get_CAM(w[0][i])
                    cam2 = get_CAM(w[1][i])
                    pref = name.split("_00")[0]
                    pos = db_train.output_scale.inverse_transform(y_spt_one[i].cpu().numpy().reshape(1,-1)).squeeze()
                    if mode == "polar":
                        try:
                            img = cv.imread('/home/tesca/data/part-affordance-dataset/polar_tools/' + name + '_polar.jpg')
                        except:
                            img = None
                    elif mode == "center":
                        img = cv.imread('/home/tesca/data/part-affordance-dataset/center_tools/' + name + '_center.jpg')
                        # Scale target point to position in 224x224 img
                        mult = [(pos[0] * 224/img.shape[0])-112, (pos[1] * 224/img.shape[1])-112]
                        r = 224*np.sqrt(mult[0]**2 + mult[1]**2)/(0.5*np.sqrt(2*(224**2)))
                        a = (224/(2*np.pi)) * math.atan2(mult[1],mult[0]) % 224
                        tf_pos = [r,a]
                        tf_img,polar_map = img_polar_tf(cv.resize(img, (224,224)))
                        '''inv_x = r*(0.5*np.sqrt(2*(224**2))/224) * math.cos(2*np.pi*a/224)
                        inv_y = r*(0.5*np.sqrt(2*(224**2))/224) * math.sin(2*np.pi*a/224)
                        #inv_y = np.exp(r*np.log(224)/224) * math.sin(2*np.pi*a/224)
                        pdb.set_trace()
                        print(tf_pos)
                        cv.circle(tf_img,(int(tf_pos[1]),int(tf_pos[0])),5,[255,255,0])
                        cv.imshow("im", tf_img)
                        cv.waitKey(0)'''
                    else:
                        img = cv.imread('/home/tesca/data/part-affordance-dataset/tools/' + pref + '/' + name + '_rgb.jpg')
                    if img is not None:
                        height, width, _ = img.shape
                        heatmap1 = cv.applyColorMap(cv.resize(cam1,(width, height)), cv.COLORMAP_JET)
                        result1 = heatmap1 * 0.3 + img * 0.5
                        cv.circle(result1,(int(pos[1]),int(pos[0])),5,[255,255,0])
                        cv.imwrite('data/cam/' + name + '_CAM_1.jpg', result1)
                        height, width, _ = tf_img.shape
                        heatmap2 = cv.applyColorMap(cv.resize(cam2,(width, height)), cv.COLORMAP_JET)
                        result2 = heatmap2 * 0.3 + tf_img * 0.5
                        cv.circle(result2,(int(tf_pos[1]),int(tf_pos[0])),5,[255,255,0])
                        cv.imwrite('data/cam/' + name + '_CAM_2.jpg', result2)

            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_al.pt")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
    argparser.add_argument('--sample_size', type=int, help='epoch number', default=10)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
