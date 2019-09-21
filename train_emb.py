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


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    #logger = SummaryWriter()
    print(args)

    polar = False
    dim_output = 2
    sample_size = args.sample_size # number of images per object

    db_train = Affordances(
                       polar=polar,
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
        ('linear', [512,dim,True]),
        ('relu', [True]),
        ('linear', [1,512,True]),
        ('relu', [True]),
        #('leakyrelu', [0.01,True]),
        ('reshape',[196]),
        ('linear', [196,196,True]),
        ('relu', [True]),
        ('linear', [dim_output,196,True])
    ]


    #device = torch.device('cpu')
    device = torch.device('cuda')
    maml = Meta(args, config, F.mse_loss, None).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    losses,training = [],[]
    k_spt = args.k_spt * sample_size
    for epoch in range(args.epoch):
        batch_x, batch_y,_ = db_train.next()
        x_spt = batch_x[:,:k_spt,:]
        y_spt = batch_y[:,:k_spt,:]
        x_qry = batch_x[:,k_spt:,:]
        y_qry = batch_y[:,k_spt:,:]
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

        acc, loss, train_acc = maml(x_spt, y_spt, x_qry, y_qry)
        losses.append(acc)
        training.append(train_acc)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttesting  loss:', np.array(losses).mean(axis=0))
            print('step:', epoch, '\ttraining loss:', np.array(training).mean(axis=0))
            losses,training = [],[]

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
                    cam = get_CAM(w[i])
                    pref = name.split("_00")[0]
                    if polar:
                        try:
                            img = cv.imread('/home/tesca/data/part-affordance-dataset/polar_tools/' + name + '_polar.jpg')
                        except:
                            img = None
                    else:
                        img = cv.imread('/home/tesca/data/part-affordance-dataset/tools/' + pref + '/' + name + '_rgb.jpg')
                    if img is not None:
                        height, width, _ = img.shape
                        heatmap = cv.applyColorMap(cv.resize(cam,(width, height)), cv.COLORMAP_JET)
                        result = heatmap * 0.3 + img * 0.5
                        #pos = db_train.output_scale.inverse_transform(y_spt_one[i].cpu().numpy().reshape(1,-1)).squeeze()
                        #cv.circle(result,(pos[1],pos[0]),5,[255,255,0])
                        cv.imwrite('data/cam/' + name + '_CAM.jpg', result)

            
            '''test_losses = []


            # [b, update_step+1]
            loss = np.array(test_losses).mean()
            print('Test loss:', loss)'''
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_al.pt")

        '''if epoch % 20000 == 0:  # evaluation
            cam = torch.mm(maml.net.parameters()[0][class_num].unsqueeze(0), embedding.reshape((2048,49)))
            outs = cam.reshape((7,7))
            outs = outs - torch.min(outs)
            outs = outs / torch.max(outs)
            outs = np.uint8(outs.cpu().detach().numpy() * 255)
            layer = cv.resize(outs, (640,480))
            heatmap = cv.applyColorMap(layer, cv.COLORMAP_JET)
            cv.addWeighted(heatmap, 0.3, polar_img, 0.7, 0, heatmap)
            cv.imshow("im", heatmap)
            cv.waitKey()
            cv.destroyAllWindows()'''

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
