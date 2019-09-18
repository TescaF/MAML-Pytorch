import pdb
#from torch.utils.tensorboard import SummaryWriter
import  torch, os
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
import torch.nn.functional as F
from basic_meta import Meta
from reg_aff import Affordances

def mod_mse_loss(data, target):
    sc = torch.cuda.FloatTensor([1])
    a = data[:,1] - target[:,1]
    y_diff = torch.remainder((data[:,1] - target[:,1]) + sc, 2) - sc
    pdb.set_trace()

    return F.mse_loss(data, target)

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=5,suppress=True)
    #logger = SummaryWriter()
    print(args)

    dim_output = 3
    sample_size = args.task_num # number of images per object

    db_train = Affordances(
                       train=True,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    db_test = Affordances(
                       train=False,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    save_path = os.getcwd() + '/data/tfs/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_exclude' + str(args.exclude) + '_epoch'
    print(str(db_train.dim_input) + "-D input")
    dim = db_train.dim_input
    config = [
        #('linear', [db_train.num_classes,db_train.dim_input])]
        ('linear', [dim, dim]),
        ('relu', [True]),
        ('avg_pool2d', [7,7,0]),
        ('flatten', []),
        ('linear', [3,49])
    ]

    device = torch.device('cpu')
    #device = torch.device('cuda')
    maml = Meta(args, config, F.mse_loss, None).to(device)
    #maml = Meta(args, config, None, torch.eq).to(device)
    #maml.loss_fn = maml.fisher_loss
    #maml.loss_fn = maml.cross_entropy_loss
    #maml = Meta(args, config, "mod_mse_loss", None).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    losses = []
    k_spt = args.k_spt * sample_size
    for epoch in range(args.epoch):
        x_spt, y_spt = db_train.next()
        x_spt, y_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).long().to(device)

        loss, acc = maml.forward_batch(x_spt, y_spt, epoch%50==0)
        losses.append(acc)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttraining loss:', np.array(losses).mean(axis=0))
            losses = []

        if epoch % 100 == 0:  # evaluation
            '''test_losses = []

            batch_x = db_test.next()
            x_spt = batch_x[:,:k_spt,:]
            x_qry = batch_x[:,k_spt:,:]
            x_spt, x_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(x_qry).float().to(device)

            for x_spt_one, x_qry_one in zip(x_spt, x_qry):
                loss = maml.test(x_spt_one)
                test_losses.append(loss)

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
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=0)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=20)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.01)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
