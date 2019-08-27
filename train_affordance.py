import pdb
from torch.utils.tensorboard import SummaryWriter
import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import torch.nn.functional as F
from basic_meta import Meta
#from al_meta import Meta
from affordances import Affordances

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=3)
    logger = SummaryWriter()
    print(args)
    al_sz = 0
    db_train = Affordances(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       train = True,
                       new_aff = args.new_aff,
                       exclude = args.exclude)

    db_test = Affordances(
                       batchsz=1,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       train = False,
                       new_aff = args.new_aff,
                       exclude = args.exclude)

    save_path = os.getcwd() + '/data/affordances/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_naff' + str(args.new_aff) + '_ex' + str(args.exclude) + '_epoch'
    print(db_train.dim_input)
    config = [
        ('linear', [32,db_train.dim_input]),
        ('relu', [True]),
        ('bn', [32]),
        ('linear', [32,32]),
        ('relu', [True]),
        ('bn', [32]),
        ('linear', [2, 32])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config, al_sz, F.mse_loss, None).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    accs,al_accs = [],[]
    for epoch in range(args.epoch):
        batch_x, batch_y = db_train.next()
        x_spt = batch_x[:,:args.k_spt,:]
        y_spt = batch_y[:,:args.k_spt,:]
        x_qry = batch_x[:,args.k_spt:,:]
        y_qry = batch_y[:,args.k_spt:,:]
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

        acc, loss = maml(x_spt, y_spt, x_qry, y_qry,(epoch%500==0))
        accs.append(acc)
        logger.add_scalar('Accs/model',acc[-1],epoch+1)
        logger.add_scalar('Loss/model',loss,epoch+1)

        if epoch % 30 == 0:
            print('step:', epoch, '\ttraining acc:', np.array(accs).mean(axis=0))
            accs,al_accs = [],[]

        if epoch % 500 == 0:  # evaluation
            al_accs, accs_all_test = [], []

            batch_x, batch_y = db_train.next()
            x_spt = batch_x[:,:args.k_spt,:]
            y_spt = batch_y[:,:args.k_spt,:]
            x_qry = batch_x[:,args.k_spt:,:]
            y_qry = batch_y[:,args.k_spt:,:]
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                         torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)


            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                acc,_ = maml.finetuning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs_all_test.append(acc)

            # [b, update_step+1]
            acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Test acc:', acc)
            logger.add_scalar('Test/model',acc[-1],epoch+1)
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_al.pt")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--k_model', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--alpha', type=float, help='task-level inner update learning rate', default=1.0)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--train_al', type=int, help='sets whether to use AL loss in updates', default=0)
    argparser.add_argument('--new_aff', type=int, help='sets whether to use AL loss in updates', default=0)
    argparser.add_argument('--exclude', type=int, help='sets whether to use AL loss in updates', default=0)

    args = argparser.parse_args()

    main()
