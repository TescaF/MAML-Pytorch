import pdb
from torch.utils.tensorboard import SummaryWriter
import  torch, os
import  numpy as np
import  scipy.stats
import  random, sys, pickle
import  argparse
import torch.nn.functional as F
from basic_meta import Meta
from affordances import Affordances

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=3)
    logger = SummaryWriter()
    print(args)

    dim_output = 3

    db_train = Affordances(
                       train=True,
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    save_path = os.getcwd() + '/data/tfs/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch'
    print(str(db_train.dim_input) + "-D input")
    config = [
        ('linear', [512,db_train.dim_input]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [128,512]),
        ('relu', [True]),
        ('bn', [128]),
        ('linear', [dim_output, 128])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config, F.mse_loss, None).to(device)

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

        acc, loss = maml(x_spt, y_spt, x_qry, y_qry)
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
                acc,_,_ = maml.finetuning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs_all_test.append(acc)

            # [b, update_step+1]
            acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Test acc:', acc)
            logger.add_scalar('Test/model',acc[-1],epoch+1)
            torch.save(maml.state_dict(), save_path + str(epoch%2000) + "_al.pt")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20001)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
