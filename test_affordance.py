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

    dim_output = 6

    db_train = Affordances(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    load_path = os.getcwd() + '/data/tfs/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch0_al.pt'
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

    train = ["trowel_02_00000030", "scoop_01_00000129", "spoon_03_00000267"]
    test = ["scoop_02_00000030", "spoon_01_00000129", "scoop_02_00000267"]
    x_spt = db_train.input_only(train)
    y_spt = 3 * np.array([[-0.32,0.03,0.4,0,0,0], [0.08,-0.16,0.4,0,0,0], [-0.25,0.33,0.4,0,0,0]])
    #y_spt = 3 * np.array([[0.32,0.52,0.7,0.52,0.37,0.11], [0.54,0.41,0.93,0.18,0.59,0.23], [0.38,0.67,0.65,0.47,0.47,0.06]])
    #y_spt = np.array([[0.32,0.52,0.7,-0.52,-0.37,0.11], [0.54,0.41,0.93,0.18,-0.59,0.23], [0.38,0.67,0.65,-0.47,0.47,0.06]])
    x_qry = db_train.input_only(train)
    y_qry = np.zeros((3,6))
    x_spt, y_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device)
    x_qry, y_qry = torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

    acc,_,vals = maml.finetuning(x_spt, y_spt, x_qry, y_spt)

    print('Vals:', vals)
    print('Accs:', acc)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
