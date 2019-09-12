import math
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
from sklearn import preprocessing
def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    np.set_printoptions(precision=3)
    logger = SummaryWriter()
    print(args)

    dim_output = 3
    sample_size = 5

    db_test = Affordances(
                       train=False,
                       batchsz=args.task_num,
                       exclude=args.exclude,
                       samples=sample_size,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       dim_out=dim_output)

    load_path = os.getcwd() + '/data/tfs/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_meta' + str(args.meta_lr) + '_exclude' + str(args.exclude) + '_epoch0_al.pt'
    print(load_path)
    print(str(db_test.dim_input) + "-D input")
    config = [
        ('linear', [4096,db_test.dim_input]),
        ('relu', [True]),
        ('bn', [4096]),
        ('linear', [512,4096]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [dim_output, 512]),
        ('sigmoid', [None])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config, None, None).to(device)
    maml.load_state_dict(torch.load(load_path))
    maml.eval()
    maml.loss_fn = maml.shift_loss

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    k_spt = args.k_spt * sample_size
    test_from_dataset(device, maml, db_test, k_spt)
    return 
    ## Configure training samples
    train = ["spoon_01_00000003", "trowel_01_00000003", "mug_01_00000003"]
    tfs = [[-122, -20],[212,15],[96,34]]
    y_spt = []
    for tf in tfs:
        x = (math.sqrt(tf[0]**2 + tf[1]**2)-200)/200
        y = math.atan2(tf[1],tf[0])/math.pi
        y_spt.append([y,x])
    #train = ["scissors_01_00000027", "shears_02_00000003", "scissors_08_00000003"]
    x_spt = db_test.input_only(train)
    y_spt = np.array(y_spt)
    pdb.set_trace()
    #y_spt = np.array([[0.01,-0.011,0.01,0.3,-0.6,-0.01], [-0.012,-0.015,0.02,-0.45,-0.6,-0.01], [0.001,-0.012,0.001,0,-1,0]])
    #sc = preprocessing.MinMaxScaler(feature_range=(-1,1))
    #y_spt = sc.fit_transform(y_spt[:,:3])

    ## Configure test samples
    test = ["scissors_01_00000027","scissors_01_00000060","scissors_01_00000003", "shears_01_00000009","shears_01_00000033","shears_01_00000090","scissors_02_00000009","scissors_02_00000060","scissors_02_00000030", "shears_01_00000099","shears_01_00000018","shears_01_00000021","scissors_01_00000081","scissors_01_00000096","scissors_01_00000033"]
    qry_keys, x_qry = db_test.next_input(0)
    x_qry = x_spt
    y_qry = y_spt
    #x_qry = x_qry[args.k_spt:]
    #y_qry = np.zeros((x_qry.shape[0],dim_output))
    ## Run through network
    x_spt, y_spt = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device)
    x_qry, y_qry = torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)
    acc,_,vals = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
    a = maml.tan_mse_loss(vals, y_spt)
    print(vals)
    for val in vals:
        ang = torch.atan2(torch.tanh(val[0]),torch.tanh(val[1])).item()/math.pi
        print([ang,val[2]])
    pdb.set_trace()
    #vals = sc.inverse_transform(vals.detach().cpu())
    #print(vals)

    for i in range(vals.shape[0]):
        val_str = ""
        for v in vals[i]:
            val_str += str(v) + ","
        print("['" + qry_keys[i] + "', [" + val_str[:-1] + "]], ")
    print('Accs:', acc)

def test_from_dataset(device, maml, db_test, k_spt):
    al_accs, accs_all_test = [], []

    batch_x, batch_y = db_test.next()
    x_spt = batch_x[:,:k_spt,:]
    y_spt = batch_y[:,:k_spt,:]
    x_qry = batch_x[:,k_spt:,:]
    y_qry = batch_y[:,k_spt:,:]
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                 torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)


    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
        acc,_,vals = maml.finetuning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
        pdb.set_trace()
        accs_all_test.append(acc)

    # [b, update_step+1]
    acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', acc)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exclude', type=int, help='epoch number', default=0)
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
