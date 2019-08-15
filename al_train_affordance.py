import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta5 import Meta
#from al_meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('linear', [512,4096]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [4, 512])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config, F.mse_loss, F.mse_loss).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = Affordances(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       train = True,
                       new_aff = args.new_aff,
                       exclude = args.exclude)

    db_test = Affordances(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       train = False,
                       new_aff = args.new_aff,
                       exclude = args.exclude)

    save_path = os.getcwd() + '/data/affordances/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch'

    for epoch in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        accs, al_accs = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 30 == 0:
            print('step:', step, '\ttraining acc:', accs, '\tAL acc:', al_accs)

        if step % 500 == 0:  # evaluation
            al_accs, accs_all_test = [], []

            x_spt, y_spt, x_qry, y_qry = db_test.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                accs,_ = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                al_accs.append(maml.al_test(x_qry_one, y_qry_one))
                accs_all_test.append(accs)

            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            _, al_accs = np.array(al_accs).mean(axis=0).astype(np.float16)
            print('AL acc:', al_accs)
            torch.save(maml.state_dict(), save_path + str(step) + "_al.pt")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--alpha', type=float, help='task-level inner update learning rate', default=1.0)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--train_al', type=int, help='sets whether to use AL loss in updates', default=0)

    args = argparser.parse_args()

    main()
