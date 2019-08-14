import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta_sub import AL_Learner
from meta import Meta
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


    ## Task Learner Setup
    task_config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]
    last_epoch = 0
    suffix = "_v0"
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + suffix + '.pt'
    while os.path.isfile(save_path):
        valid_epoch = last_epoch
        last_epoch += 500
        save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + suffix + '.pt'
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(valid_epoch) + suffix+ '.pt'

    device = torch.device('cuda')
    task_mod = Meta(args, task_config).to(device)
    task_mod.load_state_dict(torch.load(save_path))
    task_mod.eval()

    ## AL Learner Setup
    print(args)

    al_config = [
        ('linear', [1, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = AL_Learner(args, al_config, task_mod).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/home/tesca/data/miniimagenet/', mode='train', n_way=args.n_way, k_shot=1,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('/home/tesca/data/miniimagenet/', mode='test', n_way=args.n_way, k_shot=1,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch'

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            al_accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('step:', step, '\tAL acc:', al_accs)

            if step % 500 == 0:  # evaluation
                torch.save(maml.state_dict(), save_path + str(step) + "_al_net.pt")
                '''db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                al_accs = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    al_accs.append(maml.al_test(x_qry, y_qry))

                # [b, update_step+1]
                al_accs = np.array(al_accs).mean(axis=0).astype(np.float16)
                print('AL acc:', al_accs)'''



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_model', type=int, help='k shot for support set', default=1)
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
