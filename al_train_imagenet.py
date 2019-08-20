import pdb
import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from torch.utils.tensorboard import SummaryWriter
from meta2 import Meta
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

    logger = SummaryWriter()

    print(args)

    config = [
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
        ('linear', [args.n_way + 1, 32 * 5 * 5])
    ]

    load_config = config[:-1]
    load_config.append(('linear', [args.n_way, 32 * 5 * 5]))

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    load_maml = Meta(args, load_config, al=False).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/home/tesca/data/miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('/home/tesca/data/miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.meta_lr) + '_epoch'
    load_path = os.getcwd() + '/data/model_batchsz1_stepsz0.01_epoch2000_v0.pt'
    if args.resume == 1:
        load_maml.load_state_dict(torch.load(load_path))
        load_maml.eval()
        params = list(maml.net.parameters())
        al_params = list(maml.al_net.parameters())
        load_params = list(load_maml.net.parameters())
        n1 = torch.cat([load_params[-2].data,torch.cuda.FloatTensor(1,load_params[-2].shape[-1]).fill_(0)])
        n2 = torch.cat([load_params[-1].data,torch.cuda.FloatTensor([0])])
        for p in range(len(params)-2):
            params[p].data = load_params[p].data
        params[-2] = n1
        params[-1] = n2
        maml.set_optim()

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs, al_accs, loss, al_loss, total_loss = maml(x_spt, y_spt, x_qry, y_qry)
            for tag, value in maml.net.named_parameters():
                tag = tag.replace('.','/')
                logger.add_histogram(tag, value.data.cpu().numpy(), step+1)
                if value.grad is not None:
                    logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
            logger.add_scalar('Loss/Model', loss, step+1)
            logger.add_scalar('Loss/AL', al_loss, step+1)
            logger.add_scalar('Loss/Total', total_loss, step+1)
            logger.add_scalar('Acc/Model', accs[-1], step+1)
            logger.add_scalar('Acc/AL', al_accs, step+1)


            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs, '\tAL acc:', al_accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                al_accs, accs_all_test = [], []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    al_accs.append(maml.al_test(x_qry, y_qry))
                    accs,_ = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
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
    argparser.add_argument('--resume', type=int, help='sets whether to use AL loss in updates', default=0)

    args = argparser.parse_args()

    main()
