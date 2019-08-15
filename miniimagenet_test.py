import pdb
from copy import deepcopy
import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from active import QuerySelector
from meta2 import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    #print(args)

    if args.train_al == 1:
        last_layer = ('linear', [args.n_way+1, 32 * 5 * 5])
    else:
        last_layer = ('linear', [args.n_way, 32 * 5 * 5])

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
        last_layer
    ]

    last_epoch = 0
    if args.train_al == 1:
        suffix = "_al"
        #suffix = "-trained_al"
    else:
        suffix = "_og"
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + suffix + '.pt'
    while os.path.isfile(save_path):
        valid_epoch = last_epoch
        last_epoch += 500
        save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + suffix + '.pt'
    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(valid_epoch) + suffix+ '.pt'

    device = torch.device('cuda')
    mod = Meta(args, config).to(device)
    mod.load_state_dict(torch.load(save_path))
    mod.eval()

    #tmp = filter(lambda x: x.requires_grad, maml.parameters())
    #num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    #print('Total trainable tensors:', num)

    # batchsz here means total episode number
    if args.merge_spt_qry == 1:
        mini_test = MiniImagenet('/home/tesca/data/miniimagenet/', mode='test', n_way=args.n_way, k_shot=1,
                             k_query=args.k_qry,
                             batchsz=10, resize=args.imgsz)
    else:
        mini_test = MiniImagenet('/home/tesca/data/miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=10, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    accs_all_test = []
    total_accs = []
    best_accs = []

    it = 0
    for x_spt, y_spt, x_qry, y_qry in db_test:
        if args.merge_spt_qry == 1:
            x_spt = x_qry
            y_spt = y_qry
        sys.stdout.write("\rTest %i" % it)
        sys.stdout.flush()
        it += 1
        x_spt, y_spt = x_spt.squeeze(0), y_spt.squeeze(0)
        x_qry_pt, y_qry_pt = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        x_spt_pt, y_spt_pt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)
        #x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
        #                             x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        qs = QuerySelector()
        finished = False
        all_queries, all_orderings = [], []
        while not finished:
            maml = deepcopy(mod)
            queries = None #[[], []]
            avail_cands = list(range(x_spt.shape[0]))
            test_accs = []
            q_idx = []
            w = None
            for i in range(x_spt.shape[0]):
                q = qs.query(args, maml.net, w=w, cands=x_spt_pt, avail_cands=avail_cands, targets=x_qry_pt, k=1, method=args.al_method, classification=True)
                q_idx.append(q)
                avail_cands.remove(q)
                if queries is None:
                    queries = [np.array(x_spt[q].unsqueeze(0)), np.array(y_spt[q].unsqueeze(0))]
                else:
                    queries = [np.concatenate([queries[0], np.array(x_spt[q].unsqueeze(0))]), np.concatenate([queries[1], np.array(y_spt[q].unsqueeze(0))])]
                xs, ys = torch.from_numpy(queries[0]).to(device), torch.from_numpy(queries[1]).to(device)
                #accs,w = maml.finetunning(x_spt_pt, y_spt_pt, x_qry_pt, y_qry_pt)
                accs,w = maml.finetunning(xs, ys, x_qry_pt, y_qry_pt)
                accs_all_test.append(accs)
                if len(test_accs) == 0:
                    test_accs.append(accs[0])
                test_accs.append(accs[-1])
            all_orderings.append(test_accs)
            all_queries.append(q_idx)
            del maml
            finished = not qs.next_order()
        total_accs.append(np.mean(np.array(all_orderings),axis=0))
        best_accs.append(all_queries[np.argmax(np.sum(np.array(all_orderings),axis=1))])
        oq = [all_queries[i] for i in np.argsort(np.sum(np.array(all_orderings),axis=1))]
        #pdb.set_trace()
        oq = all_orderings[np.argsort(np.sum(np.array(all_orderings),axis=1))[-1]]
        #total_accs.append(oq)

    # [b, update_step+1]
    accs = np.array(total_accs).mean(axis=0).astype(np.float16)
    #accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_model', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--al_method', type=str, help='active learning method', default='random')
    argparser.add_argument('--dropout_rate', type=float, help='task-level inner update learning rate', default=-1)
    argparser.add_argument('--train_al', type=int, help='sets whether to use AL loss in updates', default=0)
    argparser.add_argument('--merge_spt_qry', type=int, help='sets whether to use AL loss in updates', default=0)

    args = argparser.parse_args()

    main()
