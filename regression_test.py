import pdb
import random
import  torch, os, sys
import os.path
import  numpy as np
from sinusoid import Sinusoid
from polynomial import Polynomial
from imagenet import ImageNet
from cornell_grasps import CornellGrasps
import  argparse
from    torch.nn import functional as F

from meta import Meta

def main(args):

    sinusoid = {'name':'sinusoid','class':Sinusoid, 'dims':[1,1]}
    polynomial = {'name':'polynomial', 'class':Polynomial, 'dims':[2,1]}
    imagenet = {'name':'imagenet', 'class':ImageNet, 'dims':[4096,2]}
    grasps = {'name':'grasps', 'class':CornellGrasps, 'dims':[4096,2]}
    data_params = {'sinusoid':sinusoid, 'polynomial':polynomial, 'imagenet':imagenet, 'grasps':grasps}
    func_data = data_params[args.func_type]

    last_epoch = 1000
    save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + '.pt'
    while os.path.isfile(save_path):
        valid_epoch = last_epoch
        last_epoch += 1000
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + '.pt'
    save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch' + str(valid_epoch) + '.pt'

    torch.cuda.synchronize()
    torch.manual_seed(222)
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(222)
    torch.cuda.synchronize()
    np.random.seed(222)

    print("loading epoch " + str(valid_epoch))
    print(args)

    #device = torch.device('cpu')
    device = torch.device('cuda')
    torch.cuda.synchronize()

    dim_hidden = [40,40]
    dim_input, dim_output = func_data['dims']

    config = [
        ('fc', [dim_hidden[0], dim_input]),
        ('relu', [True])]#,
        #('bn', [dim_hidden[0]])]

    for i in range(1, len(dim_hidden)):
        config += [
            ('fc', [dim_hidden[i], dim_hidden[i-1]]),
            ('relu', [True])] #,
            #('bn', [dim_hidden[i]])]

    config += [
        ('fc', [dim_output, dim_hidden[-1]])] #,
        #('relu', [True])] #,
        #('bn', [dim_output])]

    mod = Meta(args, config, func_data['dims']).to(device)
    mod.load_state_dict(torch.load(save_path))
    mod.eval()

    if args.merge_spt_qry == 1:
        qry_idx = 0
        args.k_qry = 0
    else:
        qry_idx = args.k_spt

    cand_ests = None
    tgt_ests = None
    dists = None

    db_train = func_data['class'](
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry)

    all_accs = []
    batch_x, batch_y = db_train.next()

    for i in range(args.task_num):
        if i % 10 == 0:
            sys.stdout.write("\rTask %i" % i)
            sys.stdout.flush()
        queries = None
        accs = []
        c_idx = np.array(range(args.k_spt))
        q_set = torch.from_numpy(batch_x[i, qry_idx:, :]).float().to(device), torch.from_numpy(batch_y[i, qry_idx:, :]).float().to(device) 
        c_data = torch.from_numpy(batch_x[i, :args.k_spt, :]).float().to(device) 
        t_data = q_set[0]
        s_idx = None
            
        while len(c_idx) > 0:
            if args.iter_qry == 1:
                if args.al_method == "random":
                    s_idx = al_method_random(c_idx)
                if args.al_method == "k_centers":
                    s_idx = al_method_k_centers(mod, dists, cand_ests, tgt_ests, c_data, c_idx, t_data, s_idx)
                inputa = [batch_x[i, s_idx, :]]
                labela = [batch_y[i, s_idx, :]]
                if queries is None:
                    qin = np.array(inputa) #np.concatenate([inputa, inputa]) #, axis=1)     
                    ql = np.array(labela) #np.concatenate([labela, labela]) #, axis=1)     
                    s_idxs = [s_idx]
                    queries = [inputa, labela, s_idxs]
                else:
                    qin = np.concatenate([queries[0], inputa]) #, axis=1)     
                    ql = np.concatenate([queries[1], labela]) #, axis=1)     
                    s_idxs.append(s_idx)
                    queries = [qin, ql, s_idxs]
                c_idx = c_idx[c_idx != s_idx]

            else:
                qin = batch_x[i, :args.k_spt, :]
                ql = batch_y[i, :args.k_spt, :]
                c_idx = []
            qin, ql = torch.from_numpy(qin).float().to(device), torch.from_numpy(ql).float().to(device)
            test_acc = mod.finetunning(qin, ql, q_set[0], q_set[1])
            if len(accs) == 0:
                accs.append(test_acc[0])
            accs.append( test_acc[-1] )
            if np.isnan(test_acc[-1].cpu().detach().numpy()):
                pdb.set_trace()
        if args.iter_qry == 1:
            all_accs.append(accs)
        else:
            all_accs.append(test_acc)
    accs_val = np.array(all_accs).mean(axis=0).astype(np.float16)
    print('\nLoss:', accs_val)

def al_method_random(avail_cand_idx):
    return avail_cand_idx[random.randint(0, len(avail_cand_idx) - 1)]

def al_method_k_centers(mod, dists, cand_ests, tgt_ests, cand_data, avail_cand_idx, target_data, labeled_query):
    if dists is None:
        cand_ests = mod.net(cand_data, vars = mod.net.parameters())
        tgt_ests = mod.net(target_data, vars = mod.net.parameters())
    else:
        x = labeled_query[0]
        y = labeled_query[1]
        cand_ests[x] = y
        if args.merge_spt_qry == 1:
            tgt_ests[x] = y

    dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    for i in range(cand_data.shape[0]):
        for j in range(target_data.shape[0]):
            dists[i, j] = F.mse_loss(cand_ests[i], tgt_ests[j])

    labeled_cands = np.setdiff1d(range(cand_data.shape[0]), avail_cand_idx)
    max_min_dists = []
    for x in avail_cand_idx:
        min_dists = []
        for pt in range(target_data.shape[0]):
            min_dist = np.inf
            for exp_cands in np.concatenate([labeled_cands, [x]]):
                min_dist = min(min_dist, dists[exp_cands, pt])
            min_dists.append(min_dist)
        max_min_dists.append([x, max(min_dists)])
    idx = np.argmin(np.array(max_min_dists)[:,1])
    return max_min_dists[idx][0]
            

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--k_model', type=int, help='k shot for loading model params', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=100)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--iter_qry', type=int, help='learn from examples one-at-a-time', default=1)
    argparser.add_argument('--merge_spt_qry', type=int, help='select queries during AL from the test set', default=0)
    argparser.add_argument('--al_method', type=str, help='AL algorithm', default="none")
    argparser.add_argument('--func_type', type=str, help='function type', default="sinusoid")
    argparser.add_argument('--svm_lr', type=float, help='task-level inner update learning rate', default=0.001)

    args = argparser.parse_args()

    main(args)
