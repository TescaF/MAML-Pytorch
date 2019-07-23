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
from copy import deepcopy
import itertools
from meta import Meta
from categorized_grasps import CategorizedGrasps
from affordances import Affordances

def main(args):

    sinusoid = {'name':'sinusoid','class':Sinusoid, 'dims':[1,1,0]}
    polynomial = {'name':'polynomial', 'class':Polynomial, 'dims':[2,1,0]}
    imagenet = {'name':'imagenet', 'class':ImageNet, 'dims':[4096,2,0]}
    grasps = {'name':'grasps', 'class':CornellGrasps, 'dims':[4096,2,0]}
    cat_grasps = {'name':'cat_grasps', 'class':CategorizedGrasps, 'dims':[4096,2,1]} #third number is param length
    affordances = {'name':'affordances', 'class':Affordances, 'dims':[4096,3,0]} #third number is param length
    data_params = {'affordances':affordances, 'cat_grasps':cat_grasps}
    func_data = data_params[args.func_type]

    if args.leave_out > 0:
        split_txt = ''
        split_num = args.leave_out
        dir_name = "lo_" + str(args.leave_out) + "/"
    else:
        split_txt = 'split' + str(args.split)
        split_num = args.split
        dir_name = ""

    last_epoch = 1000
    if args.split_cat == 1:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-cat_epoch' + str(last_epoch) + '.pt'
    else:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-obj_epoch' + str(last_epoch) + '.pt'
    #save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + '.pt'
    while os.path.isfile(save_path):
        valid_epoch = last_epoch
        last_epoch += 1000
        #save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(last_epoch) + '.pt'
        if args.split_cat == 1:
            save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-cat_epoch' + str(last_epoch) + '.pt'
        else:
            save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-obj_epoch' + str(last_epoch) + '.pt'

    #save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + '_epoch' + str(valid_epoch) + '.pt'

    if args.split_cat == 1:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-cat_epoch' + str(valid_epoch) + '.pt'
    else:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_model) + '_stepsz' + str(args.update_lr) + split_txt + '-obj_epoch' + str(valid_epoch) + '.pt'

    #pdb.set_trace()
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

    #dim_hidden = [4096,500]
    if args.func_type == "cat_grasps":
        dim_hidden = [4096,[512,513], 128]
    if args.func_type == "affordances":
        dim_hidden = [4096,512, 128]

    #dim_hidden = [40,40]
    dims = func_data['dims']

    #dim_input, dim_output = func_data['dims']

    '''config = [
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
        #('bn', [dim_output])]'''

    config = [
        ('linear', [dim_hidden[0], dims[0]]),
        ('relu', [True]),
        ('bn', [dim_hidden[0]])]
    prev_dim = dim_hidden[0]
    for i in range(1, len(dim_hidden)):
        if type(dim_hidden[i]) == list:
            curr_dim = dim_hidden[i][0]
        else:
            curr_dim = dim_hidden[i]
        config += [
            ('linear', [curr_dim, prev_dim]),
            ('relu', [True]),
            ('bn', [curr_dim])]
        if type(dim_hidden[i]) == list:
            prev_dim = dim_hidden[i][1]
        else:
            prev_dim = curr_dim

    config += [
        ('linear', [dims[1], prev_dim])]


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
                       batchsz=-1, #args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry,
                       num_grasps=args.grasps,
                       split=split_num,
                       train=False,
                       split_cat=args.split_cat)



    all_accs = []
    for _ in range(db_train.batch_size):
        all_accs.append([])
    #for i in range(args.task_num):
    for l in range(args.task_num):
        batch_x, batch_y = db_train.next()
        sys.stdout.write("\rTask %i" % l)
        sys.stdout.flush()
        for i in range(batch_x.shape[0]):
            queries = None
            accs = []
            c_idx = np.array(range(args.k_spt))
            q_set = torch.from_numpy(batch_x[i, qry_idx:, :]).float().to(device), torch.from_numpy(batch_y[i, qry_idx:, :]).float().to(device) 
            c_data = torch.from_numpy(batch_x[i, :args.k_spt, :]).float().to(device) 
            t_data = q_set[0]
            s_idx = None
            tuned_w = None
            #curr_mod = deepcopy(mod.net)
            while len(c_idx) >= args.batch_sz: #% args.batch_sz == 0:
                if args.iter_qry == 1:
                    if args.al_method == "random":
                        s_idx = al_method_random(c_idx, args.batch_sz)
                    if args.al_method == "output_space":
                        s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, output_dist=True)
                    if args.al_method == "input_space":
                        s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, output_dist=False)
                    if args.al_method == "dropout":
                        s_idx = al_method_dropout(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz)
                        #del curr_mod
                        #torch.cuda.empty_cache()
                    inputa, labela = [], []
                    for b in s_idx:
                        inputa.append(batch_x[i,b,:])
                        labela.append(batch_y[i,b,:])
                    if queries is None:
                        #qin = np.array(inputa) #np.concatenate([inputa, inputa]) #, axis=1)     
                        #ql = np.array(labela) #np.concatenate([labela, labela]) #, axis=1)     
                        qin = np.concatenate([inputa, inputa]) #, axis=1)     
                        ql = np.concatenate([labela, labela]) #, axis=1)     
                        s_idxs = s_idx #[s_idx]
                        queries = [inputa, labela, s_idxs]
                    else:
                        qin = np.concatenate([queries[0], inputa]) #, axis=1)     
                        ql = np.concatenate([queries[1], labela]) #, axis=1)     
                        s_idxs += s_idx
                        #s_idxs.append(s_idx)
                        queries = [qin, ql, s_idxs]
                    c_new = []
                    for c in c_idx:
                        if c not in s_idx:
                            c_new.append(c)
                    c_idx = np.array(c_new)

                else:
                    qin = batch_x[i, :args.k_spt, :]
                    ql = batch_y[i, :args.k_spt, :]
                    c_idx = []
                qin, ql = torch.from_numpy(qin).float().to(device), torch.from_numpy(ql).float().to(device)
                test_acc, tuned_w = mod.finetuning(qin, ql, q_set[0], q_set[1], dims[2], args.tuned_layers,new_env=(args.split_cat==0))
                #print(str(len(queries[2])) + ": " + str(test_acc[-1]-test_acc[-5]))
                if len(accs) == 0:
                    accs.append(test_acc[0])
                accs.append( test_acc[-1] )
                if np.isnan(test_acc[-1]): #.cpu().detach().numpy()):
                    pdb.set_trace()
            if args.iter_qry == 1:
                #pdb.set_trace()
                all_accs[i].append(accs)
                #all_accs.append(np.array(accs))
            else:
                all_accs[i].append(test_acc)
    for j in range(len(all_accs)):
        accs_val = np.array(all_accs[j]).mean(axis=0).astype(np.float16)
        print('\nTask ' + str(j) + ' loss:', accs_val)

def al_method_random(avail_cand_idx, req_count=1):
    #idxs = np.random.randint(0, len(avail_cand_idx), req_count)
    idxs = np.random.choice(len(avail_cand_idx), req_count, replace=False)
    c = []
    for i in idxs:
        c.append(avail_cand_idx[i])
    return c

def al_method_dropout(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, req_count = 1):
    if tuned_w is None:
        tuned_w = mod.parameters()
    dropout_ests = []
    for i in range(cand_data.shape[0]):
        dropout_ests.append([])
    for i in range(50):
        #Tune first X layers
        #cand_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, bn_training=True, param_tensor=cand_data[:,-param_dim:],dropout=range(len(tuned_w) - args.tuned_layers))
        #Tune all layers
        #cand_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, bn_training=True, param_tensor=cand_data[:,-param_dim:],dropout=range(len(tuned_w))) # - args.tuned_layers, len(tuned_w)))
        #Tune last X layers
        cand_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, bn_training=True, param_tensor=cand_data[:,-param_dim:],dropout=range(len(tuned_w) - args.tuned_layers, len(tuned_w)),dropout_rate=args.dropout_rate)
        for j in range(cand_data.shape[0]):
            dropout_ests[j].append(cand_ests[j])
    variances = []
    variances_all = []
    for i in avail_cand_idx: #dropout_ests:
        variances.append(torch.sum(torch.var(torch.stack(dropout_ests[i]),dim=0)))
        ests = torch.stack(dropout_ests[i])
        variances_all.append(torch.sum(torch.div(torch.var(ests,dim=0),torch.abs(torch.mean(ests,dim=0)))))
    if args.all_vars == 1:
        idx = np.argmax(variances_all)
    else:
        idx = np.argmax(variances)
    return [avail_cand_idx[idx]] # max_min_dists[idx][0]

def al_method_k_centers(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, req_count = 1, output_dist=True):
    if tuned_w is None:
        tuned_w = mod.parameters()
    cand_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, param_tensor=cand_data[:,-param_dim:])
    tgt_ests = mod(target_data[:,:-param_dim], vars = tuned_w, param_tensor=target_data[:,-param_dim:])
    dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    for i in range(cand_data.shape[0]):
        for j in range(target_data.shape[0]):
            if output_dist:
                dists[i, j] = torch.norm(cand_ests[i] - tgt_ests[j])
            else:
                dists[i, j] = torch.norm(cand_data[i] - target_data[j])

    labeled_cands = np.setdiff1d(range(cand_data.shape[0]), avail_cand_idx)
    max_min_dists = []
    C = list(itertools.combinations(avail_cand_idx, req_count))
    for i in range(len(C)):
        x = C[i]
        min_dists = []
        for pt in range(target_data.shape[0]):
            min_dist = np.inf
            for exp_cands in np.concatenate([labeled_cands, x]):
                min_dist = min(min_dist, dists[exp_cands, pt])
            min_dists.append(min_dist)
        max_min_dists.append([x, max(min_dists)])
    idx = np.argmin(np.array(max_min_dists)[:,1])
    return C[idx] # max_min_dists[idx][0]
            

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--k_model', type=int, help='k shot for loading model params', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=-1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--iter_qry', type=int, help='learn from examples one-at-a-time', default=1)
    argparser.add_argument('--merge_spt_qry', type=int, help='select queries during AL from the test set', default=0)
    argparser.add_argument('--al_method', type=str, help='AL algorithm', default="none")
    argparser.add_argument('--func_type', type=str, help='function type', default="sinusoid")
    argparser.add_argument('--svm_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--batch_sz', type=int, help='task-level inner update learning rate', default=1)
    argparser.add_argument('--grasps', type=int, help='number of grasps per object sample', default=1)
    argparser.add_argument('--tuned_layers', type=int, help='number of grasps per object sample', default=2)
    argparser.add_argument('--split', type=float, help='training/testing data split', default=0.5)
    argparser.add_argument('--dropout_rate', type=float, help='training/testing data split', default=0.5)
    argparser.add_argument('--split_cat', type=int, help='1 if training/testing data is split by category, 0 if split by object id', default=1)
    argparser.add_argument('--all_vars', type=int, help='1 if training/testing data is split by category, 0 if split by object id', default=1)
    argparser.add_argument('--leave_out', type=int, help='affordance number to leave out during training (2-6)', default=-1)


    args = argparser.parse_args()

    main(args)
