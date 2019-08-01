import pdb
import math
import random
import  torch, os, sys
import os.path
import  numpy as np
from numpy.random import RandomState
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
from affordances import Affordances, Affordances2D, Affordances2DTT

def main(args):

    sinusoid = {'name':'sinusoid','class':Sinusoid, 'dims':[1,1,0]}
    polynomial = {'name':'polynomial', 'class':Polynomial, 'dims':[2,1,0]}
    imagenet = {'name':'imagenet', 'class':ImageNet, 'dims':[4096,2,0]}
    grasps = {'name':'grasps', 'class':CornellGrasps, 'dims':[4096,2,0]}
    cat_grasps = {'name':'cat_grasps', 'class':CategorizedGrasps, 'dims':[4096,2,1]} #third number is param length
    affordances = {'name':'affordances', 'class':Affordances, 'dims':[4096,3,0]} #third number is param length
    affordances_2d = {'name':'affordances_2d', 'class':Affordances2D, 'dims':[4096,2,0]} #third number is param length
    affordances_tt = {'name':'affordances_tt', 'class':Affordances2DTT, 'dims':[4096,2,2]} #third number is param length
    data_params = {'affordances':affordances, 'cat_grasps':cat_grasps, 'affordances_2d':affordances_2d, 'affordances_tt':affordances_tt}
    func_data = data_params[args.func_type]

    if args.leave_out >= 0:
        split_txt = ''
        split_num = args.leave_out
        if args.split_cat == 1:
            dir_name = "lo_" + str(args.leave_out) + "/"
        else:
            dir_name = "cat_" + str(args.leave_out) + "/"
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

    #print("loading epoch " + str(valid_epoch))
    #print(args)

    #device = torch.device('cpu')
    device = torch.device('cuda')
    torch.cuda.synchronize()

    #dim_hidden = [4096,500]
    if args.func_type == "cat_grasps":
        dim_hidden = [4096,[512,513], 128]
    if args.func_type == "affordances":
        dim_hidden = [4096,512, 128]
    if args.func_type == "affordances_2d":
        dim_hidden = [4096,512, 128]
    if args.func_type == "affordances_tt":
        dim_hidden = [4096,[512,514], 128]

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

    if args.split_cat == 0:
        prev_net = task_tune(func_data,device,mod,dims)
        #prev_net = None
    else:
        prev_net = None

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
    rand = RandomState(222)



    all_accs = []
    for _ in range(db_train.batch_size):
        all_accs.append([])
    #for i in range(args.task_num):
    for l in range(args.task_num):
        batch_x, batch_y = db_train.next()
        sys.stdout.write("\rTask %i" % l)
        sys.stdout.flush()
        if args.al_method == "random":
            orderings = list(itertools.permutations(range(args.k_spt)))
            orderings = [orderings[i] for i in rand.choice(len(orderings), min(len(orderings),200), replace=False)]
        else:
            orderings = [None]
        for o in orderings:
            for i in range(batch_x.shape[0]):
                queries = None
                accs = []
                c_idx = np.array(range(args.k_spt))
                q_set = torch.from_numpy(batch_x[i, qry_idx:, :]).float().to(device), torch.from_numpy(batch_y[i, qry_idx:, :]).float().to(device) 
                c_data = torch.from_numpy(batch_x[i, :args.k_spt, :]).float().to(device) 
                t_data = q_set[0]
                s_idx = None
                tuned_w = None
                prev_vars = None #torch.zeros(args.k_spt)
                #curr_mod = deepcopy(mod.net)
                var_history = []
                while len(c_idx) >= args.batch_sz: #% args.batch_sz == 0:
                    if args.iter_qry == 1:
                        if args.al_method == "random":
                            s_idx = al_method_random(c_idx, args.batch_sz, o)
                        if args.al_method == "output_space":
                            s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, dist_metric="output")
                        if args.al_method == "input_space":
                            s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, dist_metric="input")
                        if args.al_method == "feature_output":
                            s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, dist_metric="feature_output")
                        if args.al_method == "input_output":
                            s_idx = al_method_k_centers(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz, dist_metric="input_output")
                        if args.al_method == "dropout":
                            s_idx = al_method_dropout(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz)
                        if args.al_method == "gaussian":
                            s_idx,prev_vars = al_method_gaussian(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], prev_vars, args.batch_sz)
                            var_history.append(prev_vars)
                        if args.al_method == "gradient":
                            s_idx = al_method_grad(mod.net, tuned_w, c_data, c_idx, t_data, dims[2], args.batch_sz)
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
                    test_acc, tuned_w = mod.finetuning(qin, ql, q_set[0], q_set[1], dims[2], args.tuned_layers,new_env=(args.split_cat==0),prev_net=prev_net)
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
                #if not args.al_method == "random":
                #    print("\n" + str(queries[2]))
    for j in range(len(all_accs)):
        if False: #args.al_method == "random":
            sum_vals = np.sum(np.array(all_accs),2)[0]
            min_idx = np.argmin(sum_vals)
            sum_f = ['{:.2e}'.format(i) for i in [sum_vals[min_idx]]]
            accs_val = np.array(all_accs[j])[min_idx]
            accs_f = ['{:.2e}'.format(i) for i in accs_val]
            print("\n" + str(orderings[min_idx]))
        if True:
            accs_val = np.array(all_accs[j]).mean(axis=0).astype(np.float16)
            accs_f = ['{:.2e}'.format(i) for i in accs_val]
            sum_vals = np.sum(np.array(all_accs),2)[0]
            sum_f = ['{:.2e}'.format(i) for i in sum_vals]
        print('Loss:', accs_f)
        #print('Sum:', sum_f)
        #print('\nTask ' + str(j) + ' loss:', accs_val)

def al_method_random(avail_cand_idx, req_count=1, o = None):
    #idxs = np.random.randint(0, len(avail_cand_idx), req_count)
    if o is None:
        idxs = np.random.choice(len(avail_cand_idx), req_count, replace=False)
        c = []
        for i in idxs:
            c.append(avail_cand_idx[i])
        return c
    else:
        for i in o:
            if i in avail_cand_idx:
                return [i] 

def al_method_gaussian(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, prev_vars, req_count = 1):
    if tuned_w is None:
        tuned_w = mod.parameters()
    cand_num = cand_data.shape[0]

    # Get dropout predictions
    dropout_ests = []
    for i in range(cand_num):
        dropout_ests.append([])
    for i in range(1000):
        cand_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, bn_training=True, param_tensor=cand_data[:,-param_dim:],dropout=range(len(tuned_w) - args.tuned_layers, len(tuned_w)),dropout_rate=args.dropout_rate)
        for j in range(cand_num):
            dropout_ests[j].append(cand_ests[j])
    feat_ests = mod(cand_data[:,:-param_dim], vars = tuned_w, bn_training=True, param_tensor=cand_data[:,-param_dim:],hook=len(tuned_w)-args.tuned_layers)

    # Get prediction variances
    variances, means, disps = [], [], []
    disp_tensor = torch.zeros(cand_num)
    for i in range(cand_num): #dropout_ests:
        ests = torch.stack(dropout_ests[i])
        #variances.append(torch.sum(torch.var(ests,dim=0)))
        dist_var = 0.0
        for e in ests:
            dist_var += torch.sum((e-torch.mean(ests,dim=0))**2.0)
        dist_var = dist_var / len(ests)
        variances.append(dist_var)
        #variances.append(torch.sum(torch.var(ests,dim=0)))
        means.append(torch.mean(ests,dim=0))
        disps.append(torch.sum(torch.var(ests,dim=0)/torch.abs(torch.mean(ests,dim=0))))
        disp_tensor[i] = torch.sum(torch.var(ests,dim=0)/torch.abs(torch.mean(ests,dim=0)))
    # Get distances between prediction means for each candidate
    dists = torch.zeros([cand_num,cand_num])
    in_dists = torch.zeros([cand_num,cand_num])
    feat_dists = torch.zeros([cand_num,cand_num])
    c1_dists = torch.zeros([cand_num,cand_num])
    c2_dists = torch.zeros([cand_num,cand_num])
    c3_dists = torch.zeros([cand_num,cand_num])
    for i in range(cand_num):
        for j in range(cand_num):
            dists[i,j] = torch.sum((means[i] - means[j])**2.0)
            in_dists[i,j] = torch.sum((cand_data[i] - cand_data[j])**2.0)
            feat_dists[i,j] = torch.sum((feat_ests[i] - feat_ests[j])**2.0)
            c1_dists[i,j] = torch.sqrt(in_dists[i,j]) * torch.sqrt(dists[i,j])
            if i == j:
                c2_dists[i,j] = np.inf
                c3_dists[i,j] = 0
            else:
                c2_dists[i,j] = torch.sqrt(in_dists[i,j]) * torch.sqrt(dists[i,j])
                c3_dists[i,j] = torch.sqrt(feat_dists[i,j]) / torch.sqrt(dists[i,j])
    c3_mean = torch.sum(c3_dists)/len(c3_dists.nonzero())
    #c3_dists = torch.sum((c3_dists-c3_mean)**2.0,dim=0)
    c3_dists = (c3_dists-c3_mean)**2.0

    dists = dists/torch.max(dists)
    #dists = torch.ones_like(dists) - dists
    in_dists = in_dists/torch.max(in_dists)
    feat_dists = feat_dists/torch.max(feat_dists)
        
    # Calculate sum of weighted variances after each candidate is labeled
    total_vars, total_disps, in_disps, feat_disps = [], [], [], []
    total_vars2, total_disps2, in_disps2, feat_disps2 = [], [], [], []
    var_change = []
    for i in avail_cand_idx:
        total_var = 0.0
        total_disp = 0.0
        in_disp = 0.0
        feat_disp = 0.0
        for j in avail_cand_idx:
            total_disp += (disps[j] * dists[i,j])
            total_var += (variances[j] * dists[i,j])
            in_disp += (disps[j] * in_dists[i,j])
            feat_disp += (disps[j] * feat_dists[i,j])
        disp = torch.mul(disp_tensor, dists[i])
        total_disps2.append(sum([disp[j] for j in avail_cand_idx]))
        total_vars.append(total_var)
        total_disps.append(total_disp)
        in_disps.append(in_disp)
        feat_disps.append(feat_disp)
        if prev_vars is not None:
            var_change.append(variances[i] - prev_vars[i])
    var = [variances[i] for i in avail_cand_idx]
    #dist_sum = torch.sum(c1_dists,dim=0)
    #dist_sum = [sum([c1_dists[i,j] for j in avail_cand_idx if not i==j]) for i in avail_cand_idx]
    dist_sum = [[c1_dists[i,j] for j in avail_cand_idx] for i in avail_cand_idx]
    c3_sum = [sum([c3_dists[i,j] for j in avail_cand_idx if not i==j]) for i in avail_cand_idx]
    #Select based on least expected variance:
    idx = np.argmax(np.array(total_vars)) 
    #idx1 = np.argmin(np.array(c3_sum)) 
    idx1 = np.argmin(np.max(np.array(dist_sum),0))
    #idx1 = np.argmax(np.min(np.array(dist_sum),0))
    #Select based on least expected disparity, using 3D output space to determine candidate similarity:
    idx2 = np.argmin(np.array(total_disps))
    #Select based on least expected disparity, using 4096D input space to determine candidate similarity:
    idx3 = np.argmin(np.array(in_disps)) 
    #Select based on least expected disparity, using 512D intermediate space to determine candidate similarity:
    idx4 = np.argmin(np.array(feat_disps))
    #Output-space disparity returns best results right now
    pdb.set_trace()
    return [avail_cand_idx[idx1]], np.array(variances) # max_min_dists[idx][0]
    #idx5 = np.argmax(np.array(var_change))
    #return [avail_cand_idx[idx5]], np.array(variances) # max_min_dists[idx][0]

def al_method_grad(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, req_count = 1):
    if tuned_w is None:
        tuned_w = mod.parameters()
    if param_dim > 0:
        p_cand = cand_data[:,-param_dim:]
        x_cand = cand_data[:,:-param_dim]
        p_tgt = target_data[:,-param_dim:]
        x_tgt = target_data[:,:-param_dim]
    else:
        p_cand = cand_data.shape[0] * [None]
        x_cand = cand_data
        p_tgt = target_data.shape[0] * [None]
        x_tgt = target_data
    dropout_ests, loss_ests, feat_loss_ests = [], [], []

    in_dists = []
    dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    for i in range(cand_data.shape[0]):
        in_dists.append(sum([torch.norm(cand_data[i] - target_data[j]).item() for j in range(target_data.shape[0])]))
        for j in range(target_data.shape[0]):
            dists[i, j] = torch.norm(cand_data[i] - target_data[j])


    for i in range(cand_data.shape[0]):
        dropout_ests.append(0.0)
        loss_ests.append(0.0)
        feat_loss_ests.append(0.0)
    original_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand) 
    feat_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,hook=len(tuned_w)-args.tuned_layers)
    for i in range(1000):
        if args.all_layers == 1:
            #Tune all layers
            cand_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,dropout=range(len(tuned_w)), dropout_rate=args.dropout_rate) 
        else:
            #Tune last X layers
            cand_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,dropout=range(len(tuned_w) - args.tuned_layers, len(tuned_w)),dropout_rate=args.dropout_rate)
        for j in range(cand_data.shape[0]):
            loss = F.mse_loss(cand_ests[j], original_ests[j])
            grad = torch.autograd.grad(loss, tuned_w[-args.tuned_layers:], retain_graph=True)
            norm = 0.0
            for layer in grad:
                norm += (layer.data.norm(2).item()**2.0) 
            loss_ests[j] += torch.norm(loss * x_cand[j]).item()
            feat_loss_ests[j] += torch.norm(loss * feat_ests[j]).item()
            dropout_ests[j] += (math.sqrt(norm)/100.0)

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
    idx1 = np.argmin(np.array(max_min_dists)[:,1])

    exp_grads = [dropout_ests[d[0][0]] for d in max_min_dists]
    #data = np.array(exp_grads)/np.array(max_min_dists)[:,1]
    #idx2 = np.argmax(data)

    idx = np.argmax(np.array(exp_grads))
    return [avail_cand_idx[idx]] # max_min_dists[idx][0]

def al_method_dropout(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, req_count = 1):
    if tuned_w is None:
        tuned_w = mod.parameters()
    if param_dim > 0:
        p_cand = cand_data[:,-param_dim:]
        x_cand = cand_data[:,:-param_dim]
    else:
        p_cand = cand_data.shape[0] * [None]
        x_cand = cand_data
    dropout_ests = []
    for i in range(cand_data.shape[0]):
        dropout_ests.append([])
    for i in range(1000):
        if args.all_layers == 1:
            #Tune all layers
            cand_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,dropout=range(len(tuned_w)), dropout_rate=args.dropout_rate) 
        else:
            #Tune last X layers
            cand_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,dropout=range(len(tuned_w) - args.tuned_layers, len(tuned_w)),dropout_rate=args.dropout_rate)
        for j in range(cand_data.shape[0]):
            dropout_ests[j].append(cand_ests[j])
    variances = []
    for i in avail_cand_idx: #dropout_ests:
        ests = torch.stack(dropout_ests[i])
        mean = torch.mean(ests)
        dist = 0.0
        for c in dropout_ests[i]:
            dist += torch.sum((c - mean)**2.0)
        variances.append(dist/len(dropout_ests[i]))
    idx = np.argmax(variances)
    return [avail_cand_idx[idx]] # max_min_dists[idx][0]

def al_method_k_centers(mod, tuned_w, cand_data, avail_cand_idx, target_data, param_dim, req_count = 1, dist_metric="output"):
    if tuned_w is None:
        tuned_w = mod.parameters()
    if param_dim > 0:
        p_cand = cand_data[:,-param_dim:]
        x_cand = cand_data[:,:-param_dim]
        p_tgt = target_data[:,-param_dim:]
        x_tgt = target_data[:,:-param_dim]
    else:
        p_cand = cand_data.shape[0] * [None]
        x_cand = cand_data
        p_tgt = target_data.shape[0] * [None]
        x_tgt = target_data
    cand_ests = mod(x_cand, vars = tuned_w, param_tensor=p_cand)
    tgt_ests = mod(x_tgt, vars = tuned_w, param_tensor=p_tgt)
    cand_feat_ests = mod(x_cand, vars = tuned_w, bn_training=True, param_tensor=p_cand,hook=len(tuned_w)-args.tuned_layers)
    tgt_feat_ests = mod(x_tgt, vars = tuned_w, bn_training=True, param_tensor=p_tgt,hook=len(tuned_w)-args.tuned_layers)
    dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    out_dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    in_dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    feat_dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    for i in range(cand_data.shape[0]):
        for j in range(target_data.shape[0]):
            in_dists[i, j] = torch.norm(cand_data[i] - target_data[j])
            out_dists[i, j] = torch.norm(cand_ests[i] - tgt_ests[j])
            feat_dists[i, j] = torch.norm(cand_feat_ests[i] - tgt_feat_ests[j])
            if dist_metric == "output":
                dists[i, j] = out_dists[i,j]
            elif dist_metric == "feature_output":
                dists[i, j] = out_dists[i,j] * feat_dists[i,j]
            elif dist_metric == "input_output":
                dists[i, j] = out_dists[i,j] * in_dists[i,j]
            elif dist_metric == "input":
                dists[i, j] = in_dists[i,j]

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
    #pdb.set_trace()
    return C[idx] # max_min_dists[idx][0]
            
def task_tune(func_data,device,mod,dims):
    task_train = func_data['class'](
                       batchsz=1, #args.task_num,
                       k_shot=0,
                       k_qry=0,
                       num_grasps=args.grasps,
                       split=args.leave_out,
                       train=True,
                       split_cat=0)
    batch_x, batch_y = task_train.all_samples(str(task_train.ignored_objects[0]))
    qin, ql = torch.from_numpy(batch_x).float().to(device), torch.from_numpy(batch_y).float().to(device)
    tuned_net = deepcopy(mod.net)
    test_acc, tuned_w = mod.finetuning(qin, ql, qin, ql, dims[2], args.tuned_layers,new_env=False)
    base = len(tuned_net.parameters())-args.tuned_layers
    for i in range(args.tuned_layers):
        tuned_net.state_dict()['vars.'+str(base+i)].copy_(tuned_w[base+i])
    return tuned_net

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
    argparser.add_argument('--all_layers', type=int, help='affordance number to leave out during training (2-6)', default=-1)


    args = argparser.parse_args()

    main(args)
