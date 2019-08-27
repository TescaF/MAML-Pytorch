import torch, os, sys, pdb, math
from torch.nn import functional as F
import numpy as np
import itertools
from numpy.random import RandomState

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))
    pdb.set_trace()

def al_method_grad(mod, w, x_cand, avail_cands, args, c):
    if w is None:
        tuned_w = mod.parameters()
    else:
        tuned_w = w

    #dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    #for i in range(cand_data.shape[0]):
    #    for j in range(target_data.shape[0]):
    #        dists[i, j] = torch.norm(cand_data[i] - target_data[j]).item()


    grad_ests = []
    for i in range(len(avail_cands)): #x_cand.shape[0]):
        grad_ests.append(0.0)

    with torch.no_grad():
        logits = mod(x_cand, vars = tuned_w, bn_training=False)
        original_ests = F.softmax(logits, dim=1).argmax(dim=1)
    for i in range(100):
        cand_ests = mod(x_cand, vars = tuned_w, bn_training=False, dropout_rate=args.dropout_rate) 
        for j in range(len(avail_cands)): #x_cand.shape[0]):
            if c: #classification
                loss = F.cross_entropy(cand_ests[avail_cands[j]].unsqueeze(0), original_ests[avail_cands[j]].unsqueeze(0))
            else:
                loss = F.mse_loss(cand_ests[avail_cands[j]], original_ests[avail_cands[j]])
            grad = torch.autograd.grad(loss, tuned_w, retain_graph=True)
            norm = 0.0
            for layer in grad:
                norm += (layer.data.norm(2).item()**2.0) 
            grad_ests[j] += math.sqrt(norm)

    '''labeled_cands = np.setdiff1d(range(cand_data.shape[0]), avail_cand_idx)
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

    pdb.set_trace()
    exp_grads = [dropout_ests[d[0][0]] for d in max_min_dists]

    idx = np.argmax(np.array(exp_grads))'''
    return avail_cands[np.argmax(np.array(grad_ests))] # max_min_dists[idx][0]

def al_method_learned(mod,an, x, w, n):
    samples = list(itertools.combinations(list(range(x.shape[0])), r=n))
    ## Get training loss for each sample
    variances=[]
    for s in samples:
        x_s = torch.stack([x[i] for i in s])
        logits = mod(x_s, vars=w, bn_training=True)[:,-an:].squeeze(1)
        variances.append(torch.var(logits).item())
        
    pdb.set_trace()
    queries = samples[np.argmax(np.array(variances))]
    return queries
    

def al_method_dropout(mod, tuned_w, cand_data, avail_cand_idx, args):
    if tuned_w is None:
        tuned_w = mod.parameters()
    x_cand = cand_data
    dropout_ests = []
    for i in range(cand_data.shape[0]):
        dropout_ests.append([])
    #for i in range(1000):
    for i in range(100):
        cand_ests = mod(x_cand, vars = tuned_w, bn_training=True, dropout_rate=args.dropout_rate) 

        for j in range(cand_data.shape[0]):
            dropout_ests[j].append(cand_ests[j].cpu().detach().numpy())
    variances = []
    for i in avail_cand_idx: #dropout_ests:
        ests = np.stack(dropout_ests[i])
        mean = np.mean(ests,axis=0)
        dist = 0.0
        for c in dropout_ests[i]:
            dist += np.sum((c - mean)**2.0)
        variances.append(dist/len(dropout_ests[i]))
    idx = np.argmax(variances)
    return avail_cand_idx[idx] # max_min_dists[idx][0]

def al_method_learned_centers(al_mod, mod, x_cand):
    hooks = mod(x_cand, vars = mod.parameters(), bn_training=True, hook=len(mod.config)-1)
    al_space = al_mod(hooks, vars = al_mod.parameters(), bn_training=True)
    al_dists = torch.cuda.FloatTensor(x_cand.shape[0], x_cand.shape[0]).fill_(0.0)
    for i in range(x_cand.shape[0]):
        for j in range(x_cand.shape[0]):
            al_dists[i, j] = torch.norm(al_space[i] - al_space[j])

    max_min_dists = []
    samples = list(itertools.combinations(list(range(x_cand.shape[0])), r=3))

    # For each potential labeled query set...
    for i in range(len(samples)):
        s = samples[i]
        min_dists = []
        # ...find the distance between each point and...
        for pt in range(x_cand.shape[0]):
            min_dist = torch.cuda.FloatTensor([np.inf])
            # ...the nearest labeled query in the current set
            for ps in s:
                min_dist = torch.min(min_dist, al_dists[ps, pt])
            min_dists.append(min_dist)
        # Save the point that is the longest distance from a labeled query
        max_min_dists.append(torch.max(torch.stack(min_dists)))
    max_min_dists = torch.stack(max_min_dists)
    best = samples[torch.argmin(max_min_dists)]
    return best

def al_method_k_centers(al_mod, mod, tuned_w, cand_data, avail_cand_idx, target_data, dist_metric="output"):
    if tuned_w is None:
        tuned_w = mod.parameters()
    x_cand = cand_data
    x_tgt = target_data
    cand_ests = mod(x_cand, vars = tuned_w, bn_training=False)
    tgt_ests = mod(x_tgt, vars = tuned_w, bn_training=False)
    dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    out_dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    in_dists = np.ones((cand_data.shape[0], target_data.shape[0]))
    for i in range(cand_data.shape[0]):
        for j in range(target_data.shape[0]):
            in_dists[i, j] = torch.norm(cand_data[i] - target_data[j])
            out_dists[i, j] = torch.norm(cand_ests[i] - tgt_ests[j])
            al_dists[i, j] = torch.norm(al_space[i] - al_space[j])
            if dist_metric == "output":
                dists[i, j] = out_dists[i,j]
            elif dist_metric == "input_output":
                dists[i, j] = out_dists[i,j] * in_dists[i,j]
            elif dist_metric == "input":
                dists[i, j] = in_dists[i,j]
            elif dist_metric == "al":
                dists[i, j] = al_dists[i,j]

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
    #idx = np.argmin(np.array(max_min_dists)[:,1])
    return avail_cand_idx[idx] # max_min_dists[idx][0]

class QuerySelector():
    def __init__(self):
        self.query_order = None
        self.order_idx = 0
        self.rand = RandomState(222)
        self.query_order = []

    def all_queries(self, cands, k):
        orderings = itertools.combinations(cands, k) 
        #orderings = [np.random.choice(cands, k, replace=False) for _ in range(100)]
        #orderings = [np.random.permutation(cands) for _ in range(100)]
        queries = []
        for o in orderings:
            queries.append([cands[i] for i in o])
        return queries

    def next_order(self):
        self.order_idx += 1
        if self.order_idx >= len(self.query_order):
            return False
        return True

    def al_query(self, al_mod, net, cands):
        if len(self.query_order) == 0:
             self.query_order = list(al_method_learned_centers(al_mod, net, cands))
        s = self.query_order.pop(0)
        return s

    def query(self, args, net, w, cands, avail_cands, targets, k, method, classification, al_sz):
        if method == "random":
           if len(self.query_order) == 0:
                self.query_order = self.all_queries(avail_cands, k)
           s = self.query_order[self.order_idx].pop(0)
        if method == "output_space":
            s = al_method_k_centers(net, w, cands, avail_cands, targets, dist_metric="output")
        if method == "input_space":
            s = al_method_k_centers(net, w, cands, avail_cands, targets, dist_metric="input")
        if method == "input_output":
            s = al_method_k_centers(net, w, cands, avail_cands, targets, dist_metric="input_output")
        if method == "dropout":
            s = al_method_dropout(net, w, cands, avail_cands, args)
        if method == "gaussian":
            s = al_method_gaussian(net, w, cands)
        if method == "gradient":
            s = al_method_grad(net, w, cands, avail_cands, args, classification)
        if method == "learned":
           if len(self.query_order) == 0:
                self.query_order = list(al_method_learned_centers(al_mod, net, cands))
                pdb.set_trace()
           s = self.query_order.pop(0)
           #for q in self.query_order:
           #     if q in avail_cands:
           #         return q
        return s

