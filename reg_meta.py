import itertools
import pdb
import  torch
#from torch.utils.tensorboard import SummaryWriter
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    learner import Learner
from    copy import deepcopy

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

def print_backward_stack(mod):
    l = mod.grad_fn
    while l is not None:
        print(l)
        if len(l.next_functions) == 0:
            return
        l = l.next_functions[0][0]


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, al_sz, loss_fn=F.cross_entropy, accs_fn=torch.eq):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.k_model = args.k_model
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.alpha = args.alpha
        self.net = Learner(config)
        self.an = al_sz
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.loss_fn = loss_fn
        self.accs_fn = accs_fn

    def al_dataset(self, x, y, w, n, grad=True):
        sample_loss = self.samplewise_loss(x,y,w,n)
        if grad:
            return self.al_dataset_inner(sample_loss,x,y,w,n)
        else:
            with torch.no_grad():
                return self.al_dataset_inner(sample_loss,x,y,w,n)

    def al_dataset_inner(self, sample_loss, x, y, w, n):
        inputs, labels = [], []
        ## Get model data prior to final layer
        #hooks = self.net(x, vars=w, bn_training=False,hook=len(self.net.config)-1)

        ## Get AL labels for each sample pair
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                diff = sample_loss[j] - sample_loss[k]
                if diff == 0:
                    continue
                labels.append(np.sign(diff))
                #inputs.append(torch.stack([x[j], x[k]]))
                inputs.append([j, k])
                #h_inputs.append(torch.stack([hooks[j], hooks[k]]))
        #h_inputs = torch.stack(h_inputs)
        #del hooks
        if self.an == 2:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.cuda.FloatTensor(labels)
        return inputs, labels

    def sampleset_loss(self, x, y, w, n):
        sample_loss = [0 for _ in range(x.shape[0])]

        ## Get sample sets
        samples = list(itertools.combinations(list(range(x.shape[0])), r=n))

        ## Get training loss for each sample set
        losses=[]
        for s in samples:
            s_weights = deepcopy(list(w)) 
            x_s = torch.stack([x[i] for i in s])
            y_s = torch.stack([y[i] for i in s])

            ## Tune model on current sample set
            for k in range(self.update_step):
                logits = self.net(x_s, vars=s_weights, bn_training=True)
                if self.an > 0:
                    logits = logits[:,:-self.an]
                
                loss = self.loss_fn(logits, y_s)
                grad = torch.autograd.grad(loss, s_weights)
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                del loss,logits,grad

            ## Get post-training loss on all samples
            logits_q = self.net(x, s_weights, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            loss_q = self.loss_fn(logits_q, y)
            losses.append(loss_q.item())
            del loss_q,logits_q,s_weights

        comp_labels, comp_idxs = [], []
        for i in range(len(losses)):
            for j in range(len(losses)):
                diff = losses[i] - losses[j]
                if not diff == 0:
                    comp_idxs.append([i,j])
                    comp_labels.append(np.sign(diff))
        loss_sort = np.argsort(np.array(losses))
        for l in range(len(loss_sort)):
            s = samples[loss_sort[l]]
            for idx in s:
                sample_loss[idx] += l
            
        return samples, comp_idxs, comp_labels, losses #losses #sample_loss

    def al_forward(self, mod, x, y, k):
        total_loss = 0.0
        total_acc = 0.0
        for i in range(x.shape[0]):
            loss, acc = self.queryset_loss(mod, x[i], y[i], k)
            total_acc += acc
            total_loss += loss
        total_loss = total_loss / x.shape[0]
        total_acc = total_acc / x.shape[0]
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()

        '''total_loss2 = 0.0
        total_acc2 = 0.0
        for i in range(x.shape[0]):
            loss2, acc2 = self.queryset_loss(mod, x[i], y[i], k)
            total_acc2 += acc2
            total_loss2 += loss2
        total_loss2 = total_loss2 / x.shape[0]
        total_acc2 = total_acc2 / x.shape[0]
        print("A: " + str(total_acc) + " B: " + str(total_acc2))'''
        return total_loss.item(), total_acc
        
    def sample_dist(self, samples, losses, pts, k):
        dists = torch.cuda.FloatTensor(pts, pts).fill_(0.0)
        pairs = list(itertools.combinations(list(range(pts)), r=k-1))
        for pair in pairs:
            comp_samples = []
            for s in range(len(samples)):
                if np.array([i in samples[s] for i in pair]).all():
                    comp_samples.append(s)
            for c1 in comp_samples:
                e1 = [x for x in samples[c1] if x not in pair][0]
                for c2 in comp_samples:
                    e2 = [x for x in samples[c2] if x not in pair][0]
                    dists[e1,e2] += ((losses[c1] - losses[c2])**2.0)

        d = []
        pairs = list(itertools.combinations(list(range(pts)), r=2))
        for pair1 in pairs:
            for pair2 in pairs:
                if not pair1 == pair2:
                    d.append(dists[pair1[0],pair1[1]] - dists[pair2[0],pair2[1]])
        d = torch.stack(d)

        return d

    def queryset_loss(self, mod, x, y, k):
        pts = x.shape[0]

        # Get all outputs in model space
        with torch.no_grad():
            hooks = mod.net(x, vars=mod.net.parameters(), bn_training=True,hook=len(mod.net.config)-1)
        proj = self.net(hooks, vars=self.net.parameters(), bn_training=True)

        # Get distances between all points in model space
        #dists = torch.cuda.FloatTensor(pts, pts)
        dists = []
        pairs = list(itertools.combinations(list(range(pts)), r=2))
        for pair1 in pairs:
            for pair2 in pairs:
                if not pair1 == pair2:
                    dists.append(torch.norm(proj[pair1[0]] - proj[pair1[1]]) - torch.norm(proj[pair2[0]] - proj[pair2[1]]))
        dists = torch.stack(dists)

        # Get all query sets and comparison labels
        #samples = list(itertools.combinations(list(range(pts)), r=k))
        samples, comp_idxs, comp_labels, losses = mod.sampleset_loss(x,y,mod.net.parameters(),k)
        loss_dists = self.sample_dist(samples, losses, pts, k)
        '''max_min_dists = []
        # For each potential labeled query set...
        for i in range(len(samples)):
            s = samples[i]
            min_dists = []
            # ...find the distance between each point and...
            for pt in range(pts):
                min_dist = torch.cuda.FloatTensor([np.inf])
                # ...the nearest labeled query in the current set
                for ps in s:
                    min_dist = torch.min(min_dist, dists[ps, pt])
                min_dists.append(min_dist)
            # Save the point that is the longest distance from a labeled query
            max_min_dists.append(torch.max(torch.stack(min_dists)))

        diffs = []
        for c in comp_idxs:
            diffs.append(max_min_dists[c[0]] - max_min_dists[c[1]])
        diffs = torch.stack(diffs)'''
        loss = self.smooth_hinge_loss(dists, torch.sign(loss_dists))
        pred = torch.sign(dists)
        corrects = torch.eq(torch.sign(loss_dists), pred).sum().item()/pred.shape[0]
        return loss, corrects


    def samplewise_loss_old(self, x, y, w):
        sample_loss = []

        ## Get training loss for each sample
        for s in range(x.shape[0]):
            s_weights = deepcopy(list(w)) 

            ## Tune model on current sample
            for k in range(self.update_step):
                logits = self.net(x, vars=s_weights, bn_training=True)[s].unsqueeze(0)
                if self.an > 0:
                    logits = logits[:,:-self.an]
                loss = self.loss_fn(logits, y[s].unsqueeze(0))
                grad = torch.autograd.grad(loss, s_weights)
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                del loss,logits,grad

            ## Get post-training loss on all samples
            logits_q = self.net(x, s_weights, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            loss_q = self.loss_fn(logits_q, y)
            sample_loss.append(loss_q.item())
            del loss_q,logits_q,s_weights

        return sample_loss

    def smooth_hinge_loss(self, y, l):
        loss = torch.cuda.FloatTensor([0.0])
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            if ty <= 0:
                loss += (0.5 - ty)
            elif 0 < ty <= 1:
                loss += (0.5 * ((1-ty)**2.0))
        return loss / y.shape[0] 

    def newish_forward(self, x_spt, y_spt, x_qry, y_qry,print_flag=False):
        sample_loss = []
        losses_q, losses_al = 0, 0
        corrects = [0 for _ in range(self.update_step + 1)]
        al_corrects = 0.0
        task_num = x_spt.shape[0]

        for i in range(task_num):
            sample_loss = [0 for _ in range(x_spt.shape[1])]
            samples = torch.cat([x_spt[i],x_qry[i]])
            ## Get loss before first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], None, bn_training=True)
                if self.an > 0:
                    logits_q = logits_q[:,:-self.an]
                if self.accs_fn is None:
                    correct = self.loss_fn(logits_q, y_qry[i]).item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    del pred_q
                corrects[0] = corrects[0] + correct

            ## Get training loss for each sample
            '''ts = [] #training sets
            for j in range(x_spt.shape[1]):
                ts += itertools.combinations(list(range(x_spt.shape[1])), r=j)'''
            ts = [[x] for x in range(x_spt.shape[1])]
            #ts = []
            #for j in range(x_spt.shape[1]):
            #    ts += list(itertools.combinations(list(range(x_spt.shape[1])), r=j))
            #ts = list(filter(None, ts))
            res = []
            for s in ts:
                s_weights = deepcopy(list(self.net.parameters())) 
                ## Tune model on current sample
                for k in range(self.update_step):
                    logits = self.net(samples, vars=s_weights, bn_training=True)
                    if self.an > 0:
                        logits = logits[:,:-self.an]
                    loss = 0
                    for idx in s:
                        loss += self.loss_fn(logits[idx].unsqueeze(0), y_spt[i,idx].unsqueeze(0))
                    loss = loss / len(s)
                    grad = torch.autograd.grad(loss, s_weights)
                    s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                    del loss,logits,grad

                    # Get accuracy after update
                    logits_q = self.net(x_qry[i], s_weights, bn_training=True)
                    if self.an > 0:
                        logits_q = logits_q[:,:-self.an]
                    with torch.no_grad():
                        if self.accs_fn is None:
                            correct = self.loss_fn(logits_q, y_qry[i]).item()
                        else:
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                            del pred_q
                        corrects[k + 1] += (correct/len(ts))

                ## Get post-tuning losses for task
                loss_q = self.loss_fn(logits_q, y_qry[i])
                res.append(loss_q.item())
                if len(s) == 1:
                    sample_loss[s[0]] = loss_q.item()
                losses_q += (loss_q / len(ts))
                del s_weights,logits_q
                torch.cuda.empty_cache()

            ## Calculate AL loss
            '''al_logits = self.net(samples, vars=self.net.parameters(), bn_training=True)
            #al_logits = self.net(x_spt[i], vars=self.net.parameters(), bn_training=True)
            if self.an > 0:
                al_logits = al_logits[:,-self.an:]
            ## Get AL output for each sample pair
            comps,al_labels = [],[]
            for j in range(x_spt.shape[1]):
                for k in range(x_spt.shape[1]):
                    diff = sample_loss[j] - sample_loss[k]
                    if not diff == 0:
                        comps.append(al_logits[j] - al_logits[k])
                        al_labels.append(np.sign(diff))
            logits = torch.stack(comps)
            if self.an == 2:
                al_labels = torch.cuda.LongTensor(al_labels)
                loss = F.cross_entropy(logits, al_labels)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
            else:
                al_labels = torch.cuda.FloatTensor(al_labels)
                loss = self.smooth_hinge_loss(logits, al_labels)
                pred = torch.sign(logits)
            losses_al += loss
            al_corrects += torch.eq(al_labels, pred.squeeze(1)).sum().item()/len(al_labels)'''

        ## Get total losses after all tasks
        loss_q =  losses_q / task_num
        loss_al = self.alpha * losses_al / task_num
        total_loss = loss_q #+ loss_al
        if print_flag:
            print("total: %.2f    loss: %.2f    al: %.2f" %(total_loss, loss_q, loss_al))

        ## Optimize parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()
        loss_results = [loss_q.item(), 0, total_loss.item()]
        #loss_results = [loss_q.item(), loss_al.item(), total_loss.item()]
        del loss_q,losses_q,losses_al,loss_al,total_loss

        if len(x_spt.shape) > 4:
            accs = np.array(corrects) / (x_qry.shape[1] * task_num)
        else:
            accs = np.array(corrects) / task_num
        al_accs = np.array(al_corrects) / task_num
        return accs, al_accs, loss_results

    def forward(self, x_spt, y_spt, x_qry, y_qry,print_flag=False):
        task_num = x_spt.size(0)
        losses_q, losses_al = 0, 0
        corrects = [0 for _ in range(self.update_step + 1)]
        al_corrects = 0.0

        for i in range(task_num):
            s_weights = self.net.parameters()

            ## Get loss before first update
            logits_q = self.net(x_qry[i], None, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            with torch.no_grad():
                if self.accs_fn is None:
                    correct = self.loss_fn(logits_q, y_qry[i]).item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    del pred_q
                corrects[0] = corrects[0] + correct

            ## Update model w.r.t. task loss
            for k in range(self.update_step):

                # Update step
                #with torch.no_grad():
                #    hooks = self.net(x_spt[i], vars=s_weights, bn_training=True,hook=len(self.net.config)-1)
                #logits_a = self.net(hooks, vars=s_weights, bn_training=True,last_layer=True)
                logits_a = self.net(x_spt[i], vars=s_weights, bn_training=True)
                if self.an > 0:
                    logits_a = logits_a[:,:-self.an]
                loss = self.loss_fn(logits_a, y_spt[i])
                grad_a = list(torch.autograd.grad(loss, s_weights,allow_unused=True))
                for g in range(len(grad_a)):
                    if grad_a[g] is None:
                        grad_a[g] = torch.zeros_like(s_weights[g])
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, s_weights)))
                del loss, logits_a, grad_a

                # Get accuracy after update
                logits_q = self.net(x_qry[i], s_weights, bn_training=True)
                if self.an > 0:
                    logits_q = logits_q[:,:-self.an]
                with torch.no_grad():
                    if self.accs_fn is None:
                        correct = self.loss_fn(logits_q, y_qry[i]).item()
                    else:
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        del pred_q
                    corrects[k + 1] = corrects[k + 1] + correct

            ## Get post-tuning losses for task and AL objectives
            losses_q += self.loss_fn(logits_q, y_qry[i])
            if print_flag and (self.accs_fn is None):
                print((logits_q-y_qry[i])*torch.cuda.FloatTensor([4.4,2.1]))
            del s_weights,logits_q
            torch.cuda.empty_cache()
            #al_loss1, _ = self.al_test(x_spt[i], y_spt[i], w=self.net.parameters())
            if self.alpha > 0:
                al_loss2, al_acc = self.al_test(x_qry[i], y_qry[i], w=self.net.parameters(), n=x_spt.shape[1])
                losses_al += al_loss2
                al_corrects += al_acc
            else:
                losses_al = torch.cuda.FloatTensor([0.0])
                al_loss2 = torch.cuda.FloatTensor([0.0])

        ## Get total losses after all tasks
        loss_q =  losses_q / task_num
        loss_al = self.alpha * losses_al / task_num
        total_loss = loss_q + loss_al
        if print_flag:
            print("total: %.2f    loss: %.2f    al: %.2f" %(total_loss, loss_q, loss_al))

        ## Optimize parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()
        loss_results = [loss_q.item(), loss_al.item(), total_loss.item()]
        del loss_q,losses_q,al_loss2,loss_al,total_loss

        if len(x_spt.shape) > 4:
            accs = np.array(corrects) / (x_qry.shape[1] * task_num)
        else:
            accs = np.array(corrects) / task_num
        al_accs = np.array(al_corrects) / task_num
        return accs, al_accs, loss_results

    def al_test(self, x, y, w, n):
        ## Get sample pairs and comparison labels
        al_samples, al_idxs, al_labels = self.sampleset_loss(x, y, self.net.parameters(), n)
        #al_inputs, al_labels = self.al_dataset(x, y, self.net.parameters(), n, grad=True)
        logits,diffs = [],[]
        for s in al_samples:
            x_s = torch.stack([x[i] for i in s])
            logits.append(self.net(x_s, vars=w, bn_training=True)[:,-self.an:].squeeze(1))
        for i in al_idxs:
            v1 = torch.var(logits[i[0]])
            v2 = torch.var(logits[i[1]])
            diffs.append(v2-v1)
        ## Compare AL output to comparison labels
        diffs = torch.stack(diffs)
        al_labels = torch.cuda.FloatTensor(al_labels)
        if self.an == 2:
            loss = F.cross_entropy(diffs, al_labels)
            pred = F.softmax(diffs, dim=1).argmax(dim=1)
        else:
            loss = self.smooth_hinge_loss(diffs, al_labels)
            pred = torch.sign(diffs)
        corrects = torch.eq(al_labels, pred).sum().item()
        del logits
        return loss, corrects/al_labels.shape[0]

    #def finetuning(self, x_spt, y_spt, x_qry, y_qry):
    def finetuning(self, lbl_idxs, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)
        fast_weights = list(net.parameters())

        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            if self.accs_fn is None:
                correct = self.loss_fn(logits_q, y_qry).item()
            else:
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = self.accs_fn(pred_q, y_qry).sum().item()/querysz
            corrects[0] = corrects[0] + correct

        ## Update model w.r.t. task loss
        for k in range(self.update_step_test):
            logits_a = self.net(x_spt, vars=fast_weights, bn_training=True)
            if self.an > 0:
                logits_a = logits_a[:,:-self.an]
            loss = self.loss_fn(logits_a, y_spt)
            #loss = torch.cuda.FloatTensor([0.0]) 
            #for idx in lbl_idxs:
            #    loss += self.loss_fn(logits_a[idx], y_lbl[idx])
            #if len(lbl_idxs) == 5:
            #    pdb.set_trace()
            #loss = loss/self.k_model
            #loss = loss/self.k_model #*self.k_model)
            #loss = loss/len(lbl_idxs) #*self.k_model)
            lr_ratio = 1 #len(lbl_idxs)/self.k_model
            #lr_ratio = min(1.0, len(lbl_idxs)/self.k_model)
            grad_a = list(torch.autograd.grad(loss, fast_weights,allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    grad_a[g] = torch.zeros_like(fast_weights[g])
            fast_weights = list(map(lambda p: p[1] - (lr_ratio * self.update_lr * p[0]), zip(grad_a, fast_weights)))
            del loss, logits_a, grad_a

            # Get accuracy after update
            with torch.no_grad():
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                if self.an > 0:
                    logits_q = logits_q[:,:-self.an]
                if self.accs_fn is None:
                    correct = self.loss_fn(logits_q, y_qry).item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry).sum().item()/querysz  # convert to numpy
                    del pred_q
                corrects[k + 1] = corrects[k + 1] + correct

        '''for k in range(self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            if self.an > 0:
                logits = logits[:,:-self.an]
            loss = 0.0
            for idx in lbl_idxs:
                loss += self.loss_fn(logits[idx].unsqueeze(0), y_lbl[idx])
            loss = loss/len(lbl_idxs)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            with torch.no_grad():
                logits_q = net(x_qry, fast_weights, bn_training=True)
                pdb.set_trace()
                if self.an > 0:
                    logits_q = logits_q[:,:-self.an]
                if self.accs_fn is None:
                    loss_q = self.loss_fn(logits_q, y_qry)
                    correct = loss_q.item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry).sum().item()  # convert to numpy
                #correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct'''
        del net
        accs = np.array(corrects) 
        return accs,fast_weights

def main():
    pass


if __name__ == '__main__':
    main()
