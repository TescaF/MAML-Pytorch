import pdb
import itertools
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    learner import Learner
from    copy import deepcopy
from torchviz import make_dot

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
        l = l.next_functions[0][0]

class AL_Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, task_mod):
        """

        :param args:
        """
        super(AL_Learner, self).__init__()

        self.meta_lr = args.meta_lr
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.task_net = task_mod
        self.net = Learner(config, args.imgc, args.imgsz)
        #self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr, momentum=0.9)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        #self.weights = list(self.net.parameters())

    def al_triples(self, x, y):
        sample_loss, all_losses = self.samplewise_loss(x,y)
        ## Get AL labels
        inputs, labels, loss_diffs = [], [], []
        with torch.no_grad():
            hooks = self.task_net.net(x, vars=self.task_net.net.parameters(), bn_training=True,hook=16)
        idxs = list(itertools.permutations(list(range(x.shape[0])),3))
        for p in idxs:
            diff = all_losses[p[1]][p[0]] - all_losses[p[2]][p[0]]
            if not diff == 0:
                labels.append(torch.clamp(torch.sign(diff),0,np.inf))
                inputs.append(torch.stack([hooks[p[0]],hooks[p[1]],hooks[p[2]]]))
        inputs = torch.stack(inputs)
        labels = torch.cuda.FloatTensor(labels)
        return inputs, labels

    def samplewise_loss(self, x, y):
        ## Get training loss for each sample
        sample_loss, all_losses = [], []
        for s in range(x.shape[0]):
            accs, w = self.task_net.finetunning(x[s].unsqueeze(0), y[s].unsqueeze(0), x, y)
            vals = self.task_net.net(x, w)
            losses = []
            for s in range(x.shape[0]):
                losses.append(self.task_net.loss_fn(vals[s].unsqueeze(0), y[s].unsqueeze(0)))
            sample_loss.append(accs[-1])
            all_losses.append(losses)
        return sample_loss, all_losses

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        al_corrects = [0.0, 0.0]

        dist = torch.nn.PairwiseDistance()
        crit = torch.nn.BCEWithLogitsLoss()
        for i in range(task_num):
            ## Train over all samples
            al_inputs, al_labels = self.al_triples(x_qry[i], y_qry[i])
            if al_inputs is None:
                return None, None
            weights = list(self.net.parameters()) #self.weights
            for _ in range(10):
                logits_a = []
                #weights = self.net.parameters() #self.weights
                for j in range(al_inputs.shape[0]):
                    la = self.net(al_inputs[j,0], vars=weights, bn_training=True)
                    lb = self.net(al_inputs[j,1], vars=weights, bn_training=True)
                    lc = self.net(al_inputs[j,2], vars=weights, bn_training=True)
                    d1 = dist(lb.unsqueeze(0), la.unsqueeze(0))
                    d2 = dist(lc.unsqueeze(0), la.unsqueeze(0))
                    #d1 = torch.norm(lb - la)
                    #d2 = torch.norm(lc - la)
                    logits_a.append(d1 - d2)
                logits_a = torch.stack(logits_a)
                #loss = self.smooth_hinge_loss(logits_a, al_labels) 
                loss = crit(logits_a.squeeze(1), al_labels)
                pred = (torch.sigmoid(logits_a) > 0.5).float().squeeze(1)
                c1 = torch.eq(al_labels, pred).sum().item()
            #c = torch.eq(al_labels, torch.sign(logits_a)).sum().item()
                al_corrects.append(c1/al_labels.shape[0])
                #al_corrects[0] += (c1/al_labels.shape[0])
                if loss > 0:
                    grad = torch.autograd.grad(loss, weights)
                    weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, weights)))
            pdb.set_trace()
            loss_a = loss.item()
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()

            
            logits_b = []
            with torch.no_grad():
                weights = self.net.parameters() #self.weights
                for j in range(al_inputs.shape[0]):
                    la = self.net(al_inputs[j,0], vars=weights, bn_training=True)
                    lb = self.net(al_inputs[j,1], vars=weights, bn_training=True)
                    lc = self.net(al_inputs[j,2], vars=weights, bn_training=True)
                    d1 = dist(lb.unsqueeze(0), la.unsqueeze(0))
                    d2 = dist(lc.unsqueeze(0), la.unsqueeze(0))
                    #d1 = torch.norm(lb - la)
                    #d2 = torch.norm(lc - la)
                    logits_b.append(d1 - d2)
                logits_b = torch.stack(logits_b)
                #loss2 = self.smooth_hinge_loss(logits_b, al_labels) 
                loss2 = crit(logits_b.squeeze(1), al_labels)
                #pred = F.sigmoid(logits_b) #, dim=1).argmax(dim=1)
                pred = (torch.sigmoid(logits_b) > 0.5).float().squeeze(1)
                c2 = torch.eq(al_labels, pred).sum().item()
                #c = torch.eq(al_labels, torch.sign(logits_b)).sum().item()
                loss_b = loss2.item()
                al_corrects[1] += (c2/al_labels.shape[0])
            del loss, logits_a, loss2, logits_b

        return np.array(al_corrects), loss_b
        al_accs = np.array(al_corrects) / task_num
        return al_accs, [loss_a,loss_b]

    def al_test(self, x, y):
        dist = torch.nn.PairwiseDistance()
        crit = torch.nn.BCEWithLogitsLoss()
        al_inputs, al_labels = self.al_triples(x, y)
        logits_b = []
        al_corrects = [0,0]
        with torch.no_grad():
            weights = self.net.parameters() #self.weights
            for j in range(al_inputs.shape[0]):
                la = self.net(al_inputs[j,0], vars=weights, bn_training=True)
                lb = self.net(al_inputs[j,1], vars=weights, bn_training=True)
                lc = self.net(al_inputs[j,2], vars=weights, bn_training=True)
                d1 = dist(lb.unsqueeze(0), la.unsqueeze(0))
                d2 = dist(lc.unsqueeze(0), la.unsqueeze(0))
                logits_b.append(d1 - d2)
            logits_b = torch.stack(logits_b)
            loss2 = crit(logits_b.squeeze(1), al_labels)
            pred = (torch.sigmoid(logits_b) > 0.5).float().squeeze(1)
            c2 = torch.eq(al_labels, pred).sum().item()
            loss_b = loss2.item()
            al_corrects[1] += (c2/al_labels.shape[0])
        del loss2, logits_b
        return np.array(al_corrects), loss_b

def main():
    pass


if __name__ == '__main__':
    main()
