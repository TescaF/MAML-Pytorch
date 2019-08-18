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
from torch.utils.tensorboard import SummaryWriter

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
        self.w = SummaryWriter()
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
                labels.append(torch.sign(diff))
                inputs.append(torch.stack([hooks[p[0]],hooks[p[1]],hooks[p[2]]]))
        inputs = torch.stack(inputs)
        labels = torch.cuda.FloatTensor(labels)
        return inputs, labels

    def al_dataset(self, x, y, pair=False):
        sample_loss, all_losses = self.samplewise_loss(x,y)
        ## Get AL labels
        inputs, labels, loss_diffs = [], [], []
        with torch.no_grad():
            hooks = self.task_net.net(x, vars=self.task_net.net.parameters(), bn_training=True,hook=16)
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                loss_diff = sample_loss[j] - sample_loss[k]
                if loss_diff == 0:
                    continue
                if list(self.net.parameters())[-1].shape[0] == 1:
                    labels.append(np.sign(loss_diff))
                if list(self.net.parameters())[-1].shape[0] == 2:
                    labels.append(max(0,np.sign(loss_diff)))
                if pair:
                    inputs.append(torch.stack([hooks[j], hooks[k]]))
                else:
                    inputs.append(hooks[j] - hooks[k])
        if len(inputs) == 0:
            return None, None
        inputs = torch.stack(inputs)
        if list(self.net.parameters())[-1].shape[0] == 1:
            labels = torch.cuda.FloatTensor(labels)
        if list(self.net.parameters())[-1].shape[0] == 2:
            labels = torch.cuda.LongTensor(labels)
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

    def smooth_hinge_loss(self, y, l):
        loss_total = torch.cuda.FloatTensor([0.0])
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            loss = 0
            if ty <= 0:
                loss = (0.5 - ty)
            elif 0 < ty <= 1:
                loss = (0.5 * ((1-ty)**2.0))
            loss_total += loss
        return loss_total / y.shape[0] 
    

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

        for i in range(task_num):
            ## Train over all samples
            al_inputs, al_labels = self.al_triples(x_qry[i], y_qry[i])
            if al_inputs is None:
                return None, None
            logits_a = []
            for j in range(al_inputs.shape[0]):
                la = self.net(al_inputs[j,0], vars=self.net.parameters(), bn_training=True)
                lb = self.net(al_inputs[j,1], vars=self.net.parameters(), bn_training=True)
                lc = self.net(al_inputs[j,2], vars=self.net.parameters(), bn_training=True)
                d1 = torch.norm(lb - la)
                d2 = torch.norm(lc - la)
                logits_a.append(d1 - d2)
            logits_a = torch.stack(logits_a)
            loss = self.smooth_hinge_loss(logits_a, al_labels) 
            c = torch.eq(al_labels, torch.sign(logits_a)).sum().item()
            al_corrects[0] += (c/al_labels.shape[0])
            loss_a = loss.item()
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()
            #if loss > 0:
                #grad = torch.autograd.grad(loss, self.weights)
                #self.weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, self.weights)))

            logits_b = []
            for j in range(al_inputs.shape[0]):
                la = self.net(al_inputs[j,0], vars=self.net.parameters(), bn_training=True)
                lb = self.net(al_inputs[j,1], vars=self.net.parameters(), bn_training=True)
                lc = self.net(al_inputs[j,2], vars=self.net.parameters(), bn_training=True)
                d1 = torch.norm(lb - la)
                d2 = torch.norm(lc - la)
                logits_b.append(d1 - d2)
            logits_b = torch.stack(logits_b)
            loss2 = self.smooth_hinge_loss(logits_b, al_labels) 
            c = torch.eq(al_labels, torch.sign(logits_b)).sum().item()
            loss_b = loss2.item()
            al_corrects[1] += (c/al_labels.shape[0])
            del loss, logits_a, loss2, logits_b

        al_accs = np.array(al_corrects) / task_num
        return al_accs, [loss_a,loss_b]

    def forward1(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        al_corrects = [0.0, 0.0]

        for i in range(task_num):
            ## Train over all samples
            pair = True
            al_inputs, al_labels = self.al_dataset(x_qry[i], y_qry[i],pair=pair)
            if al_inputs is None:
                return None, None
            #al_inputs, al_labels = self.al_dataset(x_spt[i], y_spt[i])
            if pair:
                logits_a = []
                for j in range(al_inputs.shape[0]):
                    logits_data = self.net(al_inputs[j], vars=self.weights, bn_training=True)
                    logits_a.append(logits_data[0] - logits_data[1])
                logits_a = torch.stack(logits_a)
            else:
                logits_a = self.net(al_inputs, vars=self.weights, bn_training=True)
            #logits_a = self.net(al_inputs, vars=self.net.parameters(), bn_training=True)
            if list(self.net.parameters())[-1].shape[0] == 1:
                loss = self.smooth_hinge_loss(logits_a, al_labels) 
                c = torch.eq(al_labels, torch.sign(logits_a.squeeze(1))).sum().item()
            if list(self.net.parameters())[-1].shape[0] == 2:
                loss = F.cross_entropy(logits_a, al_labels) 
                pred_a = F.softmax(logits_a, dim=1).argmax(dim=1)
                c = torch.eq(al_labels, pred_a).sum().item()
            al_corrects[0] += (c/al_labels.shape[0])
            loss_a = loss.item()
            if loss > 0:
                #self.meta_optim.zero_grad()
                #loss.backward()
                #self.meta_optim.step()
                grad = torch.autograd.grad(loss, self.weights)
                self.weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, self.weights)))

            if pair:
                logits_b = []
                for j in range(al_inputs.shape[0]):
                    logits_data = self.net(al_inputs[j], vars=self.weights, bn_training=True)
                    logits_b.append(logits_data[0] - logits_data[1])
                logits_b = torch.stack(logits_b)
            else:
                logits_b = self.net(al_inputs, vars=self.weights, bn_training=True)
            #logits_b = self.net(al_inputs, vars=self.net.parameters(), bn_training=True)
            if list(self.net.parameters())[-1].shape[0] == 1:
                loss2 = self.smooth_hinge_loss(logits_b, al_labels) 
                c = torch.eq(al_labels, torch.sign(logits_b.squeeze(1))).sum().item()
            if list(self.net.parameters())[-1].shape[0] == 2:
                loss2 = F.cross_entropy(logits_b, al_labels) 
                pred_a = F.softmax(logits_b, dim=1).argmax(dim=1)
                c = torch.eq(al_labels, pred_a).sum().item()
            loss_b = loss2.item()
            al_corrects[1] += (c/al_labels.shape[0])
            del loss, logits_a

        al_accs = np.array(al_corrects) / task_num
        return al_accs, [loss_a,loss_b]

    def al_test(self, x, y):
        al_inputs, al_labels, loss_diffs = self.al_dataset(x, y)
        logits = self.net(al_inputs, vars=self.net.parameters(), bn_training=True)
        corrects = torch.eq(al_labels, torch.sign(logits.squeeze(1))).sum().item()
        return corrects/al_labels.shape[0]

def main():
    pass


if __name__ == '__main__':
    main()
