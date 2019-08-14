import pdb
import  torch
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
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def al_dataset(self, x, y):
        sample_loss = self.samplewise_loss(x,y)
        ## Get AL labels
        inputs, labels, loss_diffs = [], [], []
        hooks = self.task_net.net(x, vars=self.task_net.net.parameters(), bn_training=False,hook=16)
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                loss_diff = sample_loss[j] - sample_loss[k]
                if loss_diff == 0:
                    continue
                labels.append(torch.sign(loss_diff))
                inputs.append(hooks[j] - hooks[k])
                loss_diffs.append(loss_diff)
        inputs = torch.stack(inputs)
        loss_diffs = torch.stack(loss_diffs)
        labels = torch.cuda.FloatTensor(labels)
        return inputs, labels, loss_diffs

    def samplewise_loss(self, x, y):
        ## Get training loss for each sample
        sample_loss = []
        for s in range(x.shape[0]):
            accs, _, loss_q = self.task_net.finetunning(x[s].unsqueeze(0), y[s].unsqueeze(0), x, y)
            sample_loss.append(loss_q) #accs[-1])
        return sample_loss

    def smooth_hinge_loss(self, y, l, diffs=None):
        loss_total = 0.0
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            loss = 0
            if ty <= 0:
                loss = (0.5 - ty)
            elif 0 < ty <= 1:
                loss = (0.5 * ((1-ty)**2.0))
            if diffs is None:
                loss_total += loss
            else:
                loss_total += (loss * abs(diffs[i]))
        return loss_total / y.shape[0] 
    
    def diff_hinge_loss(self, y, l, diffs):
        loss = 0.0
        for i in range(y.shape[0]):
            if y[i] * l[i] < 0:
                loss += abs(diffs[i])
        return loss / y.shape[0]

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
            al_inputs, al_labels, loss_diffs = self.al_dataset(x_qry[i], y_qry[i])
            #al_inputs, al_labels = self.al_dataset(x_spt[i], y_spt[i])
            logits_a = self.net(al_inputs, vars=self.net.parameters(), bn_training=True)
            c = torch.eq(al_labels, torch.sign(logits_a.squeeze(1))).sum().item()
            al_corrects[0] += (c/al_labels.shape[0])
            #loss = self.diff_hinge_loss(logits_a, al_labels, loss_diffs) 
            loss = self.smooth_hinge_loss(logits_a, al_labels) 
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()
            logits_b = self.net(al_inputs, vars=self.net.parameters(), bn_training=True)
            c = torch.eq(al_labels, torch.sign(logits_b.squeeze(1))).sum().item()
            al_corrects[1] += (c/al_labels.shape[0])
            del loss, logits_a

        al_accs = np.array(al_corrects) / task_num
        return al_accs

    def al_test(self, x, y):
        al_inputs, al_labels, loss_diffs = self.al_dataset(x, y)
        logits = self.net(al_inputs, vars=self.net.parameters(), bn_training=False)
        corrects = torch.eq(al_labels, torch.sign(logits.squeeze(1))).sum().item()
        return corrects/al_labels.shape[0]

def main():
    pass


if __name__ == '__main__':
    main()
