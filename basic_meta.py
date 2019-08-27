import math
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

    def smooth_hinge_loss(self, y, l):
        loss = torch.cuda.FloatTensor([0.0])
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            if ty <= 0:
                loss += (0.5 - ty)
            elif 0 < ty <= 1:
                loss += (0.5 * ((1-ty)**2.0))
        return loss / y.shape[0] 

    def direct_train(self, x, y):
        logits = self.net(x, vars=self.net.parameters(), bn_training=True)
        loss = self.loss_fn(logits, y)
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()
        loss_it = loss.item()
        del loss
        return [loss_it], loss_it

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
                print((logits_q-y_qry[i])) #*torch.cuda.FloatTensor([640,480])) #,math.pi/2.0]))
            del s_weights,logits_q
            torch.cuda.empty_cache()

        ## Get total losses after all tasks
        total_loss =  losses_q / task_num

        ## Optimize parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()
        loss = total_loss.item()
        del total_loss, losses_q

        if len(x_spt.shape) > 4:
            accs = np.array(corrects) / (x_qry.shape[1] * task_num)
        else:
            accs = np.array(corrects) / task_num
        return accs, loss

    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
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
            grad_a = list(torch.autograd.grad(loss, fast_weights,allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    grad_a[g] = torch.zeros_like(fast_weights[g])
            fast_weights = list(map(lambda p: p[1] - (self.update_lr * p[0]), zip(grad_a, fast_weights)))
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

        del net
        accs = np.array(corrects) 
        return accs,fast_weights

def main():
    pass


if __name__ == '__main__':
    main()
