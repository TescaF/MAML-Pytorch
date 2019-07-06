import pdb
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    svm import SVM
from    learner import Learner
from    copy import deepcopy
import  itertools
import math

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, dims):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.svm_lr = args.svm_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config) #, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        #self.svm_weights = nn.Parameter(torch.zeros(dims[0]))
        #self.svm_optim = optim.Adam([self.svm_weights], lr=self.svm_lr)
        #self.svm_max_iter = 10000
        #self.svm_epsilon = 0.001

    def forward(self, x_spt, y_spt, x_qry, y_qry, param_dim, num_tuned_layers):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        if param_dim > 0:
            p_spt = x_spt[:,:,-param_dim:]
            p_qry = x_qry[:,:,-param_dim:]
            x_spt = x_spt[:,:,:-param_dim]
            x_qry = x_qry[:,:,:-param_dim]
        else:
            p_spt = task_num * [None]
            p_qry = task_num * [None]
        #svm_loss = 0.0
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_list = []

        ## Freeze all layers except the last one
        for p in self.net.parameters():
            p.requires_grad = False
        for i in range(num_tuned_layers):
            list(self.net.parameters())[(-1) * (i+1)].requires_grad = True
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True, param_tensor=p_spt[i])
            loss = F.mse_loss(logits, y_spt[i])
            grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.net.parameters())) #line 6 in alg
            fast_weights = list(self.net.parameters())[:(-1)*num_tuned_layers] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, self.net.parameters()))))
            # loss before first update
            if math.isnan(loss):
                pdb.set_trace()
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True, param_tensor=p_qry[i])
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[0] += loss_q
                losses_list.append(loss_q)

            # loss after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True, param_tensor=p_qry[i])
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True, param_tensor=p_spt[i])
                loss = F.mse_loss(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, fast_weights)) #line 6 in alg
                fast_weights = list(self.net.parameters())[:(-1)*num_tuned_layers] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, self.net.parameters()))))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True, param_tensor=p_qry[i])
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # optimize theta parameters
        for p in self.net.parameters():
            p.requires_grad = True
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()


        return np.array(losses_q) / task_num #, svm_loss_total
        #return accs

    def finetuning(self, x_spt, y_spt, x_qry, y_qry, param_dim, num_tuned_layers, new_env=False):
        net = deepcopy(self.net)
        if new_env:
            for p in net.parameters():
                p.requires_grad = True
            #for i in range(num_tuned_layers):
            #    list(net.parameters())[(-1) * (i+1)].requires_grad = False
            #stuck_layers = list(net.parameters())[(-1)*num_tuned_layers:]
        else:
            ## Freeze all layers except the last one
            for p in net.parameters():
                p.requires_grad = False
            for i in range(num_tuned_layers):
                list(net.parameters())[(-1) * (i+1)].requires_grad = True
            stuck_layers = list(net.parameters())[:(-1)*num_tuned_layers]

        task_num = x_spt.size(0)
        if param_dim > 0:
            p_spt = x_spt[:,-param_dim:]
            p_qry = x_qry[:,-param_dim:]
            x_spt = x_spt[:,:-param_dim]
            x_qry = x_qry[:,:-param_dim]
        else:
            p_spt = task_num * [None]
            p_qry = task_num * [None]
        losses = [0.0 for _ in range(self.update_step_test + 1)]

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, vars=net.parameters(), bn_training=True, param_tensor=p_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, net.parameters())) #line 6 in alg
        if new_env:
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        else:
            fast_weights = list(net.parameters())[:(-1)*num_tuned_layers] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, net.parameters()))))
        # loss before first update
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True, param_tensor=p_qry)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses[0] += loss_q

        # loss after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True, param_tensor=p_qry)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses[1] += loss_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True, param_tensor=p_spt)
            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, fast_weights)) #line 6 in alg
            if new_env:
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            else:
                fast_weights = list(net.parameters())[:(-1)*num_tuned_layers] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, fast_weights))))

            logits_q = net(x_qry, fast_weights, bn_training=True, param_tensor=p_qry)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses[k + 1] += loss_q
            del grad
            torch.cuda.empty_cache()

        del net
        torch.cuda.empty_cache()
        return np.array(losses)/x_qry.size(0), fast_weights


def main():
    pass


if __name__ == '__main__':
    main()
