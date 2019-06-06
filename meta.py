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



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config) #, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(1)
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            #logits = self.net(x_spt[i], vars=self.net.parameters(), bn_training=True)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters()) #line 6 in alg
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # loss before first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[0] += loss_q

            # loss after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                #loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        #pdb.set_trace()
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        #print('meta update')
        #for p in self.net.parameters()[:5]:
        #    print(torch.norm(p).item())
        self.meta_optim.step()

        return np.array(losses_q) / task_num
        #return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = x_qry.size(0)

        losses = [0.0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, net.parameters())
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # loss before first update
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss = F.mse_loss(logits_q, y_qry)
            losses[0] += loss

        # loss after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss = F.mse_loss(logits_q, y_qry)
            losses[1] += loss

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)
            losses[k + 1] += loss_q

        del net

        accs = np.array(losses) / querysz

        return accs

def main():
    pass


if __name__ == '__main__':
    main()
