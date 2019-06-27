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

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        #svm_loss = 0.0
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_list = []

        ## Freeze all layers except the last one
        for p in self.net.parameters():
            p.requires_grad = False
        list(self.net.parameters())[-2].requires_grad = True
        list(self.net.parameters())[-1].requires_grad = True
        #iter_optim = optim.Adam(self.net.parameters(), lr=self.update_lr)

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            #logits = self.net(x_spt[i], vars=self.net.parameters(), bn_training=True)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_spt[i])
            grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.net.parameters())) #line 6 in alg
            fast_weights = list(self.net.parameters())[:-2] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, self.net.parameters()))))
            # loss before first update
            with torch.no_grad():
                #tuples, comps = self.relative_dists(logits)
                #x_tuples = self.tuple_matrices(tuples, x_spt[i])
                ##svm_x, svm_y = self.diff_pairs(x_spt[i], y_spt[i])
                ##svm = SVM(self.svm_max_iter, 1.0/x_spt[i].shape[0], self.svm_epsilon)
                ##print("Fitting SVM...")
                ##fast_svm_w = svm.fit(svm_x, svm_y) #x_tuples, comps)
                ##print("Done")

                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[0] += loss_q
                losses_list.append(loss_q)

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
                grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, fast_weights)) #line 6 in alg
                fast_weights = list(self.net.parameters())[:-2] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, filter(lambda p: p.requires_grad, fast_weights))))
                #grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                #loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
            #with torch.no_grad():
                #val_x, val_y = self.diff_pairs(x_qry[i], y_qry[i])
            '''svm_loss = F.cross_entropy(torch.mm(torch.transpose(fast_svm_w, 0, -1), torch.t(val_x)), val_y)
            pdb.set_trace()
            ## SVM update
            self.svm_optim.zero_grad()
            svm_loss.backward()
            self.svm_optim.step()        '''

        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        #pdb.set_trace()
        # optimize theta parameters
        for p in self.net.parameters():
            p.requires_grad = True
        self.meta_optim.zero_grad()
        loss_q.backward()
        #print('meta update')
        #for p in self.net.parameters()[:5]:
        #    print(torch.norm(p).item())
        self.meta_optim.step()


        return np.array(losses_q) / task_num #, svm_loss_total
        #return accs

    def diff_pairs(self, x, y):
        x_diffs, x_pairs, y_diffs, y_pairs = [], [], [], []
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                x_diffs.append(x[i] - x[j])
                y_diffs.append(y[i] - y[j])
        for i in range(len(x_diffs)):
            for j in range(len(x_diffs)):
                x_pairs.append((x_diffs[i] * x_diffs[i]) - (x_diffs[j] * x_diffs[j]))
                y_pairs.append(torch.norm(y_diffs[i] - y_diffs[j], 2).sign())
        return torch.stack(x_pairs), torch.stack(y_pairs)

    def tuple_matrices(self, tuples, x):
        # produce dX3 matrix for each tuple, where each row is a feature and each column is an instance
        m = []
        for t in tuples:
            i, j, g, h = t[0,0], t[0,1], t[0,2], t[0,3]
            m.append(torch.stack((x[i], x[j], x[g], x[h]), 1))
        return torch.stack(m,0)

    def relative_dists(self, y):
        # tuple: (i, j, g, h)
        # 1 if d(i,j) < d(g, h)
        # 0 otherwise
        tuples = list(itertools.permutations(range(y.shape[0]), 4))
        comps = []
        for (i, j, g, h) in tuples:
            #d1 = np.square(y[i] - y[j])
            d1 = torch.norm(y[i] - y[j], 2)
            d2 = torch.norm(y[g] - y[h], 2)
            #d2 = np.square(y[i] - y[k])
            comps.append(torch.sign(d2 - d1)) #np.sign(d2 - d1))
        return np.matrix(tuples), torch.stack(comps)

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
            losses[0] = loss

        # loss after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss = F.mse_loss(logits_q, y_qry)
            losses[1] = loss

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
            losses[k + 1] = loss_q

        del net

        #accs = np.array(losses) / querysz

        return losses #accs

def main():
    pass


if __name__ == '__main__':
    main()
