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
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = 0
        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        all_al_loss = 0
        for i in range(task_num):
            sample_loss = []
            for s in range(setsz):
                s_weights = None
                with torch.no_grad():
                    logits_q = self.net(x_qry[i], None, bn_training=True)[:,:-1]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    del pred_q
                    corrects[0] = corrects[0] + correct
                for k in range(self.update_step):
                    logits = self.net(torch.cat([x_spt[i,:s],x_spt[i,s+1:]]), vars=s_weights, bn_training=True)
                    loss = F.cross_entropy(logits[:,:-1], torch.cat([y_spt[i,:s],y_spt[i,s+1:]]))
                    if s_weights is None:
                        grad = torch.autograd.grad(loss, self.net.parameters())
                        s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
                    else:
                        grad = torch.autograd.grad(loss, s_weights)
                        s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                    del loss
                    del logits

                    logits_q = self.net(x_qry[i], s_weights, bn_training=True)[:,:-1]
                    # loss_q will be overwritten and just keep the loss_q on last update step.

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        del pred_q
                        corrects[k + 1] = corrects[k + 1] + correct
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                sample_loss.append(loss_q)
            comps = []
            for j in range(setsz):
                for k in range(setsz):
                    if j == k:
                        continue
                    if sample_loss[j] < sample_loss[k]:
                        al_label = 1
                    else:
                        al_label = -1
                    comps.append(al_label)
                    al_input = (x_spt[i,j] - x_spt[i,k]).unsqueeze(0)
                    al_logits = self.net(al_input, vars=None, bn_training=False)[:,-1]
                    al_loss = max(0, 1 - torch.mul(al_logits, al_label))
                    all_al_loss += al_loss
        la = torch.mean(torch.stack(sample_loss))
        lb = all_al_loss[0]/(setsz**2) #+ all_al_loss[0]
        losses_q = la + lb #torch.mean(torch.stack(sample_loss)) + all_al_loss[0]/(setsz**2) #+ all_al_loss[0]
        #print("sample loss: " + str(la.item()))
        #print("al loss: " + str(lb.item()))
        del sample_loss

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (setsz * querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)[:,:-1]
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)[:,:-1]
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)[:,:-1]
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)[:,:-1]
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)[:,:-1]
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs,fast_weights




def main():
    pass


if __name__ == '__main__':
    main()
