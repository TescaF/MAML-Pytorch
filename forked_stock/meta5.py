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
        self.an = config[-1][1][0] - self.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.train_al = args.train_al == 1
        self.alpha = args.alpha
        self.net = Learner(config, args.imgc, args.imgsz)
        al_config = [('linear', [self.an, config[-1][-1][-1]])]
        self.al_net = Learner(al_config, args.imgc, args.imgsz)
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

    def al_dataset(self, x, y, w=None):
        sample_loss = self.samplewise_loss(x,y,w)
        ## Get AL labels
        inputs, labels = [], []
        hooks = self.net(x, vars=w, bn_training=True,hook=16)
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                diff = sample_loss[j] - sample_loss[k]
                if diff == 0 or sample_loss[j] == -1 or sample_loss[k] == -1:
                    continue
                labels.append(np.sign(diff))
                inputs.append(hooks[j] - hooks[k])
        inputs = torch.stack(inputs)
        if self.an == 1:
            labels = torch.cuda.FloatTensor(labels)
        if self.an == 2:
            labels = torch.cuda.LongTensor(labels)
        return inputs, labels

    def samplewise_loss(self, x, y,w=None):
        ## Get training loss for each sample
        sample_loss = []
        for s in range(x.shape[0]):
            s_weights = w
            for k in range(self.update_step):
                xi = torch.stack([x[s], x[s]])
                logits = self.net(xi, vars=s_weights, bn_training=True)[0,:-self.an].unsqueeze(0)
                loss = F.cross_entropy(logits, y[s].unsqueeze(0))
                if np.isnan(loss.item()):
                    del logits, loss
                    continue #pdb.set_trace()
                if s_weights is None:
                    grad = torch.autograd.grad(loss, self.net.parameters())
                    s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
                else:
                    grad = torch.autograd.grad(loss, s_weights)
                    s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                del loss
                del logits
                del grad
                torch.cuda.empty_cache()
                logits_q = self.net(x, s_weights, bn_training=True)[:,:-self.an]
            loss_q = F.cross_entropy(logits_q, y)
            sample_loss.append(loss_q.item())
            del logits_q
            del s_weights
            torch.cuda.empty_cache()
        return sample_loss

    def smooth_hinge_loss(self, y, l):
        loss = 0.0
        if np.isnan(y[0].item()):
            print("Output is NaN")
            return -1
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            if ty <= 0:
                loss += (0.5 - ty)
            elif 0 < ty <= 1:
                loss += (0.5 * ((1-ty)**2.0))
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
        querysz = x_qry.size(1)
        losses_q, losses_al = 0, 0
        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        al_corrects = 0.0

        all_al_loss = 0
        for i in range(task_num):
            ## Train over all samples
            s_weights = self.net.parameters()
            al_weights = self.al_net.parameters() #deepcopy(list(self.net.parameters()))
            '''for p in al_weights:
                p.requires_grad = False
            list(al_weights)[-1].requires_grad = True
            list(al_weights)[-2].requires_grad = True'''

            logits_q = self.net(x_qry[i], None, bn_training=True)[:,:-self.an]
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                del pred_q
                corrects[0] = corrects[0] + correct
            al_inputs, al_labels = self.al_dataset(x_qry[i], y_qry[i])
            for k in range(self.update_step):
                # Model loss
                logits_a = self.net(x_spt[i], vars=s_weights, bn_training=True)[:,:-self.an]
                loss = F.cross_entropy(logits_a, y_spt[i])
                grad_a = torch.autograd.grad(loss, self.net.parameters())
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, self.net.parameters())))
                del loss, logits_a, grad_a

                # AL loss
                logits_b = self.al_net(al_inputs, vars=al_weights, bn_training=True)
                #logits_b = self.net(al_inputs, vars=al_weights, bn_training=True, start_idx=16, start_bn=8)[:,-self.an:]
                if self.an == 1:
                    loss_b = self.smooth_hinge_loss(logits_b, al_labels)
                if self.an == 2:
                    loss_b = F.cross_entropy(logits_b, al_labels)

                grad_b = torch.autograd.grad(loss_b, al_weights)
                al_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_b, al_weights)))

                del loss_b, logits_b, grad_b

                logits_q = self.net(x_qry[i], s_weights, bn_training=True)[:,:-self.an]

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    del pred_q
                    corrects[k + 1] = corrects[k + 1] + correct
            # Get final losses
            losses_q += F.cross_entropy(logits_q, y_qry[i])
            m_weights = deepcopy(list(self.net.parameters()))
            m_weights[-2][-1].data = al_weights[0].squeeze(0)
            m_weights[-1][-1].data = al_weights[1].squeeze(0)
            al_loss, al_acc = self.al_test(x_qry[i], y_qry[i], w=m_weights)
            losses_al += al_loss
            al_corrects += al_acc
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q / task_num
        loss_al = losses_al / task_num
        total_loss = loss_q + loss_al

        #print("total: %.2f    loss: %.2f    al: %.2f" %(total_loss, loss_q, loss_al))
        # optimize theta parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        #loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)
        al_accs = np.array(al_corrects) / task_num
        return accs, al_accs

    def al_test(self, x, y, w=None):
        if w is None:
            w = self.net.parameters()
        al_inputs, al_labels = self.al_dataset(x, y, self.net.parameters())
        logits = self.net(al_inputs, vars=w, bn_training=True, start_idx=16, start_bn=8)[:,-self.an:]
        if self.an == 1:
            loss = self.smooth_hinge_loss(logits, al_labels)
            pred = torch.sign(logits)
        if self.an == 2:
            loss = F.cross_entropy(logits, al_labels)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
        corrects = torch.eq(al_labels, pred.squeeze(1)).sum().item()
        return loss, corrects/al_labels.shape[0]

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
        logits = net(x_spt)[:,:-self.an]
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)[:,:-self.an]
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)[:,:-self.an]
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)[:,:-self.an]
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)[:,:-self.an]
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
