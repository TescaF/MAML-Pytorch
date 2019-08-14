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
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.al_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




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
        if w is None:
            w = self.net.parameters()
        sample_loss = self.samplewise_loss(x,y,w)
        ## Get AL labels
        inputs, labels = [], []
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                if j == k:
                    continue
                if sample_loss[j] < sample_loss[k]:
                    labels.append(1)
                else:
                    labels.append(0)
                inputs.append(x[j] - x[k])
        inputs = torch.stack(inputs)
        labels = torch.cuda.LongTensor(labels)
        return inputs, labels

    def samplewise_loss(self, x, y,w=None):
        ## Get training loss for each sample
        sample_loss = []
        for s in range(x.shape[0]):
            s_weights = w
            for k in range(self.update_step):
                logits = self.net(x[s].unsqueeze(0), vars=s_weights, bn_training=False)
                loss = F.cross_entropy(logits[:,:-self.an], y[s].unsqueeze(0))
                grad = torch.autograd.grad(loss, s_weights)
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                del loss
                del logits
                del grad
                torch.cuda.empty_cache()
            logits_q = self.net(x, s_weights, bn_training=False)[:,:-self.an]
            loss_q = F.cross_entropy(logits_q, y)
            sample_loss.append(loss_q.item())
            del logits_q
            del s_weights
            torch.cuda.empty_cache()
        return sample_loss

    def smooth_hinge_loss(self, y, l):
        s = F.sigmoid(y)
        c = nn.BCELoss()
        pdb.set_trace()
        loss = c(s, torch.clamp(l,0,1))
        loss = 0.0
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
        loss_a = 0
        loss_b = 0
        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        al_corrects = 0.0

        sample_loss, al_feats = [], []
        for i in range(task_num):
            ## Get samplewise loss
            total_q = 0
            for s in range(x_spt[i].shape[0]):
                s_weights = self.net.parameters()
                logits_q = self.net(x_qry[i], s_weights, bn_training=False)[:,:-self.an]
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    corrects[0] += torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                for k in range(self.update_step):
                    # Spt loss
                    logits = self.net(x_spt[i,s].unsqueeze(0), vars=s_weights, bn_training=False)[:,:-self.an]
                    loss = F.cross_entropy(logits, y_spt[i,s].unsqueeze(0))
                    grad = torch.autograd.grad(loss, s_weights)
                    s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                    del loss, logits, grad
                    torch.cuda.empty_cache()

                    # Qry loss
                    logits_q = self.net(x_qry[i], s_weights, bn_training=False)[:,:-self.an]
                    loss_q = F.cross_entropy(logits_q, y_qry[i])

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct
                ## Get standard loss
                loss_a += F.cross_entropy(logits_q, y_qry[i])
                
                sample_loss.append(loss_q.item())
                al_feats.append(self.net(x_spt[i,s].unsqueeze(0), vars=self.net.parameters(), bn_training=False,hook=16))
                total_q += loss_q
                del pred_q, loss_q, s_weights
                torch.cuda.empty_cache()

        ## Get AL labels
        inputs, labels = [], []
        for j in range(len(al_feats)):
            for k in range(len(al_feats)):
                if j == k:
                    continue
                if sample_loss[j] < sample_loss[k]:
                    labels.append(1)
                else:
                    labels.append(0)
                inputs.append(al_feats[j] - al_feats[k])
        al_inputs = torch.cat(inputs)
        al_labels = torch.cuda.LongTensor(labels)

        ## Get AL loss
        logits_b = self.net(al_inputs, vars=self.net.parameters(), bn_training=True, start_idx=16, start_bn=8)[:,-self.an:]
        loss_b = F.cross_entropy(logits_b, al_labels)
        pred_al = F.softmax(logits_b, dim=1).argmax(dim=1)
        al_corrects = torch.eq(pred_al, al_labels).sum().item()/al_labels.shape[0]

        ## Get meta-update loss
        loss = (loss_a / (task_num*x_spt.shape[1])) + loss_b 
        #print("total: %.2f    loss: %.2f    al: %.2f" %(loss, loss_a/task_num, loss_b))


        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()
        del loss_a, loss_b, loss
        del logits_q, logits_b, pred_al, al_inputs, al_labels
        torch.cuda.empty_cache()

        accs = np.array(corrects) / (querysz * task_num * x_spt.shape[1])
        al_accs = np.array(al_corrects) 
        return accs, al_accs

    def al_test(self, x, y):
        al_inputs, al_labels = self.al_dataset(x, y)
        logits = self.net(al_inputs, vars=self.net.parameters(), bn_training=False)[:,-self.an:]
        loss = F.cross_entropy(logits, al_labels).item()
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        corrects = torch.eq(al_labels, pred).sum().item()
        del al_inputs, logits, pred
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
