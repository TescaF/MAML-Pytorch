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
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.alpha = args.alpha
        self.net = Learner(config)
        self.an = al_sz
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.loss_fn = loss_fn
        self.accs_fn = accs_fn

    def al_dataset(self, x, y, w, grad=True):
        sample_loss = self.samplewise_loss(x,y,w)
        if grad:
            return self.al_dataset_inner(sample_loss,x,y,w)
        else:
            with torch.no_grad():
                return self.al_dataset_inner(sample_loss,x,y,w)

    def al_dataset_inner(self, sample_loss, x, y, w=None):
        inputs, labels = [], []
        ## Get model data prior to final layer
        #hooks = self.net(x, vars=w, bn_training=True,hook=len(self.net.config)-1)

        ## Get AL labels for each sample pair
        for j in range(x.shape[0]):
            for k in range(x.shape[0]):
                diff = sample_loss[j] - sample_loss[k]
                if diff == 0:
                    continue
                labels.append(np.sign(diff))
                #inputs.append(torch.stack([x[j], x[k]]))
                inputs.append([j, k])
                #inputs.append(torch.stack([hooks[j], hooks[k]]))
        #inputs = torch.stack(inputs)
        #del hooks
        if self.an == 2:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.cuda.FloatTensor(labels)
        return inputs, labels

    def samplewise_loss(self, x, y, w):
        sample_loss = []

        ## Get training loss for each sample
        for s in range(x.shape[0]):
            s_weights = deepcopy(list(w)) 

            ## Tune model on current sample
            for k in range(self.update_step):
                xi = torch.stack([x[s], x[s]])
                if self.an > 0:
                    logits = self.net(xi, vars=s_weights, bn_training=True)[0,:-self.an].unsqueeze(0)
                else:
                    logits = self.net(xi, vars=s_weights, bn_training=True)[0,:].unsqueeze(0)
                loss = self.loss_fn(logits, y[s].unsqueeze(0))
                grad = torch.autograd.grad(loss, s_weights)
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))
                del loss,logits,grad

            ## Get post-training loss on all samples
            logits_q = self.net(x, s_weights, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            loss_q = self.loss_fn(logits_q, y)
            sample_loss.append(loss_q.item())
            del loss_q,logits_q,s_weights

        return sample_loss

    def smooth_hinge_loss(self, y, l):
        loss = torch.cuda.FloatTensor([0.0])
        for i in range(y.shape[0]):
            ty = y[i] * l[i]
            if ty <= 0:
                loss += (0.5 - ty)
            elif 0 < ty <= 1:
                loss += (0.5 * ((1-ty)**2.0))
        return loss / y.shape[0] 

    def forward(self, x_spt, y_spt, x_qry, y_qry):
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

                # Update step
                logits_a = self.net(x_spt[i], vars=s_weights, bn_training=True)
                if self.an > 0:
                    logits_a = logits_a[:,:-self.an]
                loss = self.loss_fn(logits_a, y_spt[i])
                grad_a = torch.autograd.grad(loss, s_weights)
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
            del s_weights,logits_q
            torch.cuda.empty_cache()
            al_loss1, _ = self.al_test(x_spt[i], y_spt[i], w=self.net.parameters())
            al_loss2, al_acc = self.al_test(x_qry[i], y_qry[i], w=self.net.parameters())
            losses_al += (al_loss1 + al_loss2)
            al_corrects += al_acc

        ## Get total losses after all tasks
        loss_q = self.alpha * losses_q / task_num
        loss_al = losses_al / task_num
        total_loss = loss_q + loss_al
        #print("total: %.2f    loss: %.2f    al: %.2f" %(total_loss, loss_q, loss_al))

        ## Optimize parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()
        loss_results = [loss_q.item(), loss_al.item(), total_loss.item()]
        del loss_q,losses_q,al_loss1,al_loss2,loss_al,total_loss

        if len(x_spt.shape) > 4:
            accs = np.array(corrects) / (x_qry.shape[1] * task_num)
        else:
            accs = np.array(corrects) / task_num
        al_accs = np.array(al_corrects) / task_num
        return accs, al_accs, loss_results

    def al_test(self, x, y, w):
        ## Get sample pairs and comparison labels
        al_inputs, al_labels = self.al_dataset(x, y, self.net.parameters(), grad=True)
        logits = []

        ## Get AL output for each sample pair
        #for j in range(al_inputs.shape[0]):
        for j in range(len(al_inputs)):
            pair = torch.stack([x[al_inputs[j][0]],x[al_inputs[j][1]]])
            logits_data = self.net(pair, vars=w, bn_training=True) #, last_layer=True)
            if self.an > 0:
                logits_data = logits_data[:,-self.an:]
            del pair
            #logits_data = self.net(torch.stack(al_inputs[j], vars=w, bn_training=True, last_layer=True)
            logits.append(logits_data[0] - logits_data[1])

        ## Compare AL output to comparison labels
        logits = torch.stack(logits)
        if self.an == 2:
            loss = F.cross_entropy(logits, al_labels)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
        else:
            loss = self.smooth_hinge_loss(logits, al_labels)
            pred = torch.sign(logits)
        corrects = torch.eq(al_labels, pred.squeeze(1)).sum().item()
        del logits
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

        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        if self.an > 0:
            logits = logits[:,:-self.an]
        loss = self.loss_fn(logits, y_spt)
        #loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            # [setsz]
            # scalar
            if self.accs_fn is None:
                correct = self.loss_fn(logits_q, y_qry).item()
            else:
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = self.accs_fn(pred_q, y_qry).sum().item()
            #correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            # [setsz]
            # scalar
            if self.accs_fn is None:
                correct = self.loss_fn(logits_q, y_qry).item()
            else:
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = self.accs_fn(pred_q, y_qry).sum().item()
            #correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            if self.an > 0:
                logits = logits[:,:-self.an]
            loss = self.loss_fn(logits, y_spt)
            #loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            if self.an > 0:
                logits_q = logits_q[:,:-self.an]
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = self.loss_fn(logits_q, y_qry)
            #loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                if self.accs_fn is None:
                    correct = loss_q.item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry).sum().item()  # convert to numpy
                #correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs,fast_weights




def main():
    pass


if __name__ == '__main__':
    main()
