import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2 as cv
import torchvision.models as models
import  pdb
import  numpy as np
import  math
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    learner import Learner
from    copy import deepcopy

class Meta(nn.Module):
    def __init__(self, device, args, base_config, class_config, reg_config):
        super(Meta, self).__init__()

        self.device = device
        self.im_dir = '/u/tesca/data/cropped/'
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.base_net = Learner(base_config)
        self.class_net = Learner(class_config)
        self.reg_net = Learner(reg_config)
        self.meta_optim = optim.Adam(list(self.base_net.parameters()) + list(self.reg_net.parameters()), lr=self.meta_lr)

    def resnet_fts(self, fnames):
        imgs = []
        for n in fnames:
            i = cv.imread(self.im_dir + n.split("_label")[0] + '_rgb.jpg',-1)
            img_in = Image.fromarray(np.uint8(i)*255)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            to_tensor = transforms.ToTensor()
            scaler = transforms.Resize((224, 224))
            img_var = Variable(normalize(to_tensor(scaler(img_in))).unsqueeze(0))
            imgs.append(img_var)
        model = models.resnet50(pretrained=True).to(self.device)
        layer = model._modules.get("layer4")[-1].bn2 
        data_in = torch.cat(imgs).to(self.device)
        embedding = torch.zeros(data_in.shape[0],512,7,7).cuda() #self.dim_input)

        def copy_data(m, i, o):
            embedding.copy_(o.data.squeeze())
        hook = layer.register_forward_hook(copy_data)
        model(data_in)
        hook.remove()
        return embedding

    def update(self, base_net, class_net, reg_net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        img_input = self.resnet_fts(data_in).transpose(1,3)
        reg_input = self.resnet_fts(data_in[:qry_idx]).transpose(1,3)

        class_losses, _, s_weights = self.adadelta_update(None, self.class_output, base_net, class_net, update_step, img_input, None, None, 0, 0)
        reg_losses, loss_q, s_weights = self.adadelta_update(s_weights, self.reg_output, base_net, reg_net, update_step, reg_input, y_spt, y_qry, spt_idx, qry_idx)
        
        del img_input, reg_input
        return loss_q, [reg_losses[0], class_losses[0], reg_losses[1]], s_weights

    def reg_output(self, weights, base_net, reg_net, data_in, data_out, start_idx, end_idx, grad=True):
        base_w = weights[:len(base_net.parameters())]
        reg_w = list(reg_net.parameters())
        all_w = base_w + reg_w
        base_out = base_net(data_in, vars=base_w)
        reg_out = reg_net(base_out, vars=reg_w)[start_idx:end_idx]

        loss = F.mse_loss(reg_out, data_out)
        if grad:
            grads = list(torch.autograd.grad(loss, all_w))
        else:
            grads = None
        return reg_out, loss, grads, all_w


    def class_output(self, weights, base_net, class_net, data_in, data_out, start_idx, end_idx, grad=True):
        if weights is None:
            base_w = list(base_net.parameters())
            class_w = list(class_net.parameters())
        else:
            base_w = weights[:len(base_net.parameters())]
            class_w = weights[len(base_net.parameters()):]
        all_w = base_w + class_w
        base_out = base_net(data_in, vars=base_w)
        class_out = class_net(base_out, vars=class_w)

        n = int(data_in.shape[0]/2)
        tgt_set = torch.cat([torch.zeros(n).long().cuda(), torch.ones(n).long().cuda()])
        loss = F.cross_entropy(class_out, tgt_set)
        if grad:
            grads = list(torch.autograd.grad(loss, all_w))
        else:
            grads = None
        return class_out, loss, grads, all_w

    def adadelta_update(self, init_w, output_fn, base_net, suf_net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        test_losses = [0 for _ in range(update_step + 1)]
        train_losses = [0 for _ in range(update_step)]
        eps = 1e-6
        rho = self.update_lr

        with torch.no_grad():
            logits_q, loss_q, _, init_w = output_fn(init_w, base_net, suf_net, data_in, y_qry, spt_idx, qry_idx, grad=False)
            test_losses[0] += loss_q.item()

        squares, deltas = [], []
        s_weights = [None for _ in range(len(init_w))]
        for p in init_w:
            squares.append(torch.zeros_like(p))
            deltas.append(torch.zeros_like(p))

        for k in range(update_step):
            logits_r, loss_r, grad_r, s_weights = output_fn((init_w if k==0 else s_weights), base_net, suf_net, data_in, y_spt, 0, spt_idx)
            train_losses[k] += loss_r.item()
        
            for p in range(len(grad_r)):
                grad = grad_r[p]
                squares[p].mul_(rho).addcmul_(1-rho, grad, grad)
                std = squares[p].add_(eps).sqrt_()
                curr_delta = deltas[p].add(eps).sqrt_().div_(std).mul_(grad)
                if k == 0:
                    s_weights[p] = init_w[p] - curr_delta
                else:
                    s_weights[p] = s_weights[p] - curr_delta
                deltas[p].mul_(rho).addcmul_(1-rho, curr_delta, curr_delta)

            with torch.no_grad():
                logits_q, loss_q, _, _ = output_fn(s_weights, base_net, suf_net, data_in, y_qry, spt_idx, qry_idx, grad=False)
                test_losses[k+1] += loss_q.item()

        _, loss_q, _, _ = output_fn(s_weights, base_net, suf_net, data_in, y_qry, spt_idx, qry_idx, grad=False)
        return [test_losses, train_losses], loss_q, s_weights

    def forward(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        task_num = len(x_spt) #.size(0)
        losses_q = 0
        test_losses = [0 for _ in range(self.update_step + 1)]
        ft_train_loss = [0 for _ in range(self.update_step)]
        pt_train_loss = [0 for _ in range(self.update_step)]
        idx_a = len(x_spt[0]) #.shape[1]
        idx_b = idx_a + len(x_qry[0]) #.shape[1]

        for i in range(task_num):
            full_set = x_spt[i] + x_qry[i] + n_spt[i]  #[x_spt[i],x_qry[i]] #,n_spt[i]]
            loss_q, losses, w = self.update(self.base_net, self.class_net, self.reg_net, self.update_step, full_set, y_spt[i], y_qry[i], idx_a, idx_b)
            test_losses = [test_losses[j] + losses[0][j] for j in range(len(test_losses))]
            ft_train_loss = [ft_train_loss[j] + losses[1][j] for j in range(len(ft_train_loss))]
            pt_train_loss = [pt_train_loss[j] + losses[2][j] for j in range(len(pt_train_loss))]
            losses_q += loss_q

        ## Get total losses after all tasks
        total_loss =  losses_q / task_num
        self.meta_optim.zero_grad()
        total_loss.backward()

        ## Clip gradient
        #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.meta_optim.step()
        loss = total_loss.item()

        return loss, [np.array(test_losses)/task_num, np.array(pt_train_loss)/task_num, np.array(ft_train_loss)/task_num]

    def tune(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        base_net = deepcopy(self.base_net)
        class_net = deepcopy(self.class_net)
        reg_net = deepcopy(self.reg_net)
        test_losses = [0 for _ in range(self.update_step_test + 1)]
        ft_train_loss = [0 for _ in range(self.update_step_test)]
        pt_train_loss = [0 for _ in range(self.update_step_test)]
        idx_a = len(x_spt) #.shape[1]
        idx_b = idx_a + len(x_qry) #.shape[1]
        full_set = x_spt + x_qry+ n_spt#[x_spt,x_qry] #,n_spt]

        loss_q, losses, w = self.update(base_net, class_net, reg_net, self.update_step, full_set, y_spt, y_qry, idx_a, idx_b)
        test_losses = losses[0]
        ft_train_loss = losses[1]
        pt_train_loss = losses[2]
        qry_logits = None #logits[idx_a:idx_b]
        return [np.array(test_losses), np.array(pt_train_loss), np.array(ft_train_loss)], qry_logits

def main():
    pass

if __name__ == '__main__':
    main()
