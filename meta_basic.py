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
    def __init__(self, device, args, base_config):
        super(Meta, self).__init__()

        self.device = device
        self.im_dir = '/u/tesca/data/cropped_depth/'
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.base_net = Learner(base_config)
        self.meta_optim = optim.Adam(self.base_net.parameters(), lr=self.meta_lr)
        self.mean = torch.Tensor([0.018, 0.813, 0.006, 0.751]).double()
        self.std = torch.Tensor([0.292, 0.501, 0.035, 0.170]).double()

    def get_normal_images(self, fnames):
        imgs = []
        for n in fnames:
            i = cv.imread(self.im_dir + n.split("_label")[0] + '_depth.png',-1)
            i_rz = cv.resize(i, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
            depth_im = np.float64(i_rz)
            zx = cv.Sobel(depth_im, cv.CV_64F, 1, 0, ksize=5)
            zy = cv.Sobel(depth_im, cv.CV_64F, 0, 1, ksize=5)
            normal = np.dstack((-zx, -zy, np.ones_like(depth_im)))
            l = np.linalg.norm(normal, axis=2)
            normal = normal / np.expand_dims(l,2)
            dat = np.concatenate([normal,np.expand_dims(depth_im/1000.0,2)],axis=2)
            dat_t = torch.from_numpy(dat)
            dat_t.sub_(self.mean).div_(self.std)
            dat_var = Variable(dat_t.transpose(0,2))
            imgs.append(dat_var)
        img_data = torch.stack(imgs).float().to(self.device)
        return img_data

    def adadelta_update_base(self, net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        test_losses = [0 for _ in range(update_step + 1)]
        train_losses = [0 for _ in range(update_step)]
        eps = 1e-6
        rho = self.update_lr

        with torch.no_grad():
            logits_q = net(data_in, vars=None)[spt_idx:qry_idx]
            loss_q = F.mse_loss(torch.sigmoid(logits_q).squeeze(), y_qry)
            test_losses[0] += loss_q.item()

        squares, deltas = [], []
        s_weights = [None for _ in range(len(net.parameters()))]
        for p in list(net.parameters()):
            squares.append(torch.zeros_like(p))
            deltas.append(torch.zeros_like(p))

        for k in range(update_step):
            logits_r = net(data_in, vars=(None if k==0 else s_weights))[:spt_idx]
            #loss_r = F.mse_loss(logits_r, y_spt)
            loss_r = F.mse_loss(torch.sigmoid(logits_r).squeeze(), y_spt)
            grad_r = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights)))
            train_losses[k] += loss_r.item()
        
            for p in range(len(grad_r)):
                grad = grad_r[p]
                squares[p].mul_(rho).addcmul_(1-rho, grad, grad)
                std = squares[p].add_(eps).sqrt_()
                curr_delta = deltas[p].add(eps).sqrt_().div_(std).mul_(grad)
                if k == 0:
                    s_weights[p] = list(net.parameters())[p] - curr_delta
                else:
                    s_weights[p] = s_weights[p] - curr_delta
                deltas[p].mul_(rho).addcmul_(1-rho, curr_delta, curr_delta)

            with torch.no_grad():
                logits_q = net(data_in, vars=s_weights)[spt_idx:qry_idx]
                loss_q = F.mse_loss(torch.sigmoid(logits_q).squeeze(), y_qry)
                test_losses[k+1] += loss_q.item()
        logits_q = net(data_in, vars=s_weights)[spt_idx:qry_idx]
        loss_q = F.mse_loss(torch.sigmoid(logits_q).squeeze(), y_qry)
        return loss_q, [test_losses, train_losses]

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num = len(x_spt) #.size(0)
        losses_q = 0
        test_losses = [0 for _ in range(self.update_step + 1)]
        ft_train_loss = [0 for _ in range(self.update_step)]
        pt_train_loss = [0 for _ in range(self.update_step)]
        idx_a = len(x_spt[0]) #.shape[1]
        idx_b = idx_a + len(x_qry[0]) #.shape[1]

        for i in range(task_num):
            full_set = x_spt[i] + x_qry[i] # + n_spt[i]  #[x_spt[i],x_qry[i]] #,n_spt[i]]
            all_input = self.get_normal_images(full_set) #self.resnet_fts(data_in).transpose(1,3)
            loss_q, losses = self.adadelta_update_base(self.base_net, self.update_step, all_input, y_spt[i], y_qry[i], idx_a, idx_b)
            test_losses = [test_losses[j] + losses[0][j] for j in range(len(test_losses))]
            ft_train_loss = [ft_train_loss[j] + losses[1][j] for j in range(len(ft_train_loss))]
            #pt_train_loss = [pt_train_loss[j] + losses[2][j] for j in range(len(pt_train_loss))]
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

    def tune(self, x_spt, y_spt, x_qry, y_qry):
        base_net = deepcopy(self.base_net)
        test_losses = [0 for _ in range(self.update_step_test + 1)]
        ft_train_loss = [0 for _ in range(self.update_step_test)]
        pt_train_loss = [0 for _ in range(self.update_step_test)]
        idx_a = len(x_spt) #.shape[1]
        idx_b = idx_a + len(x_qry) #.shape[1]
        full_set = x_spt + x_qry # + n_spt[i]  #[x_spt[i],x_qry[i]] #,n_spt[i]]
        all_input = self.get_normal_images(full_set) #self.resnet_fts(data_in).transpose(1,3)
        loss_q, losses = self.adadelta_update_base(base_net, self.update_step_test, all_input, y_spt, y_qry, idx_a, idx_b)
        test_losses = losses[0]
        ft_train_loss = losses[1]
        qry_logits = None #logits[idx_a:idx_b]
        return [np.array(test_losses), np.array(pt_train_loss), np.array(ft_train_loss)], qry_logits

def main():
    pass

if __name__ == '__main__':
    main()
