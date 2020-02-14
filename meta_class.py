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
    def __init__(self, device, args, base_config, class_config):
        super(Meta, self).__init__()

        self.device = device
        self.im_dir = '/u/tesca/data/cropped_depth/'
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.base_net = Learner(base_config)
        self.class_net = Learner(class_config)
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
            dat = normal #np.concatenate([normal,np.expand_dims(depth_im/1000.0,2)],axis=2)
            dat_t = torch.from_numpy(dat)
            dat_t.sub_(self.mean[:3]).div_(self.std[:3])
            dat_var = Variable(dat_t.transpose(0,2))
            imgs.append(dat_var)
        img_data = torch.stack(imgs).float().to(self.device)
        return img_data

    def get_depth_images(self, fnames):
        imgs = []
        mean = 751
        std = 170
        for n in fnames:
            i = cv.imread(self.im_dir + n.split("_label")[0] + '_depth.png',-1)
            i_rz = cv.resize(i, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
            img_in = torch.from_numpy(np.float64(i_rz))
            img_in.sub_(mean).div_(std)
            img_var = Variable(img_in.unsqueeze(0))
            imgs.append(img_var)
        return torch.stack(imgs).float().to(self.device)

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

    def update_old(self, base_net, class_net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        img_input = self.get_depth_images(data_in) #self.resnet_fts(data_in).transpose(1,3)
        pdb.set_trace()
        #reg_input = self.resnet_fts(data_in[:qry_idx]).transpose(1,3)

        class_losses, _, s_weights = self.adadelta_update(None, self.reg_output, base_net, None, update_step, img_input, None, None, 0, 0)
        reg_weights = s_weights[:-len(self.class_net.parameters())] + list(self.base_net.parameters())[-6:]
        reg_losses, loss_q, s_weights = self.adadelta_update(reg_weights, self.reg_output, base_net, None, update_step, reg_input, y_spt, y_qry, spt_idx, qry_idx)
        
        del img_input, reg_input
        return loss_q, [reg_losses[0], class_losses[0], reg_losses[1]], s_weights

    def reg_output(self, weights, base_net, sup_net, data_in, data_out, start_idx, end_idx, grad=True):
        if grad:
            with torch.no_grad():
                inter = base_net(data_in, vars=weights, hook=len(base_net.config)-4)
            reg_out = base_net(inter, vars=weights, start_idx=len(base_net.config)-4)[start_idx:end_idx]
            loss = F.mse_loss(reg_out, data_out)
            grads = list(torch.autograd.grad(loss, weights, allow_unused=True))
            for i in range(len(grads)):
                if grads[i] is None:
                    grads[i] = torch.zeros_like(weights[i])
        else:
            reg_out = base_net(data_in, vars=weights)[start_idx:end_idx]
            loss = F.mse_loss(reg_out, data_out)
            grads = None
        return reg_out, loss, grads, weights

    def class_output(self, weights, base_net, class_net, data_in, data_out, start_idx, end_idx, grad=True):
        if weights is None:
            all_w = list(base_net.parameters())[:-6] + list(class_net.parameters())
        else:
            all_w = weights
        cw = len(class_net.parameters())
        base_out = base_net(data_in, vars=all_w[:-cw], hook=len(base_net.config)-4)
        class_out = class_net(base_out, vars=all_w[-cw:])

        n = int(data_in.shape[0]/2)
        tgt_set = torch.cat([torch.zeros(n).long().cuda(), torch.ones(n).long().cuda()])
        loss = self.ce_loss(class_out, tgt_set)
        if grad:
            grads = list(torch.autograd.grad(loss, all_w))
        else:
            grads = None
        return class_out, loss, grads, all_w #[all_w[:len(base_w[0])] + base_w[1], class_w]

    def adadelta_update_base(self, net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        test_losses = [0 for _ in range(update_step + 1)]
        train_losses = [0 for _ in range(update_step)]
        eps = 1e-6
        rho = self.update_lr

        with torch.no_grad():
            logits_q = net(data_in, vars=None)[spt_idx:qry_idx]
            loss_q = F.mse_loss(logits_q, y_qry)
            test_losses[0] += loss_q.item()

        squares, deltas = [], []
        s_weights = [None for _ in range(len(net.parameters()))]
        for p in list(net.parameters()):
            squares.append(torch.zeros_like(p))
            deltas.append(torch.zeros_like(p))

        for k in range(update_step):
            logits_r = net(data_in, vars=(None if k==0 else s_weights))[:spt_idx]
            loss_r = F.mse_loss(logits_r, y_spt)
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
                loss_q = F.mse_loss(logits_q, y_qry)
                test_losses[k+1] += loss_q.item()
        logits_q = net(data_in, vars=s_weights)[spt_idx:qry_idx]
        loss_q = F.mse_loss(logits_q, y_qry)
        return loss_q, [test_losses, train_losses]

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

    def update(self, net, update_step, data_in, y_spt, y_qry, spt_idx, qry_idx):
        test_losses = [0 for _ in range(update_step + 1)]
        pt_train_losses = [0 for _ in range(update_step)]
        all_input = self.get_depth_images(data_in) #self.resnet_fts(data_in).transpose(1,3)
        #x_spt, x_qry, n_spt = data_in

        with torch.no_grad():
            logits_q = net(all_input, vars=None)[spt_idx:qry_idx]
            loss_q = F.mse_loss(logits_q, y_qry)
            test_losses[0] += loss_q.item()

        for k in range(update_step):
            logits_r = net(all_input, vars=(None if k==0 else s_weights))[:spt_idx]
            loss_r = F.mse_loss(logits_r, y_spt)
            pt_train_losses[k] += loss_r.item()
            grad = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights)))
            s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

            with torch.no_grad():
                logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx]
                loss_q = F.mse_loss(logits_q, y_qry)
                test_losses[k+1] += loss_q.item()

        logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx]
        loss_q = F.mse_loss(logits_q, y_qry)
        pdb.set_trace()
        return loss_q, [test_losses, pt_train_losses]

    def forward(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        task_num = len(x_spt) #.size(0)
        losses_q = 0
        test_losses = [0 for _ in range(self.update_step + 1)]
        ft_train_loss = [0 for _ in range(self.update_step)]
        pt_train_loss = [0 for _ in range(self.update_step)]
        idx_a = len(x_spt[0]) #.shape[1]
        idx_b = idx_a + len(x_qry[0]) #.shape[1]

        for i in range(task_num):
            full_set = x_spt[i] + x_qry[i] # + n_spt[i]  #[x_spt[i],x_qry[i]] #,n_spt[i]]
            all_input = self.get_depth_images(full_set) #self.resnet_fts(data_in).transpose(1,3)
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

    def tune(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        base_net = deepcopy(self.base_net)
        class_net = deepcopy(self.class_net)
        test_losses = [0 for _ in range(self.update_step_test + 1)]
        ft_train_loss = [0 for _ in range(self.update_step_test)]
        pt_train_loss = [0 for _ in range(self.update_step_test)]
        idx_a = len(x_spt) #.shape[1]
        idx_b = idx_a + len(x_qry) #.shape[1]
        #full_set = x_spt + x_qry+ n_spt#[x_spt,x_qry] #,n_spt]

        #loss_q, losses = self.update(base_net, self.update_step, full_set, y_spt, y_qry, idx_a, idx_b)
        full_set = x_spt + x_qry # + n_spt[i]  #[x_spt[i],x_qry[i]] #,n_spt[i]]
        all_input = self.get_depth_images(full_set) #self.resnet_fts(data_in).transpose(1,3)
        loss_q, losses = self.adadelta_update_base(base_net, self.update_step, all_input, y_spt, y_qry, idx_a, idx_b)
        test_losses = losses[0]
        ft_train_loss = losses[1]
        #pt_train_loss = losses[2]
        qry_logits = None #logits[idx_a:idx_b]
        return [np.array(test_losses), np.array(pt_train_loss), np.array(ft_train_loss)], qry_logits

def main():
    pass

if __name__ == '__main__':
    main()
