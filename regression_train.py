import time
import pdb
import  torch, os
import  numpy as np
from sinusoid import Sinusoid
from polynomial import Polynomial
from imagenet import ImageNet
from cornell_grasps import CornellGrasps
import  argparse

from    meta import Meta

def main(args):

    sinusoid = {'name':'sinusoid','class':Sinusoid, 'dims':[1,1]}
    polynomial = {'name':'polynomial', 'class':Polynomial, 'dims':[2,1]}
    imagenet = {'name':'imagenet', 'class':ImageNet, 'dims':[4096,2]}
    grasps = {'name':'grasps', 'class':CornellGrasps, 'dims':[4096,2]}
    data_params = {'sinusoid':sinusoid, 'polynomial':polynomial, 'imagenet':imagenet, 'grasps':grasps}
    func_data = data_params[args.func_type]

    save_path = os.getcwd() + '/data/' + func_data['name'] + '/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '_epoch'

    torch.cuda.synchronize()
    torch.manual_seed(222)
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(222)
    torch.cuda.synchronize()
    np.random.seed(222)

    print(args)

    dim_hidden = [40,40]
    dim_input, dim_output = func_data['dims']
    config = [
        ('fc', [dim_hidden[0], dim_input]),
        ('relu', [True])]#,
        #('bn', [dim_hidden[0]])]

    for i in range(1, len(dim_hidden)):
        config += [
            ('fc', [dim_hidden[i], dim_hidden[i-1]]),
            ('relu', [True])] #,
            #('bn', [dim_hidden[i]])]

    config += [
        ('fc', [dim_output, dim_hidden[-1]])] #,
        #('relu', [True])] #,
        #('bn', [dim_output])]

    #device = torch.device('cpu')
    device = torch.device('cuda')
    torch.cuda.synchronize()
    maml = Meta(args, config, func_data['dims']).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = func_data['class'](
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry)

    prelosses, postlosses = [], []
    # epoch: number of training batches
    for step in range(args.epoch):
        #start = time.time()
        batch_x, batch_y = db_train.next()
        inputa = batch_x[:, :args.k_spt, :]
        labela = batch_y[:, :args.k_spt, :]
        if args.k_spt == 1:
            inputa = np.array(inputa) #np.concatenate([inputa, inputa], axis=1)     
            labela = np.array(labela) #np.concatenate([labela, labela], axis=1)     
        labelb = batch_y[:,args.k_spt:, :]
        inputb = batch_x[:,args.k_spt:, :]

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                     torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)
        #accs, svm_loss = maml(x_spt, y_spt, x_qry, y_qry)
        #time_diff = time.time() - start
        #print(time_diff)
        if step % 100 == 0:
            prelosses.append(accs[0].item())
            postlosses.append(accs[-1].item())
            #svmlosses.append(svm_loss.item())

        if step % 1000 == 0:
            preloss = ('%.10f'%np.mean(prelosses)) #accs[0].item())
            postloss = ('%.10f'%np.mean(postlosses)) #accs[-1].item())
            #svmloss = ('%.10f'%np.mean(svmlosses)) #accs[-1].item())
            print('Step ' + str(step) + ': ' + preloss + ', ' + postloss)
            #print('SVM: ' + svmloss)
            #pdb.set_trace()
            prelosses, postlosses, svmlosses = [], [], []
            torch.save(maml.state_dict(), save_path + str(step) + ".pt")
    torch.save(maml.state_dict(), save_path + str(args.epoch) + ".pt")
    #torch.save(maml.svm_weights, save_path + str(args.epoch) + "-svm_weights.pt")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=70000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=25)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--func_type', type=str, help='function type', default="sinusoid")
    argparser.add_argument('--svm_lr', type=float, help='task-level inner update learning rate', default=0.001)
    

    args = argparser.parse_args()

    main(args)
