import time
import pdb
import  torch, os
import  numpy as np
#from sinusoid import Sinusoid
#from polynomial import Polynomial
#from imagenet import ImageNet
#from cornell_grasps import CornellGrasps
from categorized_grasps import CategorizedGrasps
from leave_out_grasps import LeaveoutGrasps
from affordances import Affordances, Affordances2D, Affordances2DTT
import  argparse

from    meta import Meta
#from ab_meta import Meta

def main(args):

    #sinusoid = {'name':'sinusoid','class':Sinusoid, 'dims':[1,1,0]}
    #polynomial = {'name':'polynomial', 'class':Polynomial, 'dims':[2,1,0]}
    #imagenet = {'name':'imagenet', 'class':ImageNet, 'dims':[4096,2,0]}
    #grasps = {'name':'grasps', 'class':CornellGrasps, 'dims':[4096,2,0]}
    cat_grasps = {'name':'cat_grasps', 'class':CategorizedGrasps, 'dims':[4096,2,1]} #third number is param length
    leaveout_grasps = {'name':'leaveout_grasps', 'class':LeaveoutGrasps, 'dims':[4096,2,1]} #third number is param length
    affordances = {'name':'affordances', 'class':Affordances, 'dims':[4096,3,0]} #third number is param length
    affordances_2d = {'name':'affordances_2d', 'class':Affordances2D, 'dims':[4096,2,0]} #third number is param length
    affordances_tt = {'name':'affordances_tt', 'class':Affordances2DTT, 'dims':[4096,2,2]} #third number is param length
    data_params = {'affordances':affordances, 'cat_grasps':cat_grasps, 'affordances_2d':affordances_2d, 'affordances_tt':affordances_tt, 'leaveout_grasps':leaveout_grasps}
    func_data = data_params[args.func_type]

    if args.leave_out >= 0:
        split_txt = ''
        split_num = args.leave_out
        if args.split_cat == 1:
            dir_name = "lo_" + str(args.leave_out) + "/"
        else:
            dir_name = "cat_" + str(args.leave_out) + "/"
    else:
        split_txt = 'split' + str(args.split)
        split_num = args.split
        dir_name = ""

    if args.split_cat == 1:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + split_txt + '-cat_epoch'
    else:
        save_path = os.getcwd() + '/data/' + func_data['name'] + '/' + dir_name + 'model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + split_txt + '-obj_epoch'

    torch.cuda.synchronize()
    torch.manual_seed(222)
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(222)
    torch.cuda.synchronize()
    np.random.seed(222)

    print(args)

    if args.func_type == "cat_grasps":
        dim_hidden = [4096,[512,513], 128]
    if args.func_type == "leaveout_grasps":
        dim_hidden = [4096,[512,513], 128]
    if args.func_type == "affordances":
        dim_hidden = [4096,512, 128]
    if args.func_type == "affordances_2d":
        dim_hidden = [4096,512, 128]
    if args.func_type == "affordances_tt":
        dim_hidden = [4096,[512,514], 128]

    #dim_hidden = [4096,500]
    #dim_hidden = [40,40]
    dims = func_data['dims']

    '''
    config = [
        ('fc', [dim_hidden[0], dim_input]),
        ('relu', [True])]#,
        #('bn', [dim_hidden[0]])]

    for i in range(1, len(dim_hidden)):
        config += [
            ('fc', [dim_hidden[i], dim_hidden[i-1]]),
            ('relu', [True])] #,, 'cat':cat_grasps
            #('bn', [dim_hidden[i]])]

    config += [
        ('fc', [dim_output, dim_hidden[-1]])] #,
        #('relu', [True])] #,
        #('bn', [dim_output])]
    '''


    config = [
        ('linear', [dim_hidden[0], dims[0]]),
        ('relu', [True]),
        ('bn', [dim_hidden[0]])]
    prev_dim = dim_hidden[0]
    for i in range(1, len(dim_hidden)):
        if type(dim_hidden[i]) == list:
            curr_dim = dim_hidden[i][0]
        else:
            curr_dim = dim_hidden[i]
        config += [
            ('linear', [curr_dim, prev_dim]),
            ('relu', [True]),
            ('bn', [curr_dim])]
        if type(dim_hidden[i]) == list:
            prev_dim = dim_hidden[i][1]
        else:
            prev_dim = curr_dim

    config += [
        ('linear', [dims[1], prev_dim])] 



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
                       k_qry=args.k_qry,
                       num_grasps=args.grasps,
                       split=split_num,
                       train=True,
                       split_cat=args.split_cat)

    prelosses, postlosses = [], []
    # epoch: number of training batches
    for step in range(args.epoch):
        #start = time.time()
        batch_x, batch_y = db_train.next()
        inputa = batch_x[:, :args.k_spt*args.grasps, :]
        labela = batch_y[:, :args.k_spt*args.grasps, :]
        if args.k_spt == 1:
            inputa = np.array(inputa) #np.concatenate([inputa, inputa], axis=1)     
            labela = np.array(labela) #np.concatenate([labela, labela], axis=1)     
        labelb = batch_y[:,args.k_spt*args.grasps:, :]
        inputb = batch_x[:,args.k_spt*args.grasps:, :]

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                     torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry, dims[2], args.tuned_layers)
        #accs, svm_loss = maml(x_spt, y_spt, x_qry, y_qry)
        #time_diff = time.time() - start
        #print(time_diff)
        if step % 1 == 0:
            prelosses.append(accs[0].item())
            postlosses.append(accs[-1].item())
            #svmlosses.append(svm_loss.item())

        if step % 1000 == 0:
            preloss = '{:.3e}'.format(np.mean(prelosses)) #accs[0].item())
            postloss = '{:.3e}'.format(np.mean(postlosses)) #accs[-1].item())
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
    argparser.add_argument('--grasps', type=int, help='number of grasps per object sample', default=1)
    argparser.add_argument('--tuned_layers', type=int, help='number of grasps per object sample', default=2)
    argparser.add_argument('--split', type=float, help='training/testing data split', default=0.5)
    argparser.add_argument('--leave_out', type=int, help='affordance number to leave out during training (2-6)', default=-1)
    argparser.add_argument('--split_cat', type=int, help='1 if training/testing data is split by category, 0 if split by object id', default=1)
 

    args = argparser.parse_args()

    main(args)
