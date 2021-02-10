import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import pickle

from tools import * 
from get_data import *
from models import *

import timeit

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='MNIST Example')
#
parser.add_argument('--name', type=str, default='mnist', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=35, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
#
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay value (default: 0.1)')
#
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[30], help='decrease learning rate at these epochs.')
#
parser.add_argument('--wd', default=0.0, type=float, metavar='W', help='weight decay (default: 0.0)')
#
parser.add_argument('--gamma_W', default=0.001, type=float, metavar='W', help='diffiusion rate for W')
#
parser.add_argument('--gamma_A', default=0.001, type=float, metavar='W', help='diffiusion rate for A')
#
parser.add_argument('--beta', default=0.75, type=float, metavar='W', help='skew level')
#
parser.add_argument('--model', type=str, default='NoisyRNN', metavar='N', help='model name')
#
parser.add_argument('--solver', type=str, default='noisy', metavar='N', help='model name')
#
parser.add_argument('--n_units', type=int, default=64, metavar='S', help='number of hidden units')
#
parser.add_argument('--eps', default=0.1, type=float, metavar='W', help='time step for euler scheme')
#
parser.add_argument('--T', default=49, type=int, metavar='W', help='time steps')
#
parser.add_argument('--init_std', type=float, default=0.1, metavar='S', help='control of std for initilization')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')
#
parser.add_argument('--gclip', type=int, default=0, metavar='S', help='gradient clipping')
#
parser.add_argument('--optimizer', type=str, default='Adam', metavar='N', help='optimizer')
#
parser.add_argument('--alpha', type=float, default=1, metavar='S', help='for ablation study')
#
parser.add_argument('--add_noise', type=float, default=0.0, metavar='S', help='level of additive noise')
#
parser.add_argument('--mult_noise', type=float, default=0.0, metavar='S', help='level of multiplicative noise')
#
args = parser.parse_args()

if not os.path.isdir(args.name + '_results'):
    os.mkdir(args.name + '_results')

#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#==============================================================================
# get device
#==============================================================================
device = get_device()

#==============================================================================
# get dataset
#==============================================================================
if args.name == 'mnist':
    train_loader, test_loader, val_loader = getData(name='mnist', train_bs=args.batch_size, test_bs=args.test_batch_size)  
    model = NoisyRNN(input_dim=int(784/args.T), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma_A=args.gamma_A, gamma_W=args.gamma_W,
                 init_std=args.init_std, alpha=args.alpha, solver=args.solver, 
                 add_noise=args.add_noise, mult_noise=args.mult_noise).to(device)
            
elif args.name == 'pmnist':
    train_loader, test_loader, val_loader = getData(name='pmnist', train_bs=args.batch_size, test_bs=args.test_batch_size)  
    model = NoisyRNN(input_dim=int(784/args.T), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma_A=args.gamma_A, gamma_W=args.gamma_W,
                 init_std=args.init_std, alpha=args.alpha,  solver=args.solver, 
                 add_noise=args.add_noise, mult_noise=args.mult_noise).to(device)
    
elif args.name == 'cifar10':    
    train_loader, test_loader, val_loader = getData(name='cifar10', train_bs=args.batch_size, test_bs=args.test_batch_size)          
    model = NoisyRNN(input_dim=int(1024/args.T*3), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma_A=args.gamma_A, gamma_W=args.gamma_W,
                 init_std=args.init_std, alpha=args.alpha,  solver=args.solver, 
                 add_noise=args.add_noise, mult_noise=args.mult_noise).to(device)

elif args.name == 'cifar10_noise':  
    train_loader, test_loader, val_loader = getData(name='cifar10', train_bs=args.batch_size, test_bs=args.test_batch_size)              
    model = NoisyRNN(input_dim=int(96), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma_A=args.gamma_A, gamma_W=args.gamma_W,
                 init_std=args.init_std, alpha=args.alpha,  solver=args.solver, 
                 add_noise=args.add_noise, mult_noise=args.mult_noise).to(device)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)        
    noise = torch.randn(1,968,32,3).float()


#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#==============================================================================
# Model summary
#==============================================================================
print(model)    
print('**** Setup ****')
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')    
   

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
elif  args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    print("Unexpected optimizer!")
    raise 


loss_func = nn.CrossEntropyLoss().to(device)

# training and testing
count = 0
loss_hist = []
test_acc = []

t0 = timeit.default_timer()
for epoch in range(args.epochs):
    model.train()
    lossaccum = 0
    
    for step, (x, y) in enumerate(train_loader):
        count += 1
        
        # Reshape data for recurrent unit
        if args.name == 'mnist' or args.name == 'pmnist':
            inputs = Variable(x.view(-1, args.T, int(784/args.T))).to(device) # reshape x to (batch, time_step, input_size)
            targets = Variable(y).to(device)
            
        elif args.name == 'cifar10':            
            inputs = Variable(x.view(-1, args.T, int(1024/args.T*3))).to(device) # reshape x to (batch, time_step, input_size)
            targets = Variable(y).to(device)   

        elif args.name == 'cifar10_noise':
            x = x.view(-1, 32, int(96))
            x = torch.cat((x, noise.repeat(x.shape[0],1,1,1).view(-1, 968, int(96))), 1) # reshape x to (batch, time_step, input_size)
            inputs = Variable(x).to(device)             
            targets = Variable(y).to(device)   

                 
        # send data to recurrent unit    
        output = model(inputs, mode='train')   
        loss = loss_func(output, targets)
        
        
        optimizer.zero_grad()
        loss.backward()          
        
        if args.gclip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gclip) # gradient clip
            
        optimizer.step() # update weights
        lossaccum += loss.item()

        if args.model == 'test':
            D = model.W.weight.data.cpu().numpy()  
            u, s, v = np.linalg.svd(D, 0)
            model.W.weight.data = torch.from_numpy(u.dot(v)).float().cuda()

    loss_hist.append(lossaccum)    
     
    if epoch % 1 == 0:
        model.eval()
        correct = 0
        total_num = 0
        for data, target in test_loader:
            
            if args.name == 'mnist' or args.name == 'pmnist':
                data, target = data.to(device), target.to(device)                
                output = model(data.view(-1, args.T, int(784/args.T)))
            
            elif args.name == 'cifar10': 
                data, target = data.to(device), target.to(device)                
                output = model(data.view(-1, args.T, int(1024/args.T*3)))
            
            elif args.name == 'cifar10_noise':
                data, target = data, target.to(device)                
                x = data.view(-1, 32, 96)
                data = torch.cat((x, noise.repeat(x.shape[0],1,1,1).view(-1, 968, int(96))), 1)            
                data = Variable(data).to(device)                
                output = model(data)
                            
            
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
        
        accuracy = correct / total_num
        test_acc.append(accuracy)
        print('Epoch: ', epoch, 'Iteration: ', count, '| train loss: %.4f' % loss.item(), '| test accuracy: %.3f' % accuracy)


#        if args.model == 'NoisyRNN':
#            B = model.B.data.cpu().numpy()            
#            A = args.alpha * (args.beta * (B - B.T) + (1-args.beta) * (B + B.T) - args.gamma_A * np.eye(args.n_units))
#            A = 0.5 * (A + A.T)
#            e, _ = np.linalg.eig(A)
#            print('Eigenvalues of A (min and max): ', (np.min(np.abs(e)), np.max(np.abs(e))))
#            
#            C = model.C.data.cpu().numpy()            
#            W = args.beta * (C - C.T) + (1-args.beta) * (C + C.T) - args.gamma_W * np.eye(args.n_units)
#            e, _ = np.linalg.eig(W)
#            print('Eigenvalues of A (min and max): ', (np.min(np.abs(e)), np.max(np.abs(e))))
            
             

    # schedule learning rate decay    
    optimizer=exp_lr_scheduler(epoch, optimizer, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

print('total time: ', timeit.default_timer()  - t0 )


torch.save(model, args.name + '_results/' + args.model + '_' + args.name + '_T_' + str(args.T) 
            + '_units_' + str(args.n_units) + '_beta_' + str(args.beta) 
            + '_gamma_A_' + str(args.gamma_A) + '_gamma_W_' + str(args.gamma_W) + '_eps_' + str(args.eps) 
            + '_solver_' + str(args.solver) + '_gclip_' + str(args.gclip) + '_optimizer_' + str(args.optimizer)
            + '_addnoise_' + str(args.add_noise) + '_multnoise_' + str(args.mult_noise) 
            + '_seed_' + str(args.seed) + '.pkl')  

data = {'loss': lossaccum, 'testacc': test_acc}
f = open(args.name + '_results/' + args.model + '_' + args.name + '_T_' + str(args.T) 
            + '_units_' + str(args.n_units) + '_beta_' + str(args.beta) 
            + '_gamma_A_' + str(args.gamma_A) + '_gamma_W_' + str(args.gamma_W) + '_eps_' + str(args.eps) 
            + '_solver_' + str(args.solver) + '_gclip_' + str(args.gclip) + '_optimizer_' + str(args.optimizer)
            + '_addnoise_' + str(args.add_noise) + '_multnoise_' + str(args.mult_noise) 
            + '_seed_' + str(args.seed) + '_loss.pkl',"wb")

pickle.dump(data,f)
f.close()