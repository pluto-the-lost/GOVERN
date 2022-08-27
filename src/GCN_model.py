import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import math

import torch as t
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch.nn.init import xavier_normal_,kaiming_normal_
from torch.nn.init import uniform_,kaiming_uniform_,constant
from torch.nn.init import normal
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import data as tdata
from torch.utils import data
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected,remove_self_loops
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_softmax
device = 'cuda' if t.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    from GCN_utils import *
else:
    from src.GCN_utils import *


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class PermuteLayer(nn.Module):
    def __init__(self,permute_order): 
        super().__init__()
        self.permute_order = permute_order

    def forward(self,x):
        return x.permute(*self.permute_order)

class SAGEConv_v2(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, 
                 bias=True,activate=False,alphas=[1,1],
                 shared_weight=False,aggr = 'mean',
                 **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        # print('SAGEConv_v2 activate,aggr',activate,aggr)
        self.node_dim = 0 
        self.shared_weight = shared_weight
        self.activate = activate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if self.shared_weight:
            # print('shared weights for SAGEConv2')
            self.self_weight = self.weight 
        else:
            self.self_weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.alphas = alphas #[self_alpha, pro_alpha]
        # print('alphas',self.alphas)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        uniform(self.in_channels,self.self_weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):

        out  =  torch.matmul(x,self.self_weight )
        x = torch.matmul(x,self.weight)
        out2 = self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)
        return self.alphas[0]*out+ self.alphas[1]* out2

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1,1) * x_j

    def update(self, aggr_out):

        if self.activate:
            aggr_out = F.relu(aggr_out)
            
        # if torch.is_tensor(aggr_out):
        #     aggr_out = torch.matmul(aggr_out,self.weight )
        # else:
        #     aggr_out = (None if aggr_out[0] is None else torch.matmul(aggr_out[0], self.weight),
        #          None if aggr_out[1] is None else torch.matmul(aggr_out[1], self.weight))
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def init_weights(m):
    if type(m) == nn.Linear:
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1,lr_max_decay=0.9):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.lr_max_decay = lr_max_decay
        self.lr_max_cum_decay = 1 
        self.T_cur = 0 if last_epoch < 0 else last_epoch
        super().__init__(optimizer, last_epoch)
        # self.T_cur = last_epoch
        self.cur_n = 0 
        

    def get_lr(self):
        return [self.eta_min + (self.lr_max_cum_decay * base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every update, i.e. if one epoch has 10 iterations
        (number_of_train_examples / batch_size), we should call SGDR.step(0.1), SGDR.step(0.2), etc.
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = SGDR(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    if n!= self.cur_n :
                        #print('diff',self.cur_n,n)
                        self.cur_n = n 
                        self.lr_max_cum_decay *= self.lr_max_decay
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



class DeepNet_MLP_v1(nn.Module):
    def __init__(self,in_channel=1,mid_channel=8,out_channel=2,num_nodes=2207,edge_num=151215,
                use_nodes_for_output_list=None,graph_conv=SAGEConv_v2,**args):
        super().__init__()
        # print('DeepNet_MLP_v1 GNN+MLP v2-2 SAGE*1 + weighted-edge-softmax + concate')
        # self.max_pool_dim = 100 

        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio',0.3)
        self.activate_cls = args.get('activate_cls',nn.ReLU)
        # print('model dropout raito:',self.dropout_ratio)
        # print('model activation function:',self.activate_cls)
        
        self.conv1 = graph_conv(in_channel, mid_channel, )
        self.conv1.reset_parameters()
        self.bn1 = torch.nn.LayerNorm((num_nodes,mid_channel))
        self.act1 = nn.ReLU()
        
        self.conv2 = graph_conv(mid_channel, 1, )
        self.conv2.reset_parameters()
        self.act2 = nn.ReLU()
        
        self.edge_num = edge_num
        
        self.weight_edge_flag = True  
        # print('trainalbe edges :',self.weight_edge_flag)
        if self.weight_edge_flag:
            # self.edge_weight = nn.Parameter(t.rand(edge_num).float()*0.01)
            self.edge_weight = nn.Parameter(t.ones((edge_num)).float())
            # self.edge_weight = nn.Parameter(t.ones((num_nodes, num_nodes)).float())
            # print(self.edge_weight.shape)
            # print(self.edge_weight[:10])
        else:
            self.edge_weight = None
    
        self.reset_parameters()
        # self.
        # 对特定边初始化
    
    def reset_parameters(self,):
        self.conv1.apply(init_weights)
        pass

    def get_gcn_weight_penalty(self,mode='L2'):

        if mode == 'L1':
            func = lambda x:  t.sum(t.abs(x))
        elif mode =='L2':
            func  = lambda x: t.sqrt(t.sum(x**2))

        loss = 0 

        tmp = getattr(self.conv1,'weight',None)
        if tmp is not None: 
            loss += func(tmp)

        tmp = getattr(self.conv1,'self_weight',None)
        if tmp is not None: 
            loss += 1* func(tmp)

        tmp = getattr(self.global_conv1,'weight',None)
        if tmp is not None: 
            loss += func(tmp)
        tmp = getattr(self.global_conv2,'weight',None)
        if tmp is not None: 
            loss += func(tmp)

        return loss 

    def get_gcn_weight_penalty_org(self,mode='L2'):

        if mode == 'L1':
            func = lambda x:  t.sum(t.abs(x))
        elif mode =='L2':
            func  = lambda x: t.sqrt(t.sum(x**2))

        loss = 0 

        tmp = getattr(self.conv1,'weight',None)
        if tmp is not None: 
            loss += func(tmp)

        tmp = getattr(self.conv1,'self_weight',None)
        if tmp is not None: 
            loss += 5* func(tmp)
        tmp = getattr(self.conv1,'bias',None)
        if tmp is not None: 
            loss += func(tmp)

        return loss 

    def forward(self,data,get_latent_varaible=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
  
        if self.weight_edge_flag:
            # one_graph_edge_weight = torch.sigmoid(self.edge_weight)#*self.edge_num
            # one_graph_edge_weight = scatter_softmax(self.edge_weight, edge_index[1])
            # one_graph_edge_weight = F.softmax(self.edge_weight, dim = 1)
            # one_graph_edge_weight = self.edge_weight / t.sum(self.edge_weight, dim=1)
            # edge_weight = one_graph_edge_weight.view(-1)
            edge_weight = self.edge_weight
        else:
            edge_weight = None 

        x = self.act1(self.conv1(x, edge_index,edge_weight=edge_weight))
        # x = self.act2(self.conv2(x,edge_index,edge_weight=edge_weight))
        x = self.conv2(x,edge_index,edge_weight=edge_weight)
        
        return t.squeeze(x,dim=-1)


def train2(model,optimizer,train_loader,epoch,device,
            loss_fn =None,scheduler =None,verbose=False, 
            l1_start_at = 10):
    model.train()
    loss_all = 0
    loss_rec = 0
    loss_l1 = 0
    iters = len(train_loader)
    for idx,data in enumerate( train_loader):
        data = data.to(device)
        if verbose:
            print(data.y.shape,data.edge_index.shape)
        optimizer.zero_grad()
        output = model(data)
        if loss_fn is None:
            rec_loss = F.mse_loss(output, data.y)
        else:
            rec_loss = loss_fn(output, data.y, data.confidence)


        l1_loss = t.mean(t.abs(model.edge_weight))
        # l1_loss = t.tensor(0)
        
        if epoch<10:
            loss = rec_loss
        else:
            loss = rec_loss + l1_loss
        # print('rec = %.6f, l1 = %.6f'%(rec_loss.item(), l1_loss.item()))
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss_rec += rec_loss.item()
        loss_l1 += l1_loss.item()

        optimizer.step()

        if not (scheduler is  None):
            scheduler.step( (epoch -1) + idx/iters) # let "epoch" begin from 0 

    return loss_all/iters, loss_rec/iters, loss_l1/iters

def test2(model,loader,predicts=False,tobreak=False):
    model.eval()

    correct = 0
    y_pred =[]
    y_true=[]
    y_output=[]
    for data in loader:
        data = data.to(device)
        # print(data.y.shape)
        output = model(data)
        y = data.y.cpu().data.numpy()
        y_true.extend(y.T)
        y_output.extend(output.cpu().data.numpy().T)
        if tobreak:
            print('break for cycle..')
            break
    y_true,y_output = np.array(y_true),np.array(y_output)
    mse = np.mean((y_output-y_true)**2)
    if predicts:
         return mse,y_true,y_output
    else:
        return mse

def confidence_mse(output, target, confidence):
    return t.mean((output-target)**2 * confidence)


def weighted_mse(output, target, *arg):
    zeros = (target.detach()==0).float()
    nonzeros = (target.detach()!=0).float()
    l1 = t.sum((zeros*output-zeros*target)**2)
    l1 = l1/t.sum(zeros)
    l2 = t.sum((nonzeros*output-nonzeros*target)**2)
    l2 = l2/t.sum(nonzeros)

    # print('zero_loss = %.2f, nonz_loss = %.2f'%(l1.item(), l2.item()))
    return(0.1*l1 + l2)


