# dataset generation
# visualization 
# evaluation

import os
import pylab as pl
import numpy as np
import matplotlib
from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()
import torch
cuda_is_available = torch.cuda.is_available()
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from pynverse import inversefunc

def is_notebook(): 
    return "JPY_PARENT_PID" in os.environ

def plot_durations(y):
    pl.clf()
    pl.plot(y)
    pl.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.display(pl.gcf())
        display.clear_output(wait=True)

class GenExprDatasets():
    def __init__(self, adata, trans_func = 'sigmoid10', random_split = 0.8):
        self.expr = adata.X.todense().copy()
        self.velo = adata.layers['velocity'].copy()
        self.nCells = self.expr.shape[0]
        self.nGenes = self.expr.shape[1]
        self.trans_func = trans_func
        self._transform()
        self.nCells = len(adata)
        if random_split is not False:
            shuffle_idx = list(range(self.nCells))
            np.random.shuffle(shuffle_idx)
            train_idx = shuffle_idx[:round(random_split*self.nCells)]
            test_idx = shuffle_idx[round(random_split*self.nCells):]
        else:
            train_idx = list(range(self.nCells))
            test_idx = list(range(self.nCells))
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.train = exprDataset(self.expr[train_idx,:], self.velo_trans[train_idx,:])
        self.test = exprDataset(self.expr[test_idx,:], self.velo_trans[test_idx,:])

    def _transform(self):
        if self.trans_func == 'sigmoid10':
            self.velo_trans = (F.sigmoid(torch.tensor(self.velo).float()) * 10).numpy()
        elif self.trans_func == 'log':
            self.velo_trans = self.velo.copy()
            self.velo_trans[self.velo_trans<0] = -np.log1p(abs(self.velo_trans[self.velo_trans<0]))
            self.velo_trans[self.velo_trans>0] =  np.log1p(abs(self.velo_trans[self.velo_trans>0]))
            self.velo_trans = self.velo_trans*10

    def get_orig_velo(self, which = 'training'):
        if which == 'whole':
            return(self.velo.copy())
        if which == 'training':
            return(self.velo[self.train_idx,:].copy())
        if which == 'test':
            return(self.velo[self.test_idx,:].copy())

    def inverse_trans(self, y):
        if self.trans_func == 'sigmoid10':
            velo = y.copy()/10
            velo = np.log(velo/(1-velo))
            return velo
        elif self.trans_func == 'log':
            velo = y.copy()/10
            velo[velo<0] = -np.exp(abs(velo[velo<0]))
            velo[velo>0] =  np.exp(abs(velo[velo>0]))
            return velo

    def get_nn_datasets(self):
        return self.train, self.test



class exprDataset(torch.utils.data.Dataset):
    def __init__(self, expr, velo):
        super().__init__()
        self.device = 'cuda' if cuda_is_available else 'cpu'
        self.x = torch.tensor(expr).float().to(self.device)
        self.y = torch.tensor(velo).float().to(self.device)
    
    def __len__(self):
        return(self.y.shape[0])
    
    def __getitem__(self, idx):
        cur_x = self.x[idx, ]
        cur_y = self.y[idx, ]
        return(cur_x,cur_y)


def evaluation(velo_truth, velo_pred, eval_func = 'cosine'):
    x = velo_truth.copy()
    y = velo_pred.copy()
    if isinstance(velo_truth, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(velo_pred, torch.Tensor):
        y = y.detach().cpu().numpy()

    if eval_func=='cosine':
        xx = np.sum(x ** 2, axis=1) ** 0.5
        x = x / xx[:, np.newaxis]
        yy = np.sum(y ** 2, axis=1) ** 0.5
        y = y / yy[:, np.newaxis]
        simi = np.sum(x*y, axis=1)
        return simi
    if eval_func=='mse':
        return np.mean(np.square(velo_pred - velo_truth), axis=0)