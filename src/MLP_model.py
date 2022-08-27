# neural network models, training and test

import warnings
warnings.filterwarnings('ignore')

# science computation and plot
import os
from copy import deepcopy
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import operator
from collections import Counter
import math

from scipy.stats import spearmanr
import scipy.spatial.distance as distance  #distance.pdist()
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline, BSpline

# deep learning
import torch as t
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch.nn.init import xavier_normal_,kaiming_normal_
from torch.nn.init import uniform_,kaiming_uniform_,constant
from torch.utils import data as tdata
from torch.utils import data
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected,remove_self_loops
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data

if __name__ == "__main__":
    from MLP_utils import *
else:
    from src.MLP_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if is_notebook():
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

class myMLP(nn.Module):
    def __init__(self, in_channel = 100, mid_channel = 128, out_channel = 100, ):
        super().__init__()
        self.linear = nn.Sequential(
            Linear(in_channel, mid_channel),
            nn.ReLU(inplace=True),
            Linear(mid_channel, mid_channel),
            nn.ReLU(inplace=True),
            Linear(mid_channel, out_channel),
#             nn.Sigmoid()
        )
    def forward(self, x):
        out = self.linear(x)
        return(out)


