# dataset generation
# visualization 
# evaluation

import os
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib
from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()
import torch
import torch as t
import torch.nn.functional as F
import torch_geometric
cuda_is_available = torch.cuda.is_available()
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from tqdm.notebook import tqdm

def is_notebook():
	return("JPY_PARENT_PID" in os.environ)

def plot_durations(y, curve_names = None, title = None, file = None):
	if isinstance(y, list):
		y = np.array(y)
	
	if isinstance(y, dict):
		curve_names = list(y.keys())
		y = np.array(list(y.values()))
		
	y = y.reshape(-1,1) if len(y.shape)==1 else y
	nCurves = y.shape[1]
	if curve_names is None:
		curve_names = [str(i) for i in range(nCurves)]
		
	pl.figure(nCurves, figsize=(2*nCurves, 4), dpi= 100)
	pl.clf()
	pl.subplots_adjust(hspace =0.7)
	if title is not None:
		pl.suptitle(title, fontsize=10)
	for i in range(nCurves):
		pl.subplot(nCurves,1,i+1)
		pl.plot(y[:,i])
		pl.ylabel(curve_names[i],rotation=0, labelpad=-400)
	if is_ipython and file is None:
		display.display(pl.gcf())
		display.clear_output(wait=True)
	if file is not None:
		pl.savefig(file)
		pl.close()

def collate_func(batch):
	data0 = batch[0]
	if isinstance(data0,torch_geometric.data.Data):
		tmp_x = [xx['x'] for xx in batch]
		tmp_y = [xx['y'] for xx in batch]
		tmp_c = [xx['confidence'] for xx in batch]
	elif isinstance(data0,(list,tuple)):
		tmp_x = [xx[0] for xx in batch]
		tmp_y = [xx[1] for xx in batch]
		tmp_c = [xx[2] for xx in batch]

	tmp_data = torch_geometric.data.Data()
	tmp_data['x']= t.stack(tmp_x,dim=1)
	tmp_data['y']= t.stack(tmp_y,dim=1) #t.cat(tmp_y) # 

	tmp_data['confidence']= t.cat(tmp_c,dim=0).reshape(1,-1)
	tmp_data['edge_index']=data0.edge_index 
	tmp_data['batch'] = t.zeros_like(tmp_data['y'])
	tmp_data['num_graphs'] = 1 
	return tmp_data
	# return Batch.from_data_list([tmp_data])

# from torch_geometric.data import Collater
class DataLoader(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], 
				 **kwargs):
		if 'collate_fn' not in kwargs.keys():
			raise
			
		super().__init__(dataset, batch_size, shuffle, **kwargs)

class GenExprDatasets():
	def __init__(self, adata, edges, trans_func = 'sigmoid10', random_split = 0.8, vkey = 'velocity'):
		if isinstance(adata.X, scipy.sparse.csr.csr_matrix):
			self.expr = adata.X.todense().copy()
		else:
			self.expr = adata.X.copy()
		# self.expr = adata.X.todense().copy()
		self.velo = adata.layers[vkey].copy()
		self.vkey = vkey
		self.edges = edges
		if 'velocity_confidence' in adata.obs.columns:
			self.confidence = np.array(adata.obs.velocity_confidence).reshape(-1,1)
		else:
			self.confidence = np.ones(len(adata)).reshape(-1,1)
		self.nCells = self.expr.shape[0]
		self.nGenes = self.expr.shape[1]
		self.nEdges = self.edges.shape[0]
		self.edge_names = list(zip(adata.var_names[edges[0,:]], 
						  			adata.var_names[edges[1,:]]))
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
		# print(self.nCells, len(train_idx), len(test_idx))
		self.train = ExprDataset(self.expr[train_idx,:], self.edges,
								 self.velo_trans[train_idx,:], self.confidence[train_idx])
		self.test = ExprDataset(self.expr[test_idx,:], self.edges, 
								self.velo_trans[test_idx,:], self.confidence[test_idx])

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
			y = np.clip(y.copy(), a_min=1e-4, a_max=10-(1e-4))
			velo = y/10
			velo = np.log(velo/(1-velo))
			return velo
		elif self.trans_func == 'log':
			velo = y.copy()/10
			velo[velo<0] = -np.exp(abs(velo[velo<0]))
			velo[velo>0] =  np.exp(abs(velo[velo>0]))
			return velo

	def get_nn_datasets(self):
		return self.train, self.test

class ExprDataset(torch.utils.data.Dataset):#需要继承data.Dataset
	def __init__(self,Expr,edge,y,confidence,save_path=None):
		super().__init__()

		self.save_path = save_path
		if (save_path is not None) :
			if (os.path.exists(self.save_path) )  :
				self.Expr,self.common_edge,self.y,self.gene_num,self.edge_num = torch.load(self.save_path)
		else:
			# print('processing...')
			self.gene_num = Expr.shape[1]
			self.edge_num = edge.shape[1]
			# print('edge:',self.edge_num)
			if not isinstance(Expr,t.Tensor):
				self.Expr = t.tensor(Expr).float()
				self.common_edge =t.tensor(edge).long()
				self.y = t.tensor(y).float()
				self.confidence = t.tensor(confidence).float()
			else:
				self.Expr = Expr
				self.common_edge =edge
				self.y = y
				self.confidence = confidence
			if self.save_path is not None:
				torch.save((self.Expr,self.common_edge,self.y,self.gene_num,self.edge_num), self.save_path)
	  
	def __getitem__(self, idx):
		if isinstance(idx, int):
			data = self.get(idx)
			return data
		raise IndexError(
			'Only integers are valid '
			'indices (got {}).'.format(type(idx).__name__))
		pass

	def split(self,idx):
		return ExprDataset(self.Expr[idx,:],self.common_edge,self.y[idx],save_path=None)
	def __len__(self):
		# You should change 0 to the total size of your dataset.
		return len(self.y)

	def get(self,index):
		data = torch_geometric.data.Data()
		data['x']= self.Expr[index,:].view([-1,1])
		data['y']= self.y[index]#.view(-1,1)
		data['confidence'] = self.confidence[index]
		data['edge_index']=self.common_edge 
		return data


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

def pseudotime_to_velo(adata, kernel_method = 'umap', 
						pt_key = 'pseudotime', vkey = 'pseudo_velocity'):
	def pairwise_distances(x):
		x_norm = (x**2).sum(1).reshape(-1,1)
		y_t = x.T
		y_norm = x_norm.reshape(1, -1)
		dist = x_norm + y_norm - 2.0 * np.dot(x, y_t)
		dist = dist - np.dot(np.eye(x.shape[0]), dist.diagonal())
		return np.clip(dist, 0.0, np.inf)

	nbs = adata.uns['neighbors']['indices']

	if isinstance(adata.X, scipy.sparse.csr.csr_matrix):
		X = np.array(adata.X.todense())
	else:
		X = adata.X

	pt = adata.obs[pt_key].copy()
	pt = (pt-min(pt))/(max(pt)-min(pt)) * 10
	pt = np.array(pt).reshape(-1,1)
	t_diff = pt - pt.T + 1e-5

	if kernel_method == 'GRISLI':
		x_dist = pairwise_distances(X)
		t_dist = pairwise_distances(pt)
		sigma_t = np.quantile(t_dist, q = 0.1)
		sigma_x = np.quantile(x_dist, q = 0.1)
		kernel = np.multiply(np.multiply(t_dist,np.exp(-t_dist/(2*sigma_t ** 2))),
							np.exp(-x_dist/(2*sigma_x ** 2)))
	elif kernel_method == 'umap':
		connectivities = adata.uns['neighbors']['connectivities'].todense()
		t_dist = pairwise_distances(pt)
		t_dist_exp = np.exp(-t_dist/np.quantile(t_dist, q = 0.1))
		kernel = np.multiply(connectivities, t_dist_exp)
	
	velo = np.zeros(adata.X.shape)
	for i in tqdm(range(len(adata))):
		V_is = {j:(X[j,:]-X[i,:])/t_diff[j,i] for j in range(len(adata))}#nbs[i,1:]}

		V_i_kernel = {j:kernel[i,j] for j in range(len(adata))}#nbs[i,1:]}
		V_i_kernel_sum = sum(V_i_kernel.values())
		# print(V_i_kernel_sum, V_is[list(V_is.keys())[0]][:10])
		V_i = sum([V_is[j]*V_i_kernel[j]/V_i_kernel_sum for j in V_i_kernel])

		# V_is = np.array([(X[j,:]-X[i,:])/t_diff[j,i] for j in range(len(adata))])

		# V_i_k1 = {j:kernel[i,j] for j in nbs[i,1:] if pt[j]>pt[i]}
		# V_i_k1 = {k:w for k,w in sorted(V_i_k1.items(), key=lambda x:x[1])}
		# V_i_k2 = {j:kernel[i,j] for j in nbs[i,1:] if pt[j]<pt[i]}

		# V_i_k1 = {j:kernel[i,j] for j in range(len(adata)) if pt[j]>pt[i]}
		# V_i_k2 = {j:kernel[i,j] for j in range(len(adata)) if pt[j]<pt[i]}
		# while sum(V_i_k1.values()) < 0.0001 and len(V_i_k1)>0:
		# 	V_i_k1 = {k:w*10 for k,w in V_i_k1.items()}
		# while sum(V_i_k2.values()) < 0.0001 and len(V_i_k2)>0:
		# 	V_i_k2 = {k:w*10 for k,w in V_i_k2.items()}

		# s_k1 = sum(V_i_k1.values())
		# s_k2 = sum(V_i_k2.values())

		# V_i = sum([0.5*V_is[j]*V_i_k1[j]/s_k1 for j in V_i_k1]) + \
		# 	  sum([-0.5*V_is[j]*V_i_k2[j]/s_k2 for j in V_i_k2])
		# V_i = np.mean([V_is[j] for j in list(V_i_k1.keys())])
		# V_i = pd.Series(V_i).fillna(0).values
		velo[i,:] = V_i
	adata.layers[vkey] = velo
	return kernel