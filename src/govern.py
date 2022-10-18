from .GCN_model import DeepNet_MLP_v1, SAGEConv_v2, CosineAnnealingWarmRestarts, train2, test2, device, DeepNet_MLP_sc
from .GCN_utils import GenExprDatasets, collate_func, plot_durations, DataLoader

from itertools import product, chain
import numpy as np
import pandas as pd
import torch as t
import time

class GOVERN():
    def __init__(self, adata, v_key = 'velocity', 
                 edges = None, gene_used = None, 
                 single_cell_version = False, use_cuda = None) -> None:
        if gene_used is not None and edges is not None:
            raise ValueError("provide either gene_used or edges")
        
        if use_cuda is None:
            self.use_cuda = t.cuda.is_available()
        else:
            self.use_cuda = use_cuda
        
        if gene_used is None:
            if edges is None:
                gene_used = adata.var_names
            else:
                gene_used = list(set(chain(*edges)))
        adata = adata[:,gene_used]
        n_genes = len(gene_used)

        if edges is None:
            edges = product(gene_used, gene_used)
        
        self.adata = adata
        self.v_key = v_key
        self.edges = edges
        self.gene_used = gene_used
        self.n_genes = n_genes
        self.single_cell_version = single_cell_version
        
        model = DeepNet_MLP_sc if self.single_cell_version else DeepNet_MLP_v1
        self.model = model(in_channel=1,mid_channel=8,out_channel=1,
                            num_nodes=self.n_genes, edge_num=len(self.edges),
                            graph_conv=SAGEConv_v2)
        if self.use_cuda:
            self.model = self.model.cuda()
        
    def train(self, adata = None, batch_size = 10, epoch = 100, 
              plot_loss_during_training = False, return_epoch_edge_weight = False, 
              random_split = 0.8, save_ew_each = None, verbose = False):
        if self.single_cell_version:
            batch_size = 1
        if save_ew_each is None:
            save_ew_each = max(1, epoch//100)
        
        
        if adata is None:
            adata = self.adata
        else:
            adata = adata[:,self.gene_used]
            
        gene_idx = {g:i for i,g in enumerate(self.gene_used)}
        edges_array = np.array([(gene_idx[h], gene_idx[t]) for h,t in self.edges]).T
        nndata = GenExprDatasets(adata, edges_array, random_split = random_split, vkey=self.v_key)
        
        train_dataset,test_dataset = nndata.get_nn_datasets()
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                        collate_fn = collate_func)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,
                                collate_fn = collate_func)
        
        optimizer = t.optim.Adam(self.model.parameters(),lr=0.01 ,)
        min_lr = 0.00001
        scheduler = CosineAnnealingWarmRestarts(optimizer,2, 2, eta_min=min_lr, lr_max_decay=0.5)
        loss = []
        ew = []
        
        start_time = time.time()
        e_time = time.time()
        for e in range(epoch):
            train_loss = train2(self.model,optimizer,train_loader,e,device,
                                loss_fn = None,scheduler =scheduler, verbose = verbose)
            test_mse,test_true,test_output= test2(self.model,test_loader,predicts=True)
            loss.append(list(train_loss)+[test_mse])
            if plot_loss_during_training:
                plot_durations(np.array(loss), ['loss','rec_loss','l1_loss','test_mse'], 
                                'epoch: %s, total time: %.2f, last epoch time: %.2f'%(str(e), time.time()-start_time, time.time()-e_time))
                
            if return_epoch_edge_weight and e % save_ew_each == 0:
                epoch_ew = self.model.edge_weight.cpu().detach().numpy()
                ew.append(epoch_ew)
            e_time = time.time()
        
        print('Training complete, total time: %s second')
        return loss,ew
    
    def predict_velocity(self, adata, pred_v_key = 'velocity_pred'):
        adata = adata[:,self.gene_used]
        gene_idx = {g:i for i,g in enumerate(self.gene_used)}
        edges_array = np.array([(gene_idx[h], gene_idx[t]) for h,t in self.edges]).T
        nndata = GenExprDatasets(adata, edges_array, random_split = False, vkey=self.v_key)
        _, pred_data = nndata.get_nn_datasets()
        loader = DataLoader(pred_data,batch_size=10,shuffle=False,
                    collate_fn = collate_func)
        y_output=[]
        for data in loader:
            if self.use_cuda:
                data = data.cuda()
            # print(data.y.shape)
            output = self.model(data)
            y_output.extend(output.cpu().data.numpy().T)
        y_output = np.array(y_output)
        velo_pred = nndata.inverse_trans(y_output)
        adata.layers[pred_v_key] = velo_pred
        return adata

    def get_GRN(self, adata = None):
        if not self.single_cell_version:
            df = pd.Series(self.model.edge_weight.detach().cpu().numpy())
            df.index = self.edges
            return df
        
        if adata is None:
            adata = self.adata
        adata = adata[:, self.gene_used]
            
        X = t.tensor(adata.X.todense()).cuda()
        scew = self.model.edge_MLP(X).cpu().detach().numpy()
        df = pd.DataFrame(scew)
        df.index = adata.obs_names
        df.columns = self.edges
        return df