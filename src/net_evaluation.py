import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement
from tqdm.notebook import tqdm
import seaborn as sns
import networkx as nx

def load_edgeList(f, weighted = False):
	if f.endswith('.wdag'):
		# weighted directed acyclic graph
		network = nx.read_weighted_edgelist(f)
		edges = {e:network.edges[e]['weight'] for e in network.edges}
	elif f.endswith('.sif'):
		network = nx.read_edgelist(f, delimiter='\t', data=False)
		edges = list(network.edges())
	elif f.endswith('.csv'):
		# .csv file should have 3 columns, indicating 2 protein and a score
		# edges predicted by GCN are normally in .csv format
		if weighted:
			df = pd.read_csv(f, header = 0, names=['protein1','protein2','score'])
			edges = {(h,t):w for h,t,w in zip(df.protein1, df.protein2, df.score)}
		else:
			df = pd.read_csv(f, header = 0)
			edges = list(set(zip(df.iloc[:,0], df.iloc[:,1])))
	elif f.endswith('.txt'):
		df = pd.read_table(f, header=None)
		edges = list(set(zip(df.iloc[:,0], df.iloc[:,1])))
	else:
		raise ValueError("file type not supported")
	return edges

def HyperGeometric_evaluate(my_edges, truth_edges = None, 
							truth_net_file = None, truth_edges_directed = False,
							n_gene = None, TFs = None, n_top_edges = None, verbose = False):
	
	if truth_edges is None:
		if truth_net_file is None :
			raise ValueError("no truth graph provided")
		truth_edges = load_edgeList(truth_net_file)

	truth_edges = truth_edges.copy()
	my_edges = my_edges.copy()

	if isinstance(truth_edges, dict):
		truth_edges = list(truth_edges.keys())
	if not truth_edges_directed:
		truth_edges = set(list(truth_edges) + [(t,h) for h,t in truth_edges])

	
	if isinstance(my_edges, pd.core.frame.DataFrame):
		my_edges.columns = ['head','tail','weight']
		my_edges = {(h,t):w for h,t,w in zip(my_edges['head'],my_edges['tail'],my_edges['weight'])}

	if isinstance(my_edges, list):
		my_edges = {(h,t):0 for h,t in my_edges}
		my_edges_weighted = False
	else:
		my_edges_weighted = True


	all_genes = set(chain(*list(my_edges.keys()))) & set(chain(*truth_edges))
	n_gene = len(all_genes)
	my_edges = {(h,t):w for (h,t),w in my_edges.items() if h in all_genes and t in all_genes}

	if TFs is None:
		# only keep truth edges between used genes
		truth_edges = [(h,t) for h,t in truth_edges if h in all_genes and t in all_genes]
		n_possible_edges = n_gene**2
		
	else:
		# only keep truth edges start from TFs
		if TFs == 'head':
			TFs = set([h for h,t in my_edges.keys()])
		truth_edges = [(h,t) for h,t in truth_edges if h in TFs and t in all_genes]
		my_edges = {(h,t):w for (h,t),w in my_edges.items() if h in TFs}
		n_possible_edges = len(TFs) * n_gene
		# print([len(TFs),n_gene])

	truth_edges = set(truth_edges)
	num_truth_edges = len(truth_edges)

	# choose my edges
	if n_top_edges is None:
		n_top_edges = float('inf')
	if my_edges_weighted:
		my_edges = [k for k,v in sorted(my_edges.items(), 
										key=lambda item: abs(item[1]), reverse=True)]
		n_my_val_edges = min([num_truth_edges, n_top_edges, len(my_edges)])
		my_val_edges = set(my_edges[:n_my_val_edges])
	else:
		n_my_val_edges = len(my_edges)
		my_val_edges = set(list(my_edges.keys()))
		# print(list(my_edges.values())[:10])
	
	# hypergeometric expectation
	expect = n_my_val_edges*num_truth_edges/n_possible_edges

	overlap = my_val_edges.intersection(truth_edges)
	odd_ratio = 1. * len(overlap) / expect
	if verbose:
		print('%d\ttruth edges\n%d\tpossible edges \
			   \n%d\tchoose edges\n%d\thit edges\
			   \n%.3f\todd ratio'%(num_truth_edges,
								n_possible_edges, 
								n_my_val_edges, 
								len(overlap),
								odd_ratio))
	return len(overlap), odd_ratio, num_truth_edges


def AUC_evaluate(my_edges, truth_edges, truth_edges_directed = False,TFs = None, verbose = False):
	truth_edges = truth_edges.copy()
	my_edges = my_edges.copy()
	if isinstance(my_edges, pd.core.frame.DataFrame):
		my_edges.columns = ['head','tail','weight']
		my_edges = {(h,t):w for h,t,w in zip(my_edges['head'],my_edges['tail'],my_edges['weight'])}

	all_genes = set(chain(*list(my_edges.keys())))
	if TFs is None:
		possibleEdges = set(product(all_genes,repeat = 2))
	else:
		if TFs == 'head':
			TFs = set([h for h,t in my_edges.keys()])
		possibleEdges = set(product(TFs, all_genes))

	if isinstance(truth_edges, dict):
		truth_edges = list(truth_edges.keys())
	if not truth_edges_directed:
		truth_edges = set(list(truth_edges) + [(t,h) for h,t in truth_edges])
	truth_edges = set(truth_edges) & possibleEdges
	truth_edges = [s+'|'+t for s,t in truth_edges]
	# print(len(truth_edges))
	
	my_edges_keys = [s+'|'+t for s,t in my_edges.keys()]
	my_edges_values = [abs(w) for w in list(my_edges.values())]

	outDF = pd.DataFrame(np.zeros(shape = [len(possibleEdges), 2]))
	outDF.columns = ['TrueEdges','PredEdges']
	outDF.index = ['|'.join(p) for p in possibleEdges]

	outDF.loc[truth_edges, 'TrueEdges'] = 1
	outDF.loc[my_edges_keys, 'PredEdges'] = my_edges_values

	fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
									 y_score=outDF['PredEdges'], pos_label=1)
	AUROC = auc(fpr, tpr)
	prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
													  probas_pred=outDF['PredEdges'], pos_label=1)
	AUPRC = auc(recall, prec)

	# permutation
	ew_values = np.array(outDF.loc[:, 'PredEdges']).copy()
	permutated_AUPRC = []
	for i in range(10):
		np.random.shuffle(ew_values)
		outDF.loc[:, 'PermutedEdges'] = ew_values
		prec_, recall_, _thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
														  	probas_pred=outDF['PermutedEdges'], 
														  	pos_label=1)
		permutated_AUPRC.append(auc(recall_, prec_))
		
	odd_ratio = AUPRC / np.mean(permutated_AUPRC)

	if verbose:
		print('%.5f\tAUPRC\n%.5f\tpermuted mean AUPRC \
			   \n%.5f\todd ratio\n%.5f\tAUROC'%(AUPRC,
											np.mean(permutated_AUPRC), 
											odd_ratio, 
											AUROC))
	return AUPRC, odd_ratio, AUROC
