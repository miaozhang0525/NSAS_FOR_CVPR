###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import SearchCell
from .genotypes        import Structure



import random
from tqdm import tqdm
from shutil import copy
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .data_dvae.util import *
from .data_dvae.models_dvae import *
#from bayesian_optimization.evaluate_BN import Eval_BN

import copy


import argparse

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='DARTS: DARTS-format CNN structures; ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='./data_dvae/bench_102_num', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=5, help='number of different node types, \
                    12 for DARTS and 6 for ENAS')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=30, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=32, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args([])


args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))

cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

graph_args.num_vertex_type = 8   # original types + add types
graph_args.max_n = 12  # maximum number of nodes
graph_args.START_TYPE = 5  # predefined start vertex type
graph_args.ADD_TYPE = 6 
graph_args.END_TYPE = 7 # predefined end vertex type

model_dvae = eval(args.model)(
        graph_args.max_n, 
        graph_args.num_vertex_type, 
        graph_args.START_TYPE, 
        graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional
        )

optimizer = optim.Adam(model_dvae.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model_dvae.to(device)

model_dvae.load_state_dict(torch.load('./data_dvae/bench102_model.pt'))  ##################DVAE,,     


model_linear_nn=torch.nn.Sequential(
    torch.nn.Linear(31,100),
    torch.nn.ReLU(),    
    torch.nn.Dropout(0.5),
    torch.nn.Linear(100,100),
    torch.nn.ReLU(),
    torch.nn.Linear(100,args.nz),
     )
####A simple linear model that project vector data to latent representtaion in DVAE
model_linear_nn.load_state_dict(torch.load('./data_dvae/model_linear_nn.pt'))  ##################DVAE,,  




class TinyNetworkEENAS(nn.Module):

  #def __init__(self, C, N, max_nodes, num_classes, search_space, affine=False, track_running_stats=True):
  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(TinyNetworkEENAS, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(1, 30) )
    #self.hardwts_ = nn.Parameter( 1e-3*torch.zeros(num_edge, len(search_space)) )
    self.tau        = 10
   # self.arch_cache = self.genotype()

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

 # def hardwts(self):
 #   return self.hardwts_


  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

  def get_alphas(self):
    return [self.arch_parameters]

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    
    uu=self.arch_parameters.data.to(device)
    g_r=model_dvae.decode(uu)
    inde=g_r[0].vs['type']
    new_inde=[]
    for i in [2,4,5,7,8,9]:
      new_inde.append(inde[i])
    k=0
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        op_name = self.op_names[ new_inde[k] ]
        k +=1
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure(genotypes)

  def gene(self):
    
    uu=self.arch_parameters.data.to(device)
    g_r=model_dvae.decode(uu)
    inde=g_r[0].vs['type']
    new_inde=[]
    for i in [2,4,5,7,8,9]:
      new_inde.append(inde[i])

    return new_inde

  def index_hardwts(self):
    
    logits  = self.arch_parameters.log_softmax(dim=1)
    probs   = nn.functional.softmax(logits, dim=1)  
    probs=torch.reshape(probs,(6,5))
    uu=self.arch_parameters.data.to(device)
    g_r=model_dvae.decode(uu)
    inde=g_r[0].vs['type']
    index=torch.zeros(6,1)
    k=0
    for i in [2,4,5,7,8,9]:
      index[k]=torch.tensor(inde[i])
      k +=1
    index=index.long()  
    one_h   = torch.zeros(6, 5).scatter_(-1, index, 1.0).cuda()########num_edge, len(search_space)
    hardwts = one_h - probs.detach() + probs

    return index,hardwts

  def get_index_hardwts(self):
    
      logits  = self.arch_parameters.log_softmax(dim=1)
      probs   = nn.functional.softmax(logits, dim=1)  
      probs=torch.reshape(probs,(6,5))
      uu=self.arch_parameters.data.to(device)
      g_r=model_dvae.decode(uu)
      inde=g_r[0].vs['type']
      index=torch.zeros(6,1)
      k=0
      for i in [2,4,5,7,8,9]:
        index[k]=torch.tensor(inde[i])
        k +=1
      index=index.long()  
      one_h   = torch.zeros(6, 5).scatter_(-1, index, 1.0).cuda()########num_edge, len(search_space)
      hardwts = one_h - probs.detach() + probs
        

      return index,hardwts



  def forward(self,inputs,g,index,hardwts):
    if g==0:	       
      logits  = self.arch_parameters.log_softmax(dim=1)
      probs   = nn.functional.softmax(logits, dim=1)  
      probs= torch.reshape(probs,(6,5))
      uu=self.arch_parameters.data.to(device)
      g_r=model_dvae.decode(uu)
      inde=g_r[0].vs['type']
      new_inde=torch.Tensor(inde)
      index=torch.zeros(6,1)
      k=0
      for i in [2,4,5,7,8,9]:
        index[k]=new_inde[i]
        k +=1
      index=index.long()  
      one_h   = torch.zeros(6, 5).scatter_(-1, index, 1.0).cuda()########num_edge, len(search_space)
      hardwts = one_h - probs.detach() + probs
        
    
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell.forward_gdas(feature, hardwts, index)
        #feature = cell.forward_dynamic(feature, self.arch_cache)
      else:
        feature = cell(feature)
    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits,index,hardwts
