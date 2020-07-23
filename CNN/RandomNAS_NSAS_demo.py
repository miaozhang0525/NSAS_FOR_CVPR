import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import genotypes

import copy


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs')######change the epochs
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def random_arch_generate():
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = 4####model._step

    arch_gene = []
    for i in range(n_nodes):
        ops = np.random.choice(range(num_ops), 2)
        nodes_in_normal = np.random.choice(range(i+2), 2)##############################modify
        arch_gene.extend([(ops[0],nodes_in_normal[0]), (ops[1],nodes_in_normal[1])])
    return arch_gene  

def cal_arch_dis(arch1,arch2):#calculate the distance, smaller distance more similar
    dis=8
    n_nodes=4######genotypes.STEPS

    for i in range(n_nodes):
        if arch1[2*i]==arch2[2*i]:
            dis=dis-1
        elif arch1[2*i]==arch2[2*i+1]:
            dis=dis-1
        if arch1[2*i+1]==arch2[2*i+1]:
            dis=dis-1
        elif arch1[2*i+1]==arch2[2*i]:
            dis=dis-1                      
    dis=dis/8
    return dis 



def cal_diver_score(arch,archive):###KNN based diversity calculation
    n=len(archive)
    dis=np.zeros(n)
    for i in range(n):
        dis[i]=cal_arch_dis(arch,archive[i])
        
    sort_dis=np.sort(dis)

    diver_score=np.mean(sort_dis[0:10])###################################k=10 for knn
    
    return diver_score
 

    
def diver_arch_generate(arch_archive):############randomly genrate architecture and get the best one
    ini_diver_score=0
    arch_g=random_arch_generate()
    for i in range(10):##################
        arch=random_arch_generate()         
        diver_score=cal_diver_score(arch,arch_archive)#########kernel metric, the samller the better
        if diver_score>ini_diver_score:
            arch_g=arch
            ini_diver_score=diver_score
            
    return arch_g


def diver_arch_replace(index,arch_archive,archive_recent):############randomly generate architecture to repalce
    arch_compar=arch_archive[index]
    a=np.arange(0,index)
    b=np.arange(index+1,len(arch_archive))
    index_remain=np.append(a,b)
    
    arch_archive_remain=[arch_archive[j] for j in index_remain]
    
    ini_diver_score=cal_diver_score(arch_compar,arch_archive_remain)
    for i in range(len(archive_recent)):######################################
        arch=archive_recent[i] 
        diver_score=cal_diver_score(arch,arch_archive_remain)
        if diver_score>ini_diver_score:
            arch_compar=arch
            ini_diver_score=diver_score
            
    return arch_compar


def find_similar_arch(arch,archive):##get the index of the most similar architecture
    dis=np.zeros(len(archive))   
    
    for i in range(len(archive)):
        dis[i]=cal_arch_dis(arch,archive[i])################

    m=np.argsort(dis)
    index=m[0]
    
    return index


def arch_archive_update(arch_gene,arch_archive,normal_archive_recent,reduction_archive_recent):####update architecture archive (also the constraint subset)
    store_num=8########set the ARCIVE number M
    if len(arch_archive)==2*store_num:
        ind_arch_norm_replace=find_similar_arch(arch_gene[0],arch_archive[0:len(arch_archive):2])
        ind_arch_redu_replace=find_similar_arch(arch_gene[1],arch_archive[1:len(arch_archive):2])
        arch_archive[2*ind_arch_norm_replace]=diver_arch_replace(ind_arch_norm_replace,arch_archive[0:len(arch_archive):2],normal_archive_recent)
        arch_archive[2*ind_arch_redu_replace+1]=diver_arch_replace(ind_arch_redu_replace,arch_archive[1:len(arch_archive):2],reduction_archive_recent)
        
    else:
        normal_arch=diver_arch_generate(arch_archive[0:len(arch_archive):2])
        reduce_arch=diver_arch_generate(arch_archive[1:len(arch_archive):2])######greedy
        arch_archive.append(normal_arch)
        arch_archive.append(reduce_arch)
    return arch_archive


def get_weights_from_arch(arch_comb):
    k = sum(1 for i in range(model._steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = model._steps

    alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    offset = 0
    for i in range(n_nodes):
        normal1 = np.int_(arch_comb[0][2*i])
        normal2 = np.int_(arch_comb[0][2*i+1])
        reduce1 = np.int_(arch_comb[1][2*i])
        reduce2 = np.int_(arch_comb[1][2*i+1])
        alphas_normal[offset+normal1[1],normal1[0]] = 1
        alphas_normal[offset+normal2[1],normal2[0]] = 1
        alphas_reduce[offset+reduce1[1],reduce1[0]] = 1
        alphas_reduce[offset+reduce2[1],reduce2[0]] = 1
        offset += (i+2)

    model_weights = [
      alphas_normal,
      alphas_reduce,
    ]
    return model_weights


def set_model_weights(model, weights):
    model.alphas_normal = weights[0]
    model.alphas_reduce = weights[1]
    model._arch_parameters = [model.alphas_normal, model.alphas_reduce]
    return model



def cal_loss_archive(arch_gene,arch_archive_new,model_save,input,target,criterion):
    loss_arch=0
    
    for i in range(np.int(len(arch_archive_new)/2)):
        model_save_save=copy.deepcopy(model_save)        
        model_weights=get_weights_from_arch(arch_archive_new[2*i:2*i+2])  
        model_save_save=set_model_weights(model_save_save,model_weights)
        
        logits = model_save_save(input)        
        loss=criterion(logits, target)
        loss_arch=(loss_arch+loss.item())
        del model_save_save
    loss_archive=(loss_arch*2)/len(arch_archive_new)
    return loss_archive



def train(train_queue, valid_queue, test_queue, model, architect, arch_archive,n_archive_recent,r_archive_recent, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        #model_save=copy.deepcopy(model)
        model.train()
        #premodel.train
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)
             
        arch_gene1=random_arch_generate()
        arch_gene2=random_arch_generate()
        arch_gene=[arch_gene1,arch_gene2]
        model_weights=get_weights_from_arch(arch_gene)        
        model=set_model_weights(model,model_weights)#####
        
        n_archive_recent.extend([arch_gene1])
        r_archive_recent.extend([arch_gene2])
        n_archive_recent=n_archive_recent[-50:]###############limitate the number for architecture_recent as 50, a.k.a. C=50 in the paper
        r_archive_recent=r_archive_recent[-50:]
        
               
                
        logits = model(input)
        
        loss1=criterion(logits, target)
                
        arch_archive_new=arch_archive_update(arch_gene,arch_archive,n_archive_recent,r_archive_recent)
        
        loss_archive=cal_loss_archive(arch_gene,arch_archive_new,model,input,target,criterion)
        #print(loss_archive)
        
        loss = 0.5*loss1+0.5*loss_archive
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        


        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        #model=set_model_weights(model,arch_param_save)###########################set back

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg,n_archive_recent,r_archive_recent




def infer(valid_queue, model,arch_gen_compa, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    model_weights=get_weights_from_arch(arch_gen_compa)        ###########################
    model=set_model_weights(model,model_weights)#############
    
    

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  #  model=set_model_weights(model,arch_param_save)############################

    return top1.avg, objs.avg





if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model = model.cuda()
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
    
    
train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
split_end = int(num_train)

train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:split_end]),
      pin_memory=True, num_workers=2)

test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

architect = Architect(model, args)


arch_archive=[]
arch_gen1=random_arch_generate()
arch_gen2=random_arch_generate()


n_archive_recent=[arch_gen1]
r_archive_recent=[arch_gen2]

record_train_acc=[]
record_valid_acc=[]
record_valid_acc_retrain=[]



for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)


    # training
    train_acc, train_obj,n_archive_recent,r_archive_recent= train(train_queue, valid_queue, test_queue, model, architect, arch_archive,n_archive_recent,r_archive_recent, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation                           
                           

def random_selection(valid_queue, model, criterion, rounds=None):
    #n_rounds = int(self.B / 7 / 1000)
    if rounds is None:
        n_rounds = 5
    else:
        n_rounds = rounds
    best_rounds = []
    for r in range(n_rounds):
        sample_vals = []
        for _ in range(10):###################set the rounds for random search
           
            arch_gen1=random_arch_generate()
            arch_gen2=random_arch_generate()
            arch_gen_compa=[arch_gen1,arch_gen2]
            
            valid_acc, valid_obj = infer(valid_queue, model,arch_gen_compa, criterion)

            sample_vals.append((arch_gen_compa, valid_acc))
        sample_vals = sorted(sample_vals, key=lambda x:x[1],reverse=True)

        best_rounds.append(sample_vals[0])
    return best_rounds



best_rounds=random_selection(valid_queue, model, criterion, rounds=5)    

print(best_rounds)  
    
