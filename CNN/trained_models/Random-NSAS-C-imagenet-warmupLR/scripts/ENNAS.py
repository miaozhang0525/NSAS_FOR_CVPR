#######code for directly calculate distance between stored architectures(last 1000)
import sys
import genotypes
from model_search import Network
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import shutil
import inspect
import pickle
import argparse

from numpy.linalg import cholesky

import genotypes

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, epochs, resume_iter=None, init_channels=16):
        args = {}
        args['data'] = '/data/mzhang3/randomNAS_own/data'
        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        args['layers'] = 8
        args['drop_path_prob'] = 0.3
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.5
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = True
        args['cutout'] = False
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)


        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=32,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        model = Network(args.init_channels, 10, args.layers, self.criterion)

        model = model.cuda()
        self.model = model

     #   try:
        #    self.load()
      #      logging.info('loaded previously saved weights')
      #  except Exception as e:
      #      print(e)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_iter is not None:
            self.steps = resume_iter
            self.epochs = int(resume_iter / len(self.train_queue))
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epochs):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch):
        args = self.args
        if self.steps % len(self.train_queue) == 0:
            
            self.scheduler.step()
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
        lr = self.scheduler.get_lr()[0]

        weights = self.get_weights_from_arch(arch)
        self.set_model_weights(weights)

        step = self.steps % len(self.train_queue)
        input, target = next(self.train_iter)

        self.model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

      # get a random minibatch from the search queue with replacement
        self.optimizer.zero_grad()
        logits = self.model(input)
        loss = self.criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
        self.optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        self.objs.update(loss.data, n)
        self.top1.update(prec1.data, n)
        self.top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

        self.steps += 1
        if self.steps % len(self.train_queue) == 0:
            self.epochs += 1
            self.train_iter = iter(self.train_queue)
            valid_err = self.evaluate(arch)
            logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1-valid_err))
            self.save()

    def evaluate(self, arch, split=None):
      # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_model_weights(weights)

        self.model.eval()

        if split is None:
            n_batches = 10
        else:
            n_batches = len(self.valid_queue)

        for step in range(n_batches):
            try:
                input, target = next(self.valid_iter)
            except Exception as e:
                logging.info('looping back over valid set')
                self.valid_iter = iter(self.valid_queue)
                input, target = next(self.valid_iter)
            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

            logits = self.model(input)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % self.args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1-(top1.avg)/100

    def save(self):
        utils.save(self.model, os.path.join(self.args.save, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save, 'weights.pt'))

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
        alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

        offset = 0
        for i in range(n_nodes):
            normal1 = np.int_(arch[0][2*i])
            normal2 = np.int_(arch[0][2*i+1])
            reduce1 = np.int_(arch[1][2*i])
            reduce2 = np.int_(arch[1][2*i+1])
            alphas_normal[offset+normal1[0], normal1[1]] = 1
            alphas_normal[offset+normal2[0], normal2[1]] = 1
            alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
            alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
            offset += (i+2)

        arch_parameters = [
          alphas_normal,
          alphas_reduce,
        ]
        return arch_parameters

    def set_model_weights(self, weights):
        self.model.alphas_normal = weights[0]
        self.model.alphas_reduce = weights[1]
        self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def novelty_fitness(self,arch,store_arch,k):
        def dis_arch(arch1,arch2):
            dis=8
            n_nodes=genotypes.STEPS

            for i in range(n_nodes):
                if arch1[2*i,]==arch2[2*i,] and arch1[2*i+1,]==arch2[2*i+1,]:
                    dis=dis-1
            dis=dis/8
            return dis     

        dis=np.zeros((store_arch.shape[0]))
        for i in range(store_arch.shape[0]):
            dis[i]=dis_arch(arch,store_arch[i,])
        sort_dis=np.sort(dis)
        novelty_dis=np.mean(sort_dis[0:k])
        
        return novelty_dis         

    def sample_arch_eval(self):
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        normal = []
        reduction = []
        for i in range(n_nodes):
            ops = np.random.choice(range(num_ops), 4)
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)    
        
    def sample_arch(self,node_id,store_normal_arch,store_reduce_arch):
        num_ops = len(genotypes.PRIMITIVES)
        num_nodes=self.model._steps
        
            
        def limite_range(arch,num_ops,num_nodes):
            for i in range(num_nodes):
                arch[4*i,]=np.max((np.min((arch[4*i,],(i+1))),0))
                arch[4*i+1,]=np.max((np.min((arch[4*i+1,],num_ops-1)),0))
                arch[4*i+2,]=np.max((np.min((arch[4*i+2,],(i+1))),0))
                arch[4*i+3,]=np.max((np.min((arch[4*i+3,],num_ops-1)),0))
            return arch     
        def get_performance(self,selec_arch):
            n_nodes = genotypes.STEPS   
            normal = []
            reduction = []
            performance=np.zeros((1))
		    # selec_arch=np.zeros((2*n_nodes,))
            for i in range(n_nodes):
                normal.extend([(selec_arch[4*i], selec_arch[4*i+1]), (selec_arch[4*i+2], selec_arch[4*i+3])])
                reduction.extend([(selec_arch[4*i+4*n_nodes], selec_arch[4*i+1+4*n_nodes]), (selec_arch[4*i+2+4*n_nodes], selec_arch[4*i+3+4*n_nodes])])
			   

            arch=(normal, reduction) 
            performance[0,]=self.evaluate(arch)       
            return performance[0,]    
            
        if node_id>999:
            alfa=0.01
            n=10
            sigma=1

            mu=np.zeros((1,4*self.model._steps))
            Sigma=np.eye(4*self.model._steps)
            R=cholesky(Sigma)
            
            
            yita=np.dot(np.random.randn(n,4*self.model._steps),R)+mu
            n_yita=np.empty((n,4*self.model._steps))
            n_yita1=np.empty((n,4*num_nodes))
                        
            index0=np.random.randint(1000)
            test_normal_arch=store_normal_arch[index0,]
            test_reduce_arch=store_reduce_arch[index0,]
            
            for i in range(n):
                n_f=self.novelty_fitness(np.int_(np.round((test_normal_arch+yita[i,]))),np.int_(np.round(store_normal_arch)),10)
                n_yita[i,]=n_f*yita[i,]
                select_i=limite_range((test_normal_arch+yita[i,]),num_ops,num_nodes)
                test_arch1=np.hstack((select_i,test_reduce_arch))
               # gf=get_performance(self,test_arch1)    ######whether take the reward into consideration
              #  n_yita1[i,]=gf*yita[i,]                 ####################whether take the reward into consideration

            #selec_normal=test_normal_arch+alfa*(1/(n*sigma))*(0.5*sum(n_yita)+0.5*sum(n_yita1))################# whether take the reward into consideration
            selec_normal=test_normal_arch+alfa*(1/(n*sigma))*sum(n_yita)

            store_normal_arch[index0,]=selec_normal             
            selec_normal=np.int_(np.round(selec_normal))            
            selec_normal=limite_range(selec_normal,num_ops,num_nodes)

            
            
            
            yita=np.dot(np.random.randn(n,4*self.model._steps),R)+mu
            n_yita=np.empty((n,4*self.model._steps))
            
            index1=np.random.randint(1000)
            test_normal_arch=store_normal_arch[index0,]
            test_reduce_arch=store_reduce_arch[index0,]
            
            
          
            for i in range(n):
                n_f=self.novelty_fitness(np.int_(np.round((test_reduce_arch+yita[i,]))),np.int_(np.round(store_reduce_arch)),10)
                n_yita[i,]=n_f*yita[i,]
                select_i=limite_range((test_reduce_arch+yita[i,]),num_ops,num_nodes)
                test_arch2=np.hstack((test_normal_arch,select_i))
               # n_yita1[i,]=get_performance(self,test_arch2)*yita[i,]######whether take the reward into consideration
           # selec_reduce=test_reduce_arch+alfa*(1/(n*sigma))*(0.5*sum(n_yita)+0.5*sum(n_yita1))######whether take the reward into consideration
            selec_reduce=test_reduce_arch+alfa*(1/(n*sigma))*sum(n_yita)
            store_reduce_arch[index1,]=selec_reduce      
            selec_reduce=np.int_(np.round(selec_reduce))                
            selec_reduce=limite_range(selec_reduce,num_ops,num_nodes)
            
          
            normal = []
            reduction = []
            for i in range(self.model._steps):
                s1=np.int(selec_normal[4*i,])
                s2=np.int(selec_normal[4*i+1,])
                s3=np.int(selec_normal[4*i+2,])
                s4=np.int(selec_normal[4*i+3,])
                s5=np.int(selec_reduce[4*i,])
                s6=np.int(selec_reduce[4*i+1,])
                s7=np.int(selec_reduce[4*i+2,])
                s8=np.int(selec_reduce[4*i+3,])
                normal.extend([(s1,s2), (s3,s4)])
                reduction.extend([(s5,s6), (s7,s8)]) 
            index=(index0,index1)
                                                   
        else:     
            num_ops = len(genotypes.PRIMITIVES)
            n_nodes = self.model._steps

            normal = []
            reduction = []
            for i in range(n_nodes):
                ops = np.random.choice(range(num_ops), 4)
                nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)                
                nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)                
                normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
                
                reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
                
            normal=np.int_(normal)
            reduction=np.int_(reduction)
            index=(node_id,node_id)
######the operations from two previous node are different
        return index, (normal, reduction)


    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(genotypes.PRIMITIVES)

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch
class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
     #  self.objective_val = self.model.evaluate(arch)  
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed
        self.iters = 0

        self.arms = {}
        size_arch=self.model.model._steps*4
               
        self.store_normal_arch=np.empty((1000,size_arch))
        self.store_reduce_arch=np.zeros((1000,size_arch))
        
        
        self.node_id = 0

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n,self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n],'objective_val')]
        objective_vals = sorted(objective_vals,key=lambda x:x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)


    def get_arch(self):####need to generate architecture based on novelty    
        inde, arch = self.model.sample_arch(self.node_id,self.store_normal_arch,self.store_reduce_arch)
        normal_arch=np.array(arch[0])
        reduce_arch=np.array(arch[1])
        
        gene_len=self.model.model._steps*4
        
        normal_arch=np.reshape(normal_arch,(1,gene_len))
        reduce_arch=np.reshape(reduce_arch,(1,gene_len))
        self.store_normal_arch[inde[0],]=normal_arch[0,]
        self.store_reduce_arch[inde[1],]=reduce_arch[0,]
        
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)

        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        while self.iters < self.B:
            arch = self.get_arch()#######################need to generate architecture based on novelty
            self.model.train_batch(arch)
            self.iters += 1

            if self.iters % 500 == 0:
                self.save()
        self.save()

              
        
    def get_eval_arch(self, rounds=None):
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch_eval()
                try:
                    a=time.perf_counter()
                    ppl = self.model.evaluate(arch) 
                    torch.cuda.synchronize()
                    b=time.perf_counter()
                    print(b-a)
                    print('kkkkkkkkkkkkkkkkkk')
                except Exception as e:
                    ppl = 1000000
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))
            sample_vals = sorted(sample_vals, key=lambda x:x[1])

            full_vals = []
            if 'split' in inspect.getargspec(self.model.evaluate).args:
                for i in range(10):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x:x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])
        return best_rounds
    
    
    def EA_arch_search(self,num_pop,num_ite,num_cross,num_mutation):

        def get_init_pop(self,num_pop,n_nodes):
            pop=np.empty((num_pop,8*n_nodes))
            fitness=np.zeros((num_pop,))
            for m in range(num_pop):         
                num_ops = len(genotypes.PRIMITIVES)
                normal = []
                reduction = []
                for i in range(n_nodes):
                    ops = np.random.choice(range(num_ops), 4)
                    nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
                    nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
                    normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
                    reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
                    pop[m,4*i]=nodes_in_normal[0]
                    pop[m,4*i+1]=ops[0]
                    pop[m,4*i+2]=nodes_in_normal[1]
                    pop[m,4*i+3]=ops[1]
                    pop[m,4*i+4*n_nodes]=nodes_in_reduce[0]
                    pop[m,4*i+1+4*n_nodes]=ops[2]
                    pop[m,4*i+2+4*n_nodes]=nodes_in_reduce[1]
                    pop[m,4*i+3+4*n_nodes]=ops[3]                            
                arch=(normal, reduction) 
                fitness[m,]=self.model.evaluate(arch)          
            return pop,fitness


        def corssover(self,pop,fitness,num_cross):
            index=np.argsort(fitness)
            pop_select=pop[index[0:num_cross],]


            inde_cross=np.arange(num_cross)
            np.random.shuffle(inde_cross)
            pop_select=pop_select[inde_cross,]
            pop_cross=np.empty((num_cross,pop.shape[1]))


            for i in range(np.int(num_cross/2)):
                cross1=pop_select[2*i,]
                cross2=pop_select[2*i+1,]

                cross_points=np.arange(4*self.model.model._steps)
                np.random.shuffle(cross_points)
                cross_points=cross_points[0:2]
                cross_points=np.sort(cross_points)
                p1=2*cross_points[0]
                p2=2*cross_points[1]

                cross1_=cross1
                cross2_=cross2

                cross1_[p1:p2]=cross2[p1:p2]
                cross2_[p1:p2]=cross1[p1:p2]

                pop_cross[2*i,]= cross1_       
                pop_cross[2*i+1,]= cross2_   

            return pop_cross


        def mutation(self,pop,fitness,num_mutation):
            index=np.argsort(fitness)
            pop_select=pop[index[0:num_mutation],]
            pop_mutation=np.empty((num_mutation,pop.shape[1]))
            num_ops = len(genotypes.PRIMITIVES)


            for i in range(num_mutation):
                pop_mutation[i,]=pop_select[i,]

                for j in range(pop.shape[1]):
                    if j>((pop.shape[1])/2-1):
                        q=j-(pop.shape[1])/2
                    else:
                        q=j
                    m=q//4+2
                    if np.random.rand()<0.2:#################genes with mutation probability 0.2
                        if j%2==0:
                            pop_mutation[i,j]=np.random.randint(m)
                        else:
                            pop_mutation[i,j]=np.random.randint(num_ops)            
            return pop_mutation


        def get_fitness(self,pop):
            num_pop=pop.shape[0]
            fitness=np.zeros((num_pop))
            for m in range(num_pop):
                indiv=pop[m,]
                normal=[]
                reduction=[]
                for i in range(self.model.model._steps):
                    s1=np.int(indiv[4*i,])
                    s2=np.int(indiv[4*i+1,])
                    s3=np.int(indiv[4*i+2,])
                    s4=np.int(indiv[4*i+3,])
                    s5=np.int(indiv[4*i+16,])
                    s6=np.int(indiv[4*i+1+16,])
                    s7=np.int(indiv[4*i+2+16,])
                    s8=np.int(indiv[4*i+3+16,])
                    normal.extend([(s1,s2), (s3,s4)])
                    reduction.extend([(s5,s6), (s7,s8)]) 
                arch=(normal, reduction)
                fitness[m,]=self.model.evaluate(arch)  

            return fitness


        n_nodes = self.model.model._steps    

        pop,fitness=get_init_pop(self,num_pop,n_nodes)

        for it in range(num_ite):
            pop_cross=corssover(self,pop,fitness,num_cross)
            fitness_cross=get_fitness(self,pop_cross)
            pop_mutate=mutation(self,pop,fitness,num_mutation)
            fitness_mutate=get_fitness(self,pop_mutate) 
            pop_comb=np.concatenate((pop,pop_cross,pop_mutate),axis=0)
            fitness_comb=np.concatenate((fitness,fitness_cross,fitness_mutate),axis=0)
            index=np.argsort(fitness_comb)
            pop_comb=pop_comb[index,]
            pop=pop_comb[0:num_pop,]
            fitness=fitness_comb[0:num_pop,]

        index=np.argsort(fitness)
        indi_final=pop[index[0],]
        
        normal = []
        normal_struc=[]
        reduction = []
        reduction_struc=[]
        for i in range(self.model.model._steps):

            s1=np.int(indi_final[4*i,])
            s2=np.int(indi_final[4*i+1,])
            s3=np.int(indi_final[4*i+2,])
            s4=np.int(indi_final[4*i+3,])
            s5=np.int(indi_final[4*i+16,])
            s6=np.int(indi_final[4*i+1+16,])
            s7=np.int(indi_final[4*i+2+16,])
            s8=np.int(indi_final[4*i+3+16,])
            normal.extend([(s1,s2), (s3,s4)])
            normal_struc.append((s1, genotypes.PRIMITIVES[s2]))
            normal_struc.append((s3, genotypes.PRIMITIVES[s4]))
            
            reduction.extend([(s5,s6), (s7,s8)])            
            reduction_struc.append((s5, genotypes.PRIMITIVES[s6]))
            reduction_struc.append((s7, genotypes.PRIMITIVES[s8]))
        
        concat = range(2, self.model.model._steps+2)
        genotype = genotypes.Genotype(normal=normal_struc, normal_concat=concat,reduce=reduction_struc, reduce_concat=concat)
        best_arch=genotype

        return indi_final

sys.argv=['']; del sys
parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
parser.add_argument('--seed', dest='seed', type=int, default=100)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5)
parser.add_argument('--save_dir', dest='save_dir', type=str, default='/data/mzhang3/randomNAS_release-master/results')
parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
# PTB only argument. config=search uses proxy network for shared weights while
# config=eval uses proxyless network for shared weights.
parser.add_argument('--config', dest='config', type=str, default="search")
# CIFAR-10 only argument.  Use either 16 or 24 for the settings for random search
# with weight-sharing used in our experiments.
parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
args = parser.parse_args()
import sys 
    
# Fill in with root output path
root_dir = '/data/mzhang3/randomNAS_release-master/results'
if args.save_dir is None:
    save_dir = os.path.join(root_dir, '%s/random/trial%d' % (args.benchmark, args.seed))
else:
    save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.eval_only:
    assert args.save_dir is not None

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(args)

if args.benchmark=='ptb':
    data_size = 929589
    time_steps = 35
else:
    data_size = 25000
    time_steps = 1
B = int(args.epochs * data_size / args.batch_size / time_steps)
model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs, init_channels=args.init_channels)

searcher = Random_NAS(B, model, args.seed, save_dir)
logging.info('budget: %d' % (searcher.B))
if not args.eval_only:
    searcher.run()
    #archs = searcher.get_eval_arch()###random search based model selection
    archs1 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection
    archs2 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection
    archs3 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection
    archs4 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection
    archs5 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection
    archs6 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)###evolutionart algorithm based model selection


else:
    np.random.seed(args.seed+1)
    archs = searcher.get_eval_arch(2)
logging.info(archs)
arch = ' '.join([str(a) for a in archs[0][0]])
with open('/tmp/arch','w') as f:
    f.write(arch)

       
print(archs1)
print(archs2)
print(archs3)
print(archs4)
print(archs5)
print(archs6)
