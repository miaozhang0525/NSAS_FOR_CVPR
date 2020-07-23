import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from architect import Architect

from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype

import genotypes

import copy
import gc

import data
import model_search as model

import inspect

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, args)
search_data = batchify(corpus.valid, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


ntokens = len(corpus.dictionary)
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast, 
                       args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute)

size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')
logging.info(model.genotype())

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))

def set_model_arch(model, arch):
    for rnn in model.rnns:
        rnn.genotype = arch
        
def set_model_weights(model, weights):
    model.weights = Variable(weights.cuda(), requires_grad=True)
    model._arch_parameters = [model.weights]
    

def random_arch_gene():
    n_nodes = genotypes.STEPS
    n_ops = len(genotypes.PRIMITIVES)
    arch_gene = []
    arch=[]
    for i in range(n_nodes):
        op = np.random.choice(range(1,n_ops))
        node_in = np.random.choice(range(i+1))
        arch_gene.append((op, node_in))
        arch.append((genotypes.PRIMITIVES[op], node_in))
    concat = range(1,9)
    genotype = genotypes.Genotype(recurrent=arch, concat=concat)
    return arch_gene,arch



def _genotype(weights):
    STEPS=8
    def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
            end = start + i + 1
            W = probs[start:end].copy()
            j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((PRIMITIVES[k_best], j))
            start = end
        return gene

    gene = _parse(F.softmax(weights, dim=-1).data.cpu().numpy())
    genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
    return genotype



def _parse_gene(weights):
    STEPS=8
    gene = []
    start = 0
    weights_t=F.softmax(weights, dim=-1).data.cpu().numpy()
    for i in range(STEPS):
        end = start + i + 1
        W = weights_t[start:end].copy()
        j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
        k_best = None
        for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
        gene.append((k_best, j))
        start = end
    return gene

from torch.autograd import Variable



def get_weights_from_arch(arch_gene):
    n_nodes = 8
    n_ops = 5
    weights = torch.zeros(sum([i+1 for i in range(n_nodes)]), n_ops)

    offset = 0
    for i in range(n_nodes):
        op = arch_gene[i][0]
        node_in = arch_gene[i][1]
        ind = offset + node_in
        weights[ind, op] = 5
        offset += (i+1)

    weights = torch.autograd.Variable(weights.cuda(), requires_grad=False)

    return weights


def random_arch_generate():
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = 4####model._step

    arch_gene = []
    for i in range(n_nodes):
        ops = np.random.choice(range(num_ops), 2)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        arch_gene.extend([(ops[0],nodes_in_normal[0]), (ops[1],nodes_in_normal[1])])
    return arch_gene  

def cal_arch_dis(arch1,arch2):#################small distance more similar
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



def cal_diver_score(arch,archive):
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


def find_similar_arch(arch,archive):
    dis=np.zeros(len(archive))   
    
    for i in range(len(archive)):
        dis[i]=cal_arch_dis(arch,archive[i])################

    m=np.argsort(dis)
    index=m[0]
    
    return index



def arch_archive_update(arch_gene,arch_archive,archive_recent):
    store_num=8
    if len(arch_archive)==store_num:
        ind_arch_norm_replace=find_similar_arch(arch_gene,arch_archive)
        arch_archive[ind_arch_norm_replace]=diver_arch_replace(ind_arch_norm_replace,arch_archive,archive_recent)        
    else:
        normal_arch=diver_arch_generate(arch_archive)
        arch_archive.append(normal_arch)
    return arch_archive



def cal_loss_archive(arch_gene,arch_archive_new,model_save,input,target,criterion):
    loss_arch=0
    
    for i in range(np.int(len(arch_archive_new)/2)):
        w1=1-cal_arch_dis(arch_gene[0],arch_archive_new[2*i])##############################
        w2=1-cal_arch_dis(arch_gene[1],arch_archive_new[2*i+1])
        w=(w1+w2)/2
        model_save_save=copy.deepcopy(model_save)        
        model_weights=get_weights_from_arch(arch_archive_new[2*i:2*i+2])  
        model_save_save=set_model_weights(model_save_save,model_weights)
        
        logits = model_save_save(input)        
        loss=criterion(logits, target)
        loss_arch=w*(loss_arch+loss.item())
        del model_save_save
    loss_archive=(loss_arch*2)/len(arch_archive_new)
    del model_save
    return loss_archive
    
    
def set_model_arch(model, arch):
    for rnn in model.rnns:
        rnn.genotype = arch
        
def set_model_weights(model, weights):
    model.weights = Variable(weights.cuda(), requires_grad=True)
    model._arch_parameters = [model.weights]
    

def random_arch_gene():
    n_nodes = genotypes.STEPS
    n_ops = len(genotypes.PRIMITIVES)
    arch_gene = []
    arch=[]
    for i in range(n_nodes):
        op = np.random.choice(range(1,n_ops))
        node_in = np.random.choice(range(i+1))
        arch_gene.append((op, node_in))
        arch.append((genotypes.PRIMITIVES[op], node_in))
    concat = range(1,9)
    genotype = genotypes.Genotype(recurrent=arch, concat=concat)
    return arch_gene,arch



def _genotype(weights):
    STEPS=8
    def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
            end = start + i + 1
            W = probs[start:end].copy()
            j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((PRIMITIVES[k_best], j))
            start = end
        return gene

    gene = _parse(F.softmax(weights, dim=-1).data.cpu().numpy())
    genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
    return genotype



def _parse_gene(weights):
    STEPS=8
    gene = []
    start = 0
    weights_t=F.softmax(weights, dim=-1).data.cpu().numpy()
    for i in range(STEPS):
        end = start + i + 1
        W = weights_t[start:end].copy()
        j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
        k_best = None
        for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
        gene.append((k_best, j))
        start = end
    return gene

from torch.autograd import Variable

def get_arch_from_gene(arch_gene):
    n_nodes = 8
    n_ops = 5 
    arch=[]
    for i in range(n_nodes):
        op = arch_gene[i][0]
        node_in = arch_gene[i][1]    
        arch.append((genotypes.PRIMITIVES[op], node_in))
    concat = range(1,9)
    genotype = genotypes.Genotype(recurrent=arch, concat=concat)
    
    return genotype


def get_weights_from_arch(arch_gene):
    n_nodes = 8
    n_ops = 5
    weights = torch.zeros(sum([i+1 for i in range(n_nodes)]), n_ops)

    offset = 0
    for i in range(n_nodes):
        op = arch_gene[i][0]
        node_in = arch_gene[i][1]
        ind = offset + node_in
        weights[ind, op] = 5
        offset += (i+1)

    weights = torch.autograd.Variable(weights.cuda(), requires_grad=False)

    return weights




def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(model,epoch,arch_archive,archive_recent):
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'
        
    #temp= opt.initial_temp * np.exp(-opt.anneal_rate * epoch)
    temp= 2.5 * np.exp(-0.00003* epoch)
    #temperature=torch.FloatTensor([temp])
    
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            
            weights_save=model.weights
            
            #arch,weights_gumble=sample_arch_from_weights(weights_save,temp)  
            #arch_gene=_parse_gene(weights_gumble)
            
            arch_gene,arch=random_arch_gene()                   
            weight_arch=get_weights_from_arch(arch_gene)
            
            set_model_arch(model, arch)
            set_model_weights(model, weight_arch)
            
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
            cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.


            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            
            archive_recent.extend([arch_gene])
            archive_recent=archive_recent[-20:]
            
            arch_archive=arch_archive_update(arch_gene,arch_archive,archive_recent)
            
            loss_sr=0
            for i in range(len(arch_archive)):
                model_save_r=copy.deepcopy(model)
                
                arch_gene_r=arch_archive[i]     
                weight_arch_r=get_weights_from_arch(arch_gene_r)
                arch_r=get_arch_from_gene(arch_gene_r)


                set_model_arch(model_save_r, arch_r)
                set_model_weights(model_save_r, weight_arch_r)

                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
                cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

                optimizer.zero_grad()
                hidden[s_id] = repackage_hidden(hidden[s_id])

                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                loss_r = raw_loss
                # Activiation Regularization
                if args.alpha > 0:
                    loss_r = loss_r + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                loss_r = loss_r + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss_r *= args.small_batch_size / args.batch_size
                
                loss_sr=loss_r+loss_r
            loss_sr=loss_sr/(len(arch_archive))
            
            loss_f=0.5*loss+0.5*loss_sr
       
            
            loss_f.backward()

            
            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            logging.info(parallel_model.genotype())
            #print(F.softmax(parallel_model.weights, dim=-1))
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len
        
        
lr = args.lr
best_val_loss = []
stored_loss = 100000000

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    
    
stored_loss = 100000000
archive_recent=[]
arch_archive=[]
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(model,epoch,arch_archive,archive_recent)

    val_loss = evaluate(val_data, eval_batch_size)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    logging.info('-' * 89)

    if val_loss < stored_loss:
        save_checkpoint(model, optimizer, epoch, args.save)
        logging.info('Saving Normal!')
        stored_loss = val_loss

    best_val_loss.append(val_loss)    

