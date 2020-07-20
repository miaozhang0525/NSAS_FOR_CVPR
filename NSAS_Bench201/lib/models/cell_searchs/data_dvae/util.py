from __future__ import print_function
import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import subprocess
import collections
import igraph
import argparse
import pdb
import pygraphviz as pgv
import sys
from PIL import Image

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


'''load and save objects'''
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return


'''Data preprocessing'''




def load_bench102_graphs(name, n_types=11, fmt='igraph', rand_seed=0, with_y=False, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    #max_n = 0  # maximum number of nodes
    with open('%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):

            row,y = eval(row)
            if fmt == 'igraph':
                g, n = decode_bench102_to_igraph(row)
           # max_n = max(max_n, n)
            g_list.append((g, y)) 
    graph_args.num_vertex_type = 8   # original types + add types
    graph_args.max_n = 12  # maximum number of nodes
    graph_args.START_TYPE = 5  # predefined start vertex type
    graph_args.ADD_TYPE = 6 
    graph_args.END_TYPE = 7 # predefined end vertex type



    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    #random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.8)], g_list[int(ng*0.8):int(ng*0.9)], g_list[int(ng*0.9):],graph_args


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def decode_bench102_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = 12
    g = igraph.Graph(directed=True)
    g.add_vertices(n)
    
    g.vs[0]['type'] = 5
    g.vs[1]['type'] = 6
    g.add_edge(0, 1)    
    
    g.vs[2]['type'] =row[0]   
    g.add_edge(1, 2)   
    g.vs[3]['type'] = 6
    g.add_edge(2, 3)

    
    g.vs[4]['type'] = row[1]
    g.vs[5]['type'] = row[2]  
    g.add_edge(1, 4)
    g.add_edge(3, 5)    
    g.vs[6]['type'] = 6      
    g.add_edge(4, 6)
    g.add_edge(5, 6)    
    
    g.vs[7]['type'] = row[3]
    g.vs[8]['type'] = row[4]
    g.vs[9]['type'] = row[5]
    g.vs[10]['type'] = 6    
    g.add_edge(1, 7)
    g.add_edge(3, 8)
    g.add_edge(6, 9)
    g.add_edge(7, 10)      
    g.add_edge(8, 10)
    g.add_edge(9, 10)  
    
    
    g.vs[11]['type'] = 7#####concate node
    g.add_edge(10, 11)   
	# note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n



'''
# some code to test format transformations
row = '[[4], [0, 1], [3, 1, 0], [3, 0, 1, 1], [1, 1, 1, 1, 1], [2, 1, 1, 0, 1, 1], [5, 1, 1, 1, 1, 1, 0], [2, 0, 0, 1, 0, 0, 1, 0]]'
row = '[[2], [2, 0], [4, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0, 1], [3, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 0], [4, 0, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 1, 0, 0, 0, 0], [3, 0, 1, 1, 0, 0, 1, 0, 0, 0], [5, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]'
g, _ = decode_ENAS_to_igraph(row)
string = decode_igraph_to_ENAS(g)
print(row, string)
pdb.set_trace()
pwd = os.getcwd()
os.chdir('software/enas/')
os.system('./scripts/custom_cifar10_macro_final.sh ' + '"' + string + '"')
os.chdir(pwd)
'''


def load_BN_graphs(name, n_types=8, fmt='igraph', rand_seed=0, with_y=True):
    # load raw Bayesian network strings to igraphs or tensors
    g_list = []
    max_n = 0
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_BN_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_BN_to_tensor(row, n_types)
            max_n = max(max_n, n)
            assert(max_n == n)  # all BNs should have the same node number
            g_list.append((g, y)) 

    graph_args.num_class = 1  # how many classes of graphs
    graph_args.num_vertex_type = n_types + 2  # how many vertex types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# classes: %d' % graph_args.num_class)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args



def adjstr_to_BN(row):
    # input: '0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    # output: [[0], [1, 1], [2, 0, 0], [3, 0, 0, 0], [4, 0, 1, 0, 0], [5, 1, 1, 0, 0, 0], [6, 0, 1, 0, 0, 1, 0], [7, 0, 0, 0, 1, 1, 1, 0]]
    matrix = np.array([int(x) for x in row.split()]).reshape(8, 8)
    res = [[0]]
    for i in range(1, 8):
        cur = [i] + matrix[:i, i].tolist()
        res.append(cur)
    return res


def decode_from_latent_space(
        latent_points, model, decode_attempts=500, n_nodes='variable', return_igraph=False, 
        data_type='ENAS'):
    # decode points from the VAE model's latent space multiple attempts
    # and return the most common decoded graphs
    if n_nodes != 'variable':
        check_n_nodes = True  # check whether the decoded graphs have exactly n nodes
    else:
        check_n_nodes = False
    decoded_arcs = []  # a list of lists of igraphs
    pbar = tqdm(range(decode_attempts))
    for i in pbar:
        current_decoded_arcs = model.decode(latent_points)
        decoded_arcs.append(current_decoded_arcs)
        pbar.set_description("Decoding attempts {}/{}".format(i, decode_attempts))

    # We see which ones are decoded to be valid architectures
    valid_arcs = []  # a list of lists of strings
    if return_igraph:
        str2igraph = {}  # map strings to igraphs
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_arcs.append([])
        for j in range(decode_attempts):
            arc = decoded_arcs[j][i]  # arc is an igraph
            if data_type == 'ENAS':
                if is_valid_ENAS(arc, model.START_TYPE, model.END_TYPE):
                    if not check_n_nodes or check_n_nodes and arc.vcount() == n_nodes:
                        cur = decode_igraph_to_ENAS(arc)  # a flat ENAS string
                        if return_igraph:
                            str2igraph[cur] = arc
                        valid_arcs[i].append(cur)
            elif data_type == 'BN':  
                if is_valid_BN(arc, model.START_TYPE, model.END_TYPE, nvt=model.nvt):
                    cur = decode_igraph_to_BN_adj(arc)  # a flat BN adjacency matrix string
                    if return_igraph:
                        str2igraph[cur] = arc
                    valid_arcs[i].append(cur)
        pbar.set_description("Check validity for {}/{}".format(i, latent_points.shape[0]))

    # select the most common decoding as the final architecture
    final_arcs = []  # a list of lists of strings
    pbar = tqdm(range(latent_points.shape[ 0 ]))
    for i in pbar:
        valid_curs = valid_arcs[i]
        aux = collections.Counter(valid_curs)
        if len(aux) > 0:
            arc, num_arc = list(aux.items())[np.argmax(aux.values())]
        else:
            arc = None
            num_arc = 0
        final_arcs.append(arc)
        pbar.set_description("Latent point {}'s most common decoding ratio: {}/{}".format(
                             i, num_arc, len(valid_curs)))

    if return_igraph:
        final_arcs_igraph = [str2igraph[x] if x is not None else None for x in final_arcs]
        return final_arcs_igraph, final_arcs
    return final_arcs


'''Network visualization'''
def plot_DAG(g, res_dir, name, backbone=False, data_type='ENAS', pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name = os.path.join(res_dir, name+'.pdf')
    if data_type == 'ENAS':
        draw_network(g, file_name, backbone)
    elif data_type == 'BN':
        draw_BN(g, file_name)
    return file_name


def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 0:  
        label = 'input'
        color = 'skyblue'
    elif label == 1:
        label = 'output'
        color = 'pink'
    elif label == 2:
        label = 'conv3'
        color = 'yellow'
    elif label == 3:
        label = 'sep3'
        color = 'orange'
    elif label == 4:
        label = 'conv5'
        color = 'greenyellow'
    elif label == 5:
        label = 'sep5'
        color = 'seagreen3'
    elif label == 6:
        label = 'avg3'
        color = 'azure'
    elif label == 7:
        label = 'max3'
        color = 'beige'
    else:
        label = ''
        color = 'aliceblue'
    #label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)


def draw_BN(g, path):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    label_dict = dict(zip(range(2, 10), 'ASTLBEXD'))
    pos_dict = dict(zip(range(2, 10), ['0, 3!', '2.75, 3!', '0, 2!', '2, 2!', '3.5, 1!', '1.5, 1!', '1.5, 0!', '3.5, 0!']))

    def add_node(graph, node_id, label, shape='circle', style='filled'):
        if label in {0, 1}:
            return
        else:
            label, pos = label_dict[label], pos_dict[label]
        graph.add_node(
                node_id, label=label, color='black', fillcolor='white',
                shape=shape, style=style, pos=pos, fontsize=27,
        )
        return

    if g is None:
        graph.add_node(
                0, label='invalid', color='black', fillcolor='white', 
                shape='box', style='filled', 
        )
        graph.layout(prog='dot')
        graph.draw(path)
        return

    for idx in range(1, g.vcount()-1):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(1, g.vcount()-1):
        for node in g.get_adjlist(igraph.IN)[idx]:
            #if node != g.vcount()-1 and node != 0:  # we don't draw input/output nodes for BN
            node_type = g.vs[node]['type']
            if node_type != 0 and node_type != 1:  # we don't draw input/output nodes for BN
                graph.add_edge(node, idx)

    graph.layout()
    graph.draw(path)
    return path


'''Validity and novelty functions'''
def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True


def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    res = g.is_dag()
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.indegree() == 0 and v['type'] != START_TYPE:
            return False
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1


def is_valid_ENAS(g, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    # in addition, node i must connect to node i+1
    for i in range(g.vcount()-2):
        res = res and g.are_connected(i, i+1)
        if not res:
            return res
    # the output node n must not have edges other than from n-1
    res = res and (g.vs[g.vcount()-1].indegree() == 1)
    return res
    

def is_valid_BN(g, START_TYPE=0, END_TYPE=1, nvt=10):
    # nvt: number of vertex types in this BN
    # first need to be a DAG
    res = g.is_dag()
    # check whether start and end types only appear once
    # BN nodes need not be connected
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
    res = res and n_start == 1 and n_end == 1
    # in addition, every type must appear exactly once
    res = res and (len(set(g.vs['type'])) == nvt) and g.vcount() == nvt
    return res


'''Other util functions'''
def combine_figs_horizontally(names, new_name):
    images = list(map(Image.open, names))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(new_name)



class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))


