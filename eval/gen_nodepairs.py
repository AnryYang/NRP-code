#########################################################################
# File Name: graphreconstruc_util.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Tue 13 Nov 2018 01:48:15 PM +08
#########################################################################
#!/usr/bin/env/ python

import os
import operator
import graph_util as gutil
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import random
import multiprocessing
import argparse
import os
import os.path

def sample(data, ratio):
    fattr = '../data/%s/attr.txt'%data
    fgraph = '../data/%s/edgelist.txt'%data
    filename = '../data/%s/nodepairs.txt'%data

    n, m, directed = gutil.loadGraphAttr(fattr)

    G = gutil.loadGraphFromEdgeListTxt(fgraph, directed)

    if directed==False:
        m = m*2

    total_num = n * (n-1)
    print "real edge ratio: %f" % (m*1.0/total_num)
    num = int( total_num * ratio)
    print "sampling %d node pairs and writing to file: %s" % (num, filename)
    num_p = 0
    with open(filename, 'w') as fout:
        for i in range(num):
            u = random.randint(0,n-1)
            v = random.randint(0,n-1)
            p = 0
            if G.has_edge(u, v):
                p = 1
                num_p += 1
            fout.write(str(u)+" "+str(v)+" "+str(p)+"\n")
    print "positive samples: ", num_p
                

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

PARALLEL_LEVEL = multiprocessing.cpu_count()

print("CPU count: %d"%PARALLEL_LEVEL)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--ratio', type=float, default=0.01, help='test data ratio')
    args = parser.parse_args()

    sample(args.data, args.ratio)
