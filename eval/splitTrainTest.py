#########################################################################
# File Name: splitTrainTest.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Tue 13 Nov 2018 02:29:51 PM +08
#########################################################################
#!/usr/bin/env/ python

import argparse
import os
import os.path
import graph_util as gutil

parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--ratio', type=float, default=0.3, help='test data ratio')
parser.add_argument('--action', type=str, default="split", help='action: "split" or "select" ')
args = parser.parse_args()

fgraph = '../data/%s/edgelist.txt'%args.data
fattr = '../data/%s/attr.txt'%args.data
ftraingraph = '../data/%s/edgelist.train.txt'%args.data
ftestgraph = '../data/%s/edgelist.test.txt'%args.data
fnegativegraph = '../data/%s/edgelist.negative.txt'%args.data


if args.action=="select":
    n, m, directed = gutil.loadGraphAttr(fattr)
    print "graph attributes: n=%d, m=%d, directed=%r"%(n, m, directed)
    num_test_edge = int(m*args.ratio)
    print "selecting negative edges..."
    negative_edges = gutil.selectNegativeEdges(fgraph, n, num_test_edge, directed)

    print "writing negative edges..."
    with open(fnegativegraph, 'w') as fout:
        for (u ,v) in negative_edges:
            fout.write(str(u)+" "+str(v)+"\n")
elif args.action=="split":
    n, m, directed = gutil.loadGraphAttr(fattr)
    print "graph attributes: n=%d, m=%d, directed=%r"%(n, m, directed)

    num_test_edge = int(m*args.ratio)
    train_edges, test_edges = gutil.splitDiGraphToTrainTest(fgraph, num_test_edge)
    print "train edges: %d, test edges: %d" % (len(train_edges), len(test_edges))

    print "writing %s"%ftestgraph
    with open(ftestgraph, 'w') as fout:
        for (s,t) in test_edges:
            fout.write(str(s)+" "+str(t)+"\n")

    print "writing %s"%ftraingraph
    with open(ftraingraph, 'w') as fout:
        for (s,t) in train_edges:
            fout.write(str(s)+" "+str(t)+"\n")
else:
    print "wrong action!"
