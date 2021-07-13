#########################################################################
# File Name: linkpred_util.py
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
from scipy import spatial
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class LinkPredictionEval:
    def __init__(self, data, algo, d):
        self.embed = gutil.Embeddding(data, algo, d, full=False)
    
    def evalWithClassifier(self):
        # classifier = LogisticRegression(random_state=0, solver='lbfgs')
        # classifier.fit(X_train, Y_train)

        embX = self.embed.get_combined_embedding()
        X_test = []
        Y_true = []
        for (u, v) in self.embed.get_test_edges():
            if self.embed.directed:
                X_test.append( np.hstack([embX[u,:], embX[v,:]]) )
            else:
                X_test.append( embX[u,:] * embX[v,:] )
            Y_true.append(1)

        for (u, v) in self.embed.get_negative_edges():
            if self.embed.directed:
                X_test.append( np.hstack([embX[u,:], embX[v,:]]) )
            else:
                X_test.append( embX[u,:] * embX[v,:] )
            Y_true.append(0)

        print "training classifier..."
        classifier = LogisticRegression(random_state=0, solver='lbfgs')
        classifier.fit(X_test, Y_true)

        Y_pred = classifier.predict(X_test)
        auc_score = roc_auc_score(Y_true, Y_pred)
        print "AUC: %f" % auc_score
        
    def eval(self):
        if (self.embed.algo == gutil.VERSE and self.embed.directed==True) or (self.embed.algo == gutil.PRUNE) or (self.embed.algo == gutil.GRAPHGAN) or (self.embed.algo == gutil.GRAPHWAVE) or (self.embed.algo == gutil.LINE) or (self.embed.algo == gutil.N2V) or (self.embed.algo == gutil.NETSMF) or (self.embed.algo == gutil.PRONE):# or (self.embed.algo == gutil.DNGR):
            self.evalWithClassifier()
            return
        
        pred_node_score = {}
        pred_test = []
        i = 0
        labels = []
        for (u, v) in self.embed.get_test_edges():
            pred_score = self.embed.link_prob(u, v)
            pred_test.append(pred_score)
            pred_node_score[i] = pred_score
            i+=1
            labels.append(1)

        # if self.embed.directed==True:
        #     all_edges = self.embed.train_edges+self.embed.test_edges
        #     for (u, v) in self.embed.get_test_edges():
        #         pred_score = self.embed.link_prob(v, u)
        #         pred_test.append(pred_score)
        #         pred_node_score[i] = pred_score
        #         i+=1
        #         if (v,u) in all_edges:
        #             labels.append(1)
        #         else:
        #             labels.append(0)

        pred_negative = []
        for (u, v) in self.embed.get_negative_edges():
            pred_score = self.embed.link_prob(u, v)
            pred_negative.append(pred_score)
            pred_node_score[i] = pred_score
            i+=1
            labels.append(0)

        # if self.embed.directed==True:
        #     for (u, v) in self.embed.get_negative_edges():
        #         pred_score = self.embed.link_prob(v, u)
        #         pred_negative.append(pred_score)
        #         pred_node_score[i] = pred_score
        #         i+=1
        #         labels.append(0)

        sorted_idx = sorted(pred_node_score.items(), key=operator.itemgetter(1), reverse=True)
        neg_num = 0
        k = 5000
        for (idx, score) in sorted_idx[0:k]:
            if idx >= len(pred_test):
                neg_num+=1
        
        y = []
        for (idx, score) in sorted_idx:
            y.append(score)

        # x = range(1, len(sorted_idx)+1)
        # fig = plt.figure()
        # plt.yscale('log',basey=10)
        # plt.ylabel('score')
        # plt.plot(x, y, linewidth=2.0)
        # fig.savefig('temp.png', dpi=fig.dpi)
        
        print "negative pred: %d, precision@%d: %f"%(neg_num, k, 1.0-neg_num*1.0/k)

        
        pred_labels = np.hstack([pred_test, pred_negative])
        true_labels = labels

        auc_score = roc_auc_score(true_labels, pred_labels)

        ap_score = average_precision_score(true_labels, pred_labels)

        median = np.median(pred_labels)
        index_pos = pred_labels > median
        index_neg = pred_labels <= median
        print "positive preds: %d, negative preds: %d"%(len(index_pos), len(index_neg))
        pred_labels[index_pos] = 1
        pred_labels[index_neg] = 0
        acc_score = accuracy_score(true_labels, pred_labels)

        print "AUC: %f, AP: %f, Accuracy: %f" % (auc_score, ap_score, acc_score)









    # def eval_train(self):
    #     fpos = '../data/%s/pos.spl'%self.embed.data
    #     fneg = '../data/%s/neg.spl'%self.embed.data

    #     pred_test = []
    #     with open(fpos) as fin:
    #         for line in fin:
    #             u, v = line.strip().split()
    #             u, v = int(u)-1, int(v)-1
    #             pred_score = self.embed.link_prob(u, v)
    #             pred_test.append(pred_score)

    #     pred_negative = []
    #     with open(fneg) as fin:
    #         for line in fin:
    #             u, v = line.strip().split()
    #             u, v = int(u)-1, int(v)-1
    #             pred_score = self.embed.link_prob(u, v)
    #             pred_negative.append(pred_score)
        

    #     pred_labels = np.hstack([pred_test, pred_negative])
    #     true_labels = np.hstack([np.ones(len(pred_test)), np.zeros(len(pred_negative))])

    #     auc_score = roc_auc_score(true_labels, pred_labels)

    #     print "AUC on training data: %f" % (auc_score)

    # def eval4(self):
    #     lverse = {}
    #     i = 0
    #     for (u, v) in self.embed.get_test_edges():
    #         pred_score = self.embed.link_prob(u, v)
    #         lverse[i] = pred_score
    #         i = i+1
        
    #     fpos = '../data/%s/pos.ppr'%self.embed.data
        
    #     lppr = {}
    #     i = 0
    #     with open(fpos) as fin:
    #         for line in fin:
    #             ppr = float(line.strip())
    #             lppr[i] = ppr
    #             i = i+1

    #     y_verse = []
    #     y_ppr = []
    #     sorted_ppr_idx = sorted(lppr.items(), key=operator.itemgetter(1), reverse=True)
    #     for (idx, score) in sorted_ppr_idx:
    #         y_ppr.append(score)
    #         y_verse.append(lverse[idx])

    #     maxverse = max(y_verse)
    #     # maxppr = max(y_ppr)

    #     for i in range(len(y_verse)):
    #         y_verse[i] = y_verse[i]/maxverse

    #     # for i in range(len(y_ppr)):
    #     #     y_ppr[i] = y_ppr[i]/maxppr        

    #     x = range(1, len(y_verse)+1)
    #     fig = plt.figure()
    #     plt.title('sorted ppr/inner products for test positive samples of '+self.embed.data)
    #     plt.yscale('log',basey=10)
    #     plt.ylabel('score')
    #     plt.ylim(0.000000000001, 1)
    #     plt.scatter(x, y_verse, s=1, color="r")
    #     plt.scatter(x, y_ppr, s=1, color="y")
    #     fig.savefig(self.embed.data+'-positive.ppr.png', dpi=fig.dpi)


    #     lverse = {}
    #     i = 0
    #     for (u, v) in self.embed.get_negative_edges():
    #         pred_score = self.embed.link_prob(u, v)
    #         lverse[i] = pred_score
    #         i = i+1
        
    #     fneg = '../data/%s/neg.ppr'%self.embed.data
        
    #     lppr = {}
    #     i = 0
    #     with open(fneg) as fin:
    #         for line in fin:
    #             ppr = float(line.strip())
    #             lppr[i] = ppr
    #             i = i+1

    #     y_verse = []
    #     y_ppr = []
    #     sorted_ppr_idx = sorted(lppr.items(), key=operator.itemgetter(1), reverse=True)
    #     for (idx, score) in sorted_ppr_idx:
    #         y_ppr.append(score)
    #         y_verse.append(lverse[idx])

    #     # maxverse = max(y_verse)
    #     # maxppr = max(y_ppr)

    #     for i in range(len(y_verse)):
    #         y_verse[i] = y_verse[i]/maxverse

    #     # for i in range(len(y_ppr)):
    #     #     y_ppr[i] = y_ppr[i]/maxppr    

    #     x = range(1, len(y_verse)+1)
    #     fig = plt.figure()
    #     plt.title('sorted ppr/inner products for test negative samples of '+self.embed.data)
    #     plt.yscale('log',basey=10)
    #     plt.ylabel('score')
    #     plt.ylim(0.000000000001, 1)
    #     plt.scatter(x, y_verse, s=1, color="r")
    #     plt.scatter(x, y_ppr, s=1, color="y")
    #     fig.savefig(self.embed.data+'-negative.ppr.png', dpi=fig.dpi)
        

    # def eval3(self):
    #     fpos = '../data/%s/pos.ppr'%self.embed.data
    #     fneg = '../data/%s/neg.ppr'%self.embed.data

    #     pred_test = []
    #     with open(fpos) as fin:
    #         for line in fin:
    #             ppr = float(line.strip())
    #             pred_test.append(ppr)

    #     pred_negative = []
    #     with open(fneg) as fin:
    #         for line in fin:
    #             ppr = float(line.strip())
    #             pred_negative.append(ppr)
        

    #     pred_labels = np.hstack([pred_test, pred_negative])
    #     true_labels = np.hstack([np.ones(len(pred_test)), np.zeros(len(pred_negative))])

    #     auc_score = roc_auc_score(true_labels, pred_labels)

    #     print "AUC on training data: %f" % (auc_score)

    # def eval2(self):
    #     test_node_score_nw = {}
    #     i = 0
    #     for (u, v) in self.embed.get_negative_edges():
    #         pred_score = self.embed.link_prob(u, v)
    #         test_node_score_nw[i] = pred_score
    #         i+=1

    #     self.embed.tempLoadEmbedding4TrainGraph()
    #     test_node_score_w = {}
    #     i = 0
    #     for (u, v) in self.embed.get_negative_edges():
    #         pred_score = self.embed.link_prob(u, v)
    #         test_node_score_w[i] = pred_score
    #         i+=1

    #     sorted_idx_nw = sorted(test_node_score_nw.items(), key=operator.itemgetter(1), reverse=True)
    #     #sorted_idx_w = sorted(test_node_score_w.items(), key=operator.itemgetter(1), reverse=True)

    #     y_nw = []
    #     y_w = []
    #     for (idx, score) in sorted_idx_nw:
    #         y_nw.append(score)
    #         y_w.append(test_node_score_w[idx])

    #     x = range(1, len(sorted_idx_nw)+1)
    #     fig = plt.figure()
    #     plt.title('sorted inner products for test negative samples of '+self.embed.data)
    #     plt.yscale('log',basey=10)
    #     plt.ylabel('score')
    #     plt.ylim(0.0000000001, 1)
    #     plt.scatter(x, y_w, s=1, color="r")
    #     plt.scatter(x, y_nw, s=1, color="y")
    #     fig.savefig(self.embed.data+'-negative-weight.png', dpi=fig.dpi)
