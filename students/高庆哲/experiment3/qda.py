from __future__ import division
import sys
import math
import copy
from time import time

sys.dont_write_bytecode = True
__author__ = "Saifuddin Abdullah"

# local imports
from tools import fmin, Functions

funcs_ = Functions()

"""
Implementation of a simple QDA for binary and multiclass cases.
"""


class QDA(object):
    def __init__(self, x, y):
        self.funcs_ = funcs_
        # verify the dimensions
        if self.funcs_.verify_dimensions(x):
            if len(x) == len(y):
                self.len_all = len(x)
                self.dmean_ = self.funcs_.mean_nm(x, axis=0)
                self.std_ = self.funcs_.std_nm(x, axis=0)
                self.x = x
                self.y = y
                self.separate_sets()
            else:
                sys.exit()
        else:
            print
            'data dimensions are inaccurate..exiting..'
            sys.exit()

    # --separating the datasets based on their associated classes or labels
    def separate_sets(self):
        self.groups = {}
        self.group_names = list(set(self.y))

        # putting all the samples in a regular order so that their
        # grouping can be easier.
        combined = sorted(zip(self.x, self.y), key=lambda n: n[1])
        # --doing val,key here because (x,y) was zipped
        for val, key in combined:
            if self.groups.has_key(key):
                self.groups[key].append(val)
            else:
                self.groups[key] = []
                self.groups[key].append(val)
        # train on each group
        self.train()

    # --substracts global mean from the group point
    def substract_mean(self, group_point):
        for i, a in enumerate(group_point):
            group_point[i] = group_point[i] - self.mean_global[i]

        return group_point

    # --gets the covariance matrix for a dataset/group (t(x).x/len(x))
    def get_cov_matrix(self, matrix):
        cov_mat = []
        for i in zip(*matrix):
            l = []
            for e in zip(*matrix):
                l.append(self.funcs_.dot(i, e) / len(matrix))
            cov_mat.append(l)
        return cov_mat

    # training the model
    def train(self):
        self.mean_sets = {}
        self.covariance_sets = {}
        self.lens = {}
        self.probability_vector = {}

        self.mean_global = self.funcs_.mean_nm(self.x, axis=0)
        for k, v in self.groups.iteritems():
            self.lens[k] = len(self.groups[k])
            self.probability_vector[k] = self.lens[k] / self.len_all
            self.mean_sets[k] = self.funcs_.mean_nm(self.groups[k], axis=0)
            # mean correcting each set i.e. set - global mean
            self.groups[k] = map(self.substract_mean, self.groups[k])
            self.covariance_sets[k] = self.get_cov_matrix(self.groups[k])

        # here is the major difference between QDA and LDA .. no calculation for the
        # global pooled covariance matrix, but each class has its own pooled
        # covariance matrix.
        for k, v in self.covariance_sets.iteritems():
            self.covariance_sets[k] = self.funcs_.inv_(self.covariance_sets[k])

    # prediction or discrimnant function
    # prediction function for QDA is a little bit different then that of LDA :-)
    def predict(self, v, key_only=True):
        difference_vectors = lambda n, m: n - m
        predictions = {}
        for a in self.group_names:
            predictions[a] = self.funcs_.dot(
                self.funcs_.prod_2(self.funcs_.sclr_prod_(-0.5, map(difference_vectors, v, self.mean_sets[a])),
                                   self.covariance_sets[a]), map(difference_vectors, v, self.mean_sets[a])) - \
                             (0.5 * math.log(self.funcs_.det_(self.covariance_sets[a]))) + math.log(
                self.probability_vector[a])
        if key_only:
            return max(predictions, key=predictions.get)
        else:
            return predictions


