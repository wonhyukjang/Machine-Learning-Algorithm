# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
from datetime import datetime
import pandas as pd

def get_data(freq=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../Data/train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if freq is not None:
        X, Y = X[:freq], Y[:freq]
    return X, Y

def entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return 1 - p0*p0 - p1*p1


class DecisionTree:
    def __init__(self, depth=0, max_depth=None):
        self.max_depth = max_depth
        self.root = {} # is a tree node
        # each node will have the attributes (k-v pairs):
        # - col
        # - split
        # - left
        # - right
        # - prediction

    def fit(self, X, Y):

        current_node = self.root
        depth = 0
        queue = []

        while True:
            # If only 1 class or dataset length is 1
            if len(Y) == 1 or len(set(Y)) == 1:
                current_node['col'] = None
                current_node['split'] = None
                current_node['left'] = None
                current_node['right'] = None
                current_node['prediction'] = Y[0]

            else:
                D = X.shape[1]
                cols = range(D)

                max_ig = 0
                best_col = None
                best_split = None
                for col in cols:
                    ig, split = self.find_split(X, Y, col)

                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split

                # No more Split
                if max_ig == 0:
                    current_node['col'] = None
                    current_node['split'] = None
                    current_node['left'] = None
                    current_node['right'] = None
                    current_node['prediction'] = np.round(Y.mean())
                else:
                    current_node['col'] = best_col
                    current_node['split'] = best_split

                    if depth == self.max_depth:
                        current_node['left'] = None
                        current_node['right'] = None
                        current_node['prediction'] = [
                            np.round(Y[X[:,best_col] < self.split].mean()),
                            np.round(Y[X[:,best_col] >= self.split].mean()),
                        ]
                    else:
                        left_idx = (X[:,best_col] < best_split)

                        # TODO: bad but I can't figure out a better way atm
                        Xleft = X[left_idx]
                        Yleft = Y[left_idx]

                        new_node = {}
                        current_node['left'] = new_node
                        left_data = {
                            'node': new_node,
                            'X': Xleft,
                            'Y': Yleft,
                        }
                        queue.insert(0, left_data)

                        right_idx = (X[:,best_col] >= best_split)
                        Xright = X[right_idx]
                        Yright = Y[right_idx]

                        new_node = {}
                        current_node['right'] = new_node
                        right_data = {
                            'node': new_node,
                            'X': Xright,
                            'Y': Yright,
                        }
                        queue.insert(0, right_data)

            # setup for the next iteration of the loop
            # idea is, queue stores list of work to be done
            if len(queue) == 0:
                break

            next_data = queue.pop()
            current_node = next_data['node']
            X = next_data['X']
            Y = next_data['Y']

    def find_split(self, X, Y, col):
        # print "finding split for col:", col
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes

        # if boundaries[i] is true
        # then y_values[i] != y_values[i+1]
        # nonzero() gives us indices where arg is true
        # but for some reason it returns a tuple of size 1
        
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        last_ig = 0

        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig < last_ig:
                break
            last_ig = ig
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        # assume classes are 0 and 1

        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0 

        # Gini Index for entropy
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

    def predict_one(self, x):
        # use "is not None" because 0 means False
        p = None
        current_node = self.root
        while True:
            if current_node['col'] is not None and current_node['split'] is not None:
                feature = x[current_node['col']]
                if feature < current_node['split']:
                    if current_node['left']:
                        current_node = current_node['left']
                    else:
                        p = current_node['prediction'][0]
                        break
                else:
                    if current_node['right']:
                        current_node = current_node['right']
                    else:
                        p = current_node['prediction'][1]
                        break
            else:
                # corresponds to having only 1 prediction
                p = current_node['prediction']
                break
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
