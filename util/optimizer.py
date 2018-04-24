"""
    Hyperparameter tuning optimizer
    author: yeounoh chung (yeounohster@gmail.com)
    data: 4/11/18
"""
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
import concurrent.futures
import random
import math
import scipy.stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod

class RandomSearch:
    def __init__(self, model, param, n_iter):
        self.best_estimator_ = None
        self.best_score_ = 0.
        self.best_param_ = None
        self.model_ = model
        self.param_ = param
        self.n_iter_ = n_iter
        self.X_v = None
        self.y_v = None

    def fit(self, X, y, cv=3, scorer=f1_score):

        for i in range(self.n_iter_):
            param_args = {}
            for k, p in self.param_.items():
                if type(p) is list:
                    param_args[k] = p[random.randint(0,len(p)-1)]
                elif type(p) is scipy.stats.distributions.rv_frozen:
                    param_args[k] = p.rvs()
            self.model_.set_params(**param_args)

            scores = []
            if cv == 1:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                self.model_.fit(x_train,y_train)
                y_pred = self.model_.predict(x_test)
                scores.append(scorer(y_test, y_pred))
            else:
                kf = ShuffleSplit(n_splits=cv, test_size=0.2)
                for train, test in kf.split(X):
                    self.model_.fit(X[train], y[train])
                    y_pred = self.model_.predict(X[test])
                    scores.append(scorer(y[test], y_pred))
            score_ = np.mean(scores) 

            if score_ > self.best_score_:
                self.best_estimator_ = deepcopy(self.model_)
                self.best_score_ = score_
                self.best_param_ = param_args

        return self

class MultiArmedBandit(ABC):
    def __init__(self, model_space, history=[], rewards={}, budget=25):
        self.model_space_ = model_space
        self.history_ = history
        self.rewards_ = rewards
        self.budget_ = budget

        self.n_candidates_ = 0
        self.best_score_ = 0.
        self.best_estimator_ = None

    @abstractmethod
    def fit(self, X, y):
        pass

class RandomOptimizer(MultiArmedBandit):
    def __init__(self, model_space, history=[], budget=25):
        super(RandomOptimizer, self).__init__(model_space, history, budget=budget)

    def fit(self, X, y, cv=3, scorer=f1_score):
        n_arms = len(self.model_space_)
        for i in range(self.budget_):
            # exploration
            target = self.model_space_[random.randint(0, n_arms-1)]
            rs = RandomSearch(target[0], target[1], 1)
            rs.fit(X, y, cv=cv, scorer=scorer)
            self.history_.append((rs.best_score_, rs.best_estimator_))
            self.n_candidates_ += 1
        self.history_.sort(key=lambda t:t[0])

        self.best_score_ = self.history_[-1][0]
        self.best_estimator_= self.history_[-1][1]

        return self

    def parallel_fit(self, X, y, cv=3, scorer=f1_score):
        n_arms = len(self.model_space_)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch_jobs = {}

            for i in range(self.budget_):
                # exploration
                target = self.model_space_[random.randint(0, n_arms-1)]
                rs = RandomSearch(target[0], target[1], 1)
                batch_jobs[executor.submit(rs.fit, X, y, cv, scorer)] = i

            for job in concurrent.futures.as_completed(batch_jobs):
                if job.cancelled():
                    print('%s-th job cancelled'%batch_jobs[job])
                elif job.done():
                    tag_ = batch_jobs[job]
                    error = job.exception()
                    if error:
                        print('%s error returned: %s'%(tag_, error))
                    else:
                        result_ = job.result()
                        self.history_.append((result_.best_score_, result_.best_estimator_))
                        self.n_candidates_ += 1
        self.history_.sort(key=lambda t:t[0])

        if len(self.history_) > 0:
            self.best_score_ = self.history_[-1][0]
            self.best_estimator_= self.history_[-1][1]

        return self

class GreedyOptimizer(MultiArmedBandit):
    def __init__(self, model_space, history=[], rewards={}, budget=25, eps=0.5):
        super(GreedyOptimizer, self).__init__(model_space, history, rewards, budget)
        self.eps_ = eps

    def fit(self, X, y, cv=3, scorer=f1_score):
        n_arms = len(self.model_space_)
        for i in range(self.budget_):
            print(i,'-th iteration')
            if len(self.history_) == 0 or random.random() < self.eps_:
                # exploration
                target_idx = random.randint(0, n_arms-1)
            else:
                # exploitation 
                k = list(self.rewards_.keys())
                v = [v_[1]/v_[0] for v_ in self.rewards_.values()]
                target_idx = k[v.index(max(v))]
            target = self.model_space_[target_idx]
            rs = RandomSearch(target[0], target[1], 1)
            rs.fit(X, y, cv=cv, scorer=scorer)
            self.history_.append((rs.best_score_, rs.best_estimator_))

            if target_idx in self.rewards_:
                self.rewards_[target_idx] = (self.rewards_[target_idx][0]+1.,
                                             self.rewards_[target_idx][1]+rs.best_score_)
            else:
                self.rewards_[target_idx] = (1., rs.best_score_)
            self.n_candidates_ += 1
        self.history_.sort(key=lambda t:t[0])

        self.best_score_ = self.history_[-1][0]
        self.best_estimator_= self.history_[-1][1]

        return self

