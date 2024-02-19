# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 03:18:22 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""
import os
import numpy as np
from sklearn import preprocessing
from numpy import random
import time
from sklearn import metrics
from pickle import dump, load


class BroadLearningSystem:
    def __init__(
        self,
        saved_folder="",
        model_name="bls_regression",
        no_features=20,
        no_windows=5,
        no_enhanced_neurons=31,
        reg=2**-30,
        verbose=False,
        save_model=False,
    ):
        """
        Args:
            no_features: number of the features per window.
            no_windows: number of windows in feature layer.
            no_enhanced_neurons: number of enhanced neurons.
            shrinkage: the shrinkage parameter for enhancement nodes
            reg: the regularization parameter for sparse regualarization
            save_model: whether to save a model after training, default is False
        """
        self.model_name = model_name
        self.no_features = no_features
        self.no_windows = no_windows
        self.no_enhanced_neurons = no_enhanced_neurons
        self.reg = reg
        self.verbose = verbose
        self.save_model = save_model
        self.MODEL_PATH = f"{saved_folder}/{model_name}.pkl"
        if (not os.path.exists(saved_folder)) and self.save_model:
            os.makedirs(saved_folder)

    def tansig(self, x):
        """
        activation function.
        """
        return (2 / (1 + np.exp(-2 * x))) - 1

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def shrink(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype="double")
        ok = np.zeros([m, n], dtype="double")
        uk = np.zeros([m, n], dtype="double")
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self.shrink(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def training(self, train_x, train_y):
        u = 0
        WF = list()
        for i in range(self.no_windows):
            random.seed(i + u)
            WeightFea = 2 * random.randn(train_x.shape[1] + 1, self.no_features) - 1
            WF.append(WeightFea)
        #    random.seed(100)
        self.WeightEnhan = (
            2
            * random.randn(
                self.no_windows * self.no_features + 1, self.no_enhanced_neurons
            )
            - 1
        )
        # time_start = time.time()
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.no_windows * self.no_features])
        self.WFSparse = list()
        self.distOfMaxAndMin = np.zeros(self.no_windows)
        self.meanOfEachWindow = np.zeros(self.no_windows)
        for i in range(self.no_windows):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse = self.sparse_bls(A1, H1).T
            self.WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            y[:, self.no_features * i : self.no_features * (i + 1)] = T1

        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(self.WeightEnhan)

        T2 = self.tansig(T2)
        T3 = np.hstack([y, T2])
        self.WeightTop = self.pinv(T3, self.reg).dot(train_y)
        if self.save_model:
            self.save()

    def save(self):
        dump(
            [
                self.WFSparse,
                self.distOfMaxAndMin,
                self.meanOfEachWindow,
                self.WeightTop,
                self.WeightEnhan,
            ],
            open(self.MODEL_PATH, "wb"),
        )

    def predict(self, test_x):
        no_samples = test_x.shape[0]
        HH1 = np.hstack([test_x, 0.1 * np.ones([no_samples, 1])])
        yy1 = np.zeros([no_samples, self.no_windows * self.no_features])
        for i in range(self.no_windows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.no_features * i : self.no_features * (i + 1)] = TT1

        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        prediction = TT3.dot(self.WeightTop)
        return prediction

    def inference(self, input_seq, pred_steps, x_y_lag=1):
        """
        Args:
            input_seq: input x
            pred_steps: number of steps need to infer
            x_y_lag: time step gap between input x and output y
        """
        assert (
            len(input_seq.shape) == 2
        ), f"input sequence should be 2 dimensions: (samples, time steps), but {input_seq.shape}"
        input_steps = input_seq.shape[1]
        assert (
            x_y_lag <= input_steps
        ), "time step gap between input x and output y should be smaller than the number of steps of input sequence, otherwise, inference process cannot be continue."
        sequence = input_seq
        if os.path.exists(self.MODEL_PATH):
            (
                self.WFSparse,
                self.distOfMaxAndMin,
                self.meanOfEachWindow,
                self.WeightTop,
                self.WeightEnhan,
            ) = load(open(self.MODEL_PATH, "rb"))
        for step in range(pred_steps):
            prediction = self.predict(sequence[:, -input_steps:])
            sequence = np.append(sequence, prediction[:, -x_y_lag:], axis=1)
        return np.array(sequence[:, : input_steps + pred_steps])
