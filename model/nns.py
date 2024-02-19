"""
Created on Sat Jan 28 2020
@author: Yong Zhuang  https://stackoverflow.com/questions/50606758/vscode-how-do-you-autoformat-on-save
"""

import os
import numpy as np
import math
import json
import time
import tensorflow as tf

# import tensorflow.python.ops as tops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn

# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import gen_array_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bernoulli

# from tensorflow_probability.distributions import bernoulli
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import (
    RNN,
    GRUCell,
    GRU,
    Lambda,
    Multiply,
    Add,
    Concatenate,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    Callback,
    CSVLogger,
)
from tensorflow.keras.models import Model
from model.layers import *
from progress.bar import Bar


class BaseModel:
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="base",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
    ):
        """
        Args:
            dim: feature dimension.
            saved_folder: the folder to save the trained model, if the folder does not exist, will generate it.
            model_name: name of the model
            latent_dim: dimension of the hidden states
            verbose: show debug message
            x_y_lag: time step gap between input x and output y
        """
        self.dim = dim
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.MODEL_PATH = f"{saved_folder}/{model_name}.h5"
        self.LOG_PATH = f"{saved_folder}/{model_name}.csv"
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.verbose = verbose
        self.x_y_lag = x_y_lag
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)

    def get_callbacks(self):
        model_checkpoint = ModelCheckpoint(
            self.MODEL_PATH,
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, mode="auto", min_delta=1e-9
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-9
        )
        csv_logger = CSVLogger(self.LOG_PATH, append=True)
        callbacks = [model_checkpoint, reduce_lr, early_stopping, csv_logger]
        return callbacks

    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2,
        )
        return callbacks, history

    def fit_generator(
        self,
        batch_size,
        epochs,
        callbacks,
        datagenerator_train,
        datagenerator_val,
        num_train_samples,
        num_val_samples,
    ):
        history = self.model.fit_generator(
            datagenerator_train,
            steps_per_epoch=num_train_samples / batch_size,
            epochs=epochs,
            validation_data=datagenerator_val,
            validation_steps=num_val_samples / batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        return callbacks, history

    def training(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        callbacks = self.get_callbacks()
        return self.fit(X_train, y_train, X_val, y_val, batch_size, epochs, callbacks)

    def training_generator(
        self,
        batch_size,
        epochs,
        datagenerator_train=None,
        datagenerator_val=None,
        num_train_samples=0,
        num_val_samples=0,
    ):
        callbacks = self.get_callbacks()
        return self.fit_generator(
            batch_size,
            epochs,
            callbacks,
            datagenerator_train,
            datagenerator_val,
            num_train_samples,
            num_val_samples,
        )

    def load_weights(self):
        if os.path.exists(self.MODEL_PATH):
            self.model.load_weights(self.MODEL_PATH)
        else:
            print(f"no trained model available at {self.MODEL_PATH}")
            return

    def inference(self, input_seq, pred_steps, model_path=None):
        """
        Args:
            input_seq: input x
            pred_steps: number of steps need to infer
            x_y_lag: time step gap between input x and output y
        """
        assert (
            len(input_seq.shape) >= 3
        ), "input sequence should be greater 3 dimensions: (samples, time steps, other dimensions)"
        input_steps = input_seq.shape[1]
        assert (
            self.x_y_lag <= input_steps
        ), "time step gap between input x and output y should be smaller than the number of steps of input sequence, otherwise, inference process cannot be continue."

        if model_path is not None:
            self.model.load_weights(model_path)
        elif os.path.exists(self.MODEL_PATH):
            self.model.load_weights(self.MODEL_PATH)
        else:
            print(f"no trained model available at {self.MODEL_PATH}")
            return None
        sequence = input_seq
        # Encode the input sequence.
        encoder_y_pred, encoder_state = self.encoder_model.predict(input_seq)
        states_value = [
            encoder_state,
        ]
        # Generate empty target sequence of length 1.
        # target_seq = np.zeros((encoder_y_pred.shape[0], 1, self.dim))
        # Push the last prediction into the target sequence.
        # target_seq[:, 0] = encoder_y_pred[:,-1]
        # Sampling loop for a batch of sequences
        sequence = np.append(sequence, encoder_y_pred[:, -self.x_y_lag :], axis=1)
        # np.empty((encoder_y_pred.shape[0],0,self.dim))
        # stop_condition = False
        for i in Bar("inferring").iter(range(int(pred_steps / self.x_y_lag))):
            if i % 100 == 0:
                print(f"{self.model_name} inferring... {(i/pred_steps)*100}% completed")
            y_pred, _, state = self.decoder_model.predict(
                [sequence[:, -self.x_y_lag :]] + states_value
            )
            states_value = [
                state,
            ]
            # decoded_y_sequence = np.append(decoded_y_sequence,decoder_y_pred,axis=1)
            sequence = np.append(sequence, y_pred, axis=1)
            # Exit condition: either hit max length
            # if decoded_y_sequence.shape[1] >= (pred_steps):
            #     stop_condition = True
            # Update the target sequence (of length 1).
            # target_seq[:, 0] = decoder_y_pred[:,-1]
        return sequence[:, : input_steps + pred_steps]


class TeacherForcing(BaseModel):
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="teacher_forcing",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
    ):
        super(TeacherForcing, self).__init__(
            dim, saved_folder, model_name, latent_dim, verbose, x_y_lag
        )
        inputs = Input(shape=(None, self.dim), name="input")
        self.gru = RNN(
            GRUCell(self.latent_dim),
            return_state=True,
            return_sequences=True,
            name="gru",
        )
        rnn_outputs, state = self.gru(inputs)
        rnn_state = [
            state,
        ]

        """prediction"""
        self.output_dense = Dense(self.dim, activation="linear", name="output_dense")
        y_pred = self.output_dense(rnn_outputs)
        self.model = Model(
            [inputs],
            [y_pred],
        )
        adam = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(loss="mse", optimizer=adam, metrics=["mae"])
        if self.verbose:
            self.model.summary()

        """ Inference model """
        self.encoder_model = Model([inputs], [y_pred] + rnn_state)
        decoder_state_input = Input(shape=(self.latent_dim,))
        decoder_outputs, _state = self.gru(inputs, initial_state=decoder_state_input)
        decoder_state = [
            _state,
        ]
        decoder_y_pred = self.output_dense(decoder_outputs)
        self.decoder_model = Model(
            [inputs]
            + [
                decoder_state_input,
            ],
            [decoder_y_pred, decoder_outputs] + decoder_state,
        )


class DiscontinuousScheduledSampling(BaseModel):
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="discontinuous_scheduled_sampling",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
        ts=0,
        batch_size=0,
        decay="is",
    ):
        """
        Args:
            ts: time steps of the input.
        """
        super(DiscontinuousScheduledSampling, self).__init__(
            dim, saved_folder, f"{model_name}_{decay}", latent_dim, verbose, x_y_lag
        )

        self.beta = K.variable(1.0)
        self.ts = ts
        self.batch_size = batch_size
        self.decay = decay
        inputs = Input(shape=(None, self.dim), name="input")
        # first_input = Lambda(lambda x: x[:, 0:1, :])(inputs)
        # input_tensor = Lambda(lambda x:x)(inputs)
        # self.gru = GRU(self.latent_dim, return_sequences=True, return_state=True, name="gru")
        self.gru = RNN(
            GRUCell(self.latent_dim),
            return_state=True,
            return_sequences=True,
            name="gru",
        )
        self.output_dense = Dense(self.dim, activation="linear", name="output_dense")
        self.ss_gru = ScheduledSamplingLayer(
            "ss_gru",
            self.gru,
            self.output_dense,
            self.dim,
            self.latent_dim,
            self.beta,
            self.batch_size,
        )

        # gru_outputs, gru_states = self.gru(inputs)

        final_outputs = self.ss_gru(inputs)

        self.model = Model(inputs=[inputs], outputs=final_outputs)
        adam = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(loss=["mse"], optimizer=adam, metrics=["mae"])
        # if self.verbose:
        # self.model.summary()

        """ Inference model """
        gru_outputs, gru_states = self.gru(inputs)
        encoder_pred = self.output_dense(gru_outputs)
        self.encoder_model = Model([inputs], [encoder_pred, gru_states])

        decoder_state_input = Input(shape=(self.latent_dim,))
        decoder_inf_out, decoder_inf_state = self.gru(
            inputs, initial_state=decoder_state_input
        )
        decoder_pred = self.output_dense(decoder_inf_out)
        self.decoder_model = Model(
            [inputs, decoder_state_input],
            [decoder_pred, decoder_inf_out, decoder_inf_state],
        )

    def training(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        model_checkpoint = ModelCheckpoint(
            self.MODEL_PATH,
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        )
        # early_stopping = EarlyStopping(monitor="val_loss", patience=15, mode="auto", min_delta=1e-9)
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-9
        )
        csv_logger = CSVLogger(self.LOG_PATH, append=True)
        callbacks = [reduce_lr, csv_logger]  # early_stopping,model_checkpoint,
        if self.decay == "is":
            callbacks = callbacks + [InverseSigmoidCallback(self.beta)]
        if self.decay == "exp":
            callbacks = callbacks + [ExponentialDecayCallback(self.beta)]
        if self.decay == "linear":
            callbacks = callbacks + [LinearDecayCallback(self.beta)]
        callbacks, history = self.fit(
            X_train, y_train, X_val, y_val, batch_size, epochs, callbacks
        )
        self.model.save_weights(self.MODEL_PATH)
        return callbacks, history


class SuperHorizonForcing(BaseModel):
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="super_horizon_forcing",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
        k=10,
        lr=0.00000000001,
        suffix="",
    ):
        """
        Args:
            k: number of horizon steps for training.
            lr: init learning rate for the model training
        """
        super(SuperHorizonForcing, self).__init__(
            dim, saved_folder, f"{model_name}_{k}{suffix}", latent_dim, verbose, x_y_lag
        )
        self.k = k
        self.lr = lr
        inputs = Input(shape=(None, self.dim), name="input")

        self.gru = RNN(
            GRUCell(self.latent_dim),
            return_state=True,
            return_sequences=True,
            name="gru",
        )
        gru_outputs, gru_states = self.gru(inputs)
        """Y regression prediction"""
        self.output_dense = Dense(self.dim, activation="linear", name="output_dense")
        outputs = []
        self.time_gru = HorizonForcingGRULayer(
            f"time_gru", self.gru, self.output_dense, self.dim
        )
        for j in range(self.k + 1):
            preds, gru_outs = self.time_gru(gru_outputs)
            outputs += [preds]
            gru_outputs = gru_outs[:, :-1]
        self.model = Model([inputs], outputs[-1])
        adam = optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.model.compile(loss=["mse"], optimizer=adam, metrics=["mae"])
        # if self.verbose:
        # self.model.summary()

        """ Inference model """
        self.encoder_model = Model([inputs], [outputs[0], gru_states])
        decoder_state_input = Input(shape=(self.latent_dim,))
        decoder_outputs, decoder_states = self.gru(
            inputs, initial_state=decoder_state_input
        )
        decoder_y_pred = self.output_dense(decoder_outputs)
        self.decoder_model = Model(
            [inputs, decoder_state_input],
            [decoder_y_pred, decoder_outputs, decoder_states],
        )


class HorizonForcing(BaseModel):
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="horizon_forcing",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
        k=10,
        lr=0.00000000001,
        suffix="",
    ):
        """
        Args:
            k: number of horizon steps for training.
            lr: init learning rate for the model training
        """
        super(HorizonForcing, self).__init__(
            dim, saved_folder, f"{model_name}_{k}{suffix}", latent_dim, verbose, x_y_lag
        )
        self.k = k
        self.lr = lr
        inputs = Input(shape=(None, self.dim), name="input")
        self.gru = RNN(
            GRUCell(self.latent_dim),
            return_state=True,
            return_sequences=True,
            name="gru",
        )
        gru_outputs, gru_states = self.gru(inputs)
        """Y regression prediction"""
        self.output_dense = Dense(self.dim, activation="linear", name="output_dense")
        outputs = []
        self.time_gru = HorizonForcingGRULayer(
            f"time_gru", self.gru, self.output_dense, self.dim
        )
        for j in range(self.k + 1):
            preds, gru_outs = self.time_gru(gru_outputs)
            outputs += [preds]
            gru_outputs = gru_outs[:, :-1]
        self.model = Model([inputs], outputs[-1])
        adam = optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.model.compile(loss=["mse"], optimizer=adam, metrics=["mae"])
        # if self.verbose:
        # self.model.summary()

        """ Inference model """
        self.encoder_model = Model([inputs], [outputs[0], gru_states])
        decoder_state_input = Input(shape=(self.latent_dim,))
        decoder_outputs, decoder_states = self.gru(
            inputs, initial_state=decoder_state_input
        )
        decoder_y_pred = self.output_dense(decoder_outputs)
        self.decoder_model = Model(
            [inputs, decoder_state_input],
            [decoder_y_pred, decoder_outputs, decoder_states],
        )


class InverseSigmoidCallback(Callback):
    """
    a call back function to change the rate of the true value inputs used in scheduled sampling training  by an inverse sigmoid function
    """

    def __init__(self, beta, min_rate=0.0, k=700):
        self.min_rate = min_rate
        self.beta = beta
        # self.gap = gap
        self.k = k  # 100#+gap*4.0

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        # self.best = np.Inf
        self.decay_step = 1

    # customize your behavior
    def on_batch_end(self, batch, logs={}):
        logs["beta"] = K.get_value(self.beta)
        current = logs.get("val_loss")
        # if np.less(current, self.best):
        #     self.best = current
        update = max(
            self.min_rate, self.k / (self.k + math.exp(self.decay_step / self.k))
        )
        K.set_value(self.beta, update)
        self.decay_step += 1

    def on_epoch_end(self, epoch, logs={}):
        print(
            f"decay_step {self.decay_step}: {K.get_value(self.beta)*100}% usage of ground truth input. "
        )


class LinearDecayCallback(Callback):
    def __init__(self, beta, min_rate=0.0, decay_rate=0.0001):
        self.min_rate = min_rate
        self.beta = beta
        # self.alpha = alpha
        self.decay = decay_rate

    def on_train_begin(self, logs=None):
        #     # Initialize the best as infinity.
        #     self.best = np.Inf
        self.decay_step = 1

    # customize your behavior
    def on_batch_end(self, batch, logs={}):
        logs["beta"] = K.get_value(self.beta)
        current = logs.get("val_loss")
        # if np.less(current, self.best):
        #     self.best = current
        update = max(self.min_rate, self.beta - self.decay)
        K.set_value(self.beta, update)
        # K.set_value(self.alpha, 1-update)
        self.decay_step += 1

    def on_epoch_end(self, epoch, logs={}):
        print(
            f"decay_step {self.decay_step}: {K.get_value(self.beta)*100}% usage of ground truth input. "
        )


class ExponentialDecayCallback(Callback):
    def __init__(self, beta, k=0.9994, min_rate=0.0):
        self.min_rate = min_rate
        self.beta = beta
        # self.alpha = alpha
        # self.gap = gap
        self.k = k

    def on_train_begin(self, logs=None):
        #     # Initialize the best as infinity.
        #     self.best = np.Inf
        self.decay_step = 1

    # customize your behavior
    def on_batch_end(self, batch, logs={}):
        logs["beta"] = K.get_value(self.beta)

        # current = logs.get('val_loss')
        # if np.less(current, self.best):
        #     self.best = current
        update = max(self.min_rate, math.pow(self.k, self.decay_step))
        K.set_value(self.beta, update)
        self.decay_step += 1

    def on_epoch_end(self, epoch, logs={}):
        print(
            f"decay_step {self.decay_step}: {K.get_value(self.beta)*100}% usage of ground truth input. "
        )


class MusicTransformer(BaseModel):
    def __init__(
        self,
        dim=1,
        saved_folder="",
        model_name="music_transformer",
        latent_dim=256,
        verbose=False,
        x_y_lag=1,
        num_layer=6,
        max_seq=100,
        dropout=0.0,
        save_config=True,
    ):
        # (self,embedding_dim=256, num_layer=6,x_dim=3,
        #              max_seq=100, dropout=0.0, save_config=True, config_path=None, model_name = "MT"):
        super(MusicTransformer, self).__init__(
            dim, saved_folder, model_name, latent_dim, verbose, x_y_lag
        )
        self.CONFIG_PATH = f"{saved_folder}/{model_name}.json"
        self.save_config = save_config
        if os.path.exists(self.CONFIG_PATH):
            self.load_config_file(self.CONFIG_PATH)
        else:
            self.max_seq = max_seq
            self.num_layer = num_layer
        inputs = Input(shape=(None, self.dim), name="input")
        look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
        self.Decoder = Encoder(
            num_layers=self.num_layer,
            d_model=self.latent_dim,
            rate=dropout,
            max_len=self.max_seq,
        )
        self.fc = Dense(self.dim, activation=None, name="output")
        decoder, weights = self.Decoder(inputs, mask=look_ahead_mask)
        y_pred = self.fc(decoder)

        self.model = Model(
            [inputs, look_ahead_mask],
            [y_pred],
        )
        adam = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(loss="mse", optimizer=adam, metrics=["mae"])

    def training(self, X_train, y_train, X_val, y_val, batch_size, epochs, fold=0):
        start_time = time.time()
        callbacks = self.get_callbacks()
        train_mask = self.get_masked_with_pad_tensor(X_train[0])
        val_mask = self.get_masked_with_pad_tensor(X_val[0])
        callbacks, history = self.fit(
            X_train + [train_mask],
            y_train,
            X_val + [val_mask],
            y_val,
            batch_size,
            epochs,
            callbacks,
        )
        if self.save_config:
            running_time = time.time() - start_time
            self.save(self.CONFIG_PATH, running_time=running_time)
        return callbacks, history

    def save(self, filepath, running_time=0):
        config_path = filepath
        with open(config_path, "w") as f:
            json.dump(self.get_config(running_time=running_time), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath
        with open(config_path, "r") as f:
            config = json.load(f)
        self.max_seq = config["max_seq"]
        self.num_layer = config["num_layer"]
        self.latent_dim = config["latent_dim"]

    def get_config(self, running_time):
        config = {}
        config["max_seq"] = self.max_seq
        config["num_layer"] = self.num_layer
        config["latent_dim"] = self.latent_dim
        config["running_time"] = running_time
        return config

    def inference(self, input_seq, pred_steps=2048):
        """
        Args:
            input_seq: input x
            pred_steps: number of steps need to infer
            x_y_lag: time step gap between input x and output y
        """
        assert (
            len(input_seq.shape) >= 3
        ), "input sequence should be greater 3 dimensions: (samples, time steps, other dimensions)"
        input_steps = input_seq.shape[1]
        assert (
            self.x_y_lag <= input_steps
        ), "time step gap between input x and output y should be smaller than the number of steps of input sequence, otherwise, inference process cannot be continue."
        sequence = tf.constant(input_seq, dtype=tf.float32)
        if os.path.exists(self.MODEL_PATH):
            self.model.load_weights(self.MODEL_PATH)
        else:
            print(f"no trained model available at {self.MODEL_PATH}")
            return
        for i in Bar("inferring").iter(range(int(pred_steps / self.x_y_lag) + 1)):
            if i % 100 == 0:
                print(f"{self.model_name} inferring... {(i/pred_steps)*100}% completed")
            look_ahead_mask = self.get_masked_with_pad_tensor(
                sequence[:, -input_steps:, :]
            )

            pred = self.model.predict([sequence[:, -input_steps:, :], look_ahead_mask])
            # if tf_board: , tf_board=False
            #     tf.summary.image('generate_vector', tf.expand_dims(pred, -1), i)
            sequence = tf.concat([sequence, pred[:, -self.x_y_lag :, :]], 1)
        return sequence.numpy()[:, : input_steps + pred_steps]

    def get_masked_with_pad_tensor(self, inputs):
        array_len = np.arange(1, inputs.shape[1] + 1)

        m = ~(
            np.ones((inputs.shape[1], inputs.shape[1])).cumsum(axis=1).T <= array_len
        ).T
        m = m.astype(int)
        m = m[np.newaxis, np.newaxis, :]
        look_ahead_mask = np.repeat(m, inputs.shape[0], axis=0)
        return look_ahead_mask
