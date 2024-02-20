"""
Created on Sat Jan 28 2020
@author: Yong Zhuang
"""

import tensorflow as tf
import math as m
import numpy as np
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import backend as K

# from tensorflow.python.ops.distributions import bernoulli


class DynamicPositionEmbedding(keras.layers.Layer):
    def __init__(self, embedding_dim, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array(
            [
                [
                    [
                        m.sin(
                            pos
                            * m.exp(-m.log(10000) * i / embedding_dim)
                            * m.exp(m.log(10000) / embedding_dim * (i % 2))
                            + 0.5 * m.pi * (i % 2)
                        )
                        for i in range(embedding_dim)
                    ]
                    for pos in range(max_seq)
                ]
            ]
        )
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:, : inputs.shape[1], :])


class RelativeGlobalAttention(keras.layers.Layer):
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d))
        self.Wk = keras.layers.Dense(int(self.d))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)
        self.additional = add_emb
        if self.additional:
            self.Radd = None

    def build(self, input_shape):
        self.shape_q = input_shape[0][1]
        self.shape_k = input_shape[1][1]
        self.E = self.add_weight("emb", shape=[self.shape_q, int(self.dh)])

    def call(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q_shape = tf.shape(q)
        q = tf.reshape(q, (q_shape[0], q_shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = inputs[1]
        k = self.Wk(k)
        k_shape = tf.shape(k)
        k = tf.reshape(k, (k_shape[0], k_shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v_shape = tf.shape(v)
        v = tf.reshape(v, (v_shape[0], v_shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        self.len_k = tf.shape(k)[2]
        self.len_q = tf.shape(q)[2]

        E = self.E[: q_shape[1]]
        QE = tf.einsum("bhld,md->bhlm", q, E)
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = tf.transpose(k, [0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)

        logits = QKt + Srel
        logits = logits / m.sqrt(self.dh)
        print(type(mask))
        if mask is not None:
            logits += tf.cast(mask, tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, -1)

        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (tf.shape(out)[0], -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    @staticmethod
    def _qe_masking(qe):
        qe_shape = tf.shape(qe)
        mask = tf.sequence_mask(
            tf.range(qe_shape[-1] - 1, qe_shape[-1] - qe_shape[-2] - 1, -1),
            qe_shape[-1],
        )

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [1, 0]])
        padded_shape = tf.shape(padded)
        reshaped = tf.reshape(
            padded, shape=[-1, padded_shape[1], padded_shape[-1], padded_shape[-2]]
        )
        Srel = reshaped[:, :, 1:, :]
        return Srel


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(
            h=h, d=d_model, max_seq=max_seq, add_emb=additional
        )

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False, **kwargs):
        attn_out, w = self.rga([x, x, x], mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out + x)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2, w


class Encoder(keras.layers.Layer):
    def __init__(
        self, num_layers, d_model, rate=0.1, max_len=None, PositionEmbedding=True
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Dense(d_model)
        if PositionEmbedding:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = [
            EncoderLayer(
                d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len
            )
            for i in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        weights = []
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask, training=training)
            weights.append(w)
        return x, weights


class HorizonForcingGRULayer(keras.layers.Layer):
    def __init__(self, name, gru, output_dense, out_dim):
        super(HorizonForcingGRULayer, self).__init__(name=name)
        self.gru = gru
        self.output_dense = output_dense
        self.out_dim = out_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "name": self.name,
            }
        )
        return config

    def build(self, input_shape):
        super(HorizonForcingGRULayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        gru_output_seq = inputs
        gru_output_seq_shape = tf.shape(gru_output_seq)
        if verbose:
            print("gru_output_seq>", gru_output_seq.shape)

        def gru_step(inputs, states):
            """Step function for 1 step inference for a single time state"""
            # set the time dimension
            x_input = K.expand_dims(inputs, 1)
            pred = self.output_dense(x_input)
            if verbose:
                print(f"pred>{pred.shape}")  # => (batch_size, 1, dim)
            gru_outs, state = self.gru(
                pred, initial_state=inputs
            )  # => (batch_size, hidden)
            # concatenate the prediction and hidden state as the finial output
            out = K.concatenate(
                (pred[:, 0], state), axis=-1
            )  # => (batch_size, dim + hidden)
            if verbose:
                print(f"out shape is {out.shape}")
            return out, [out]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(
                inputs
            )  # <= (batch_size, enc_seq_len, hidden_size)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(
                fake_state, [1, hidden_size]
            )  # <= (batch_size, hidden_size)
            return fake_state

        fake_state = create_inital_state(
            gru_output_seq, self.out_dim + gru_output_seq_shape[-1]
        )

        """ Computing 1 step inference outputs """
        _, outputs, _ = K.rnn(gru_step, gru_output_seq, [fake_state])
        preds, gru_states = outputs[:, :, : self.out_dim], outputs[:, :, self.out_dim :]
        return preds, gru_states

    def compute_output_shape(self, input_shape):
        """Outputs produced by the layer"""
        return [(input_shape[0], input_shape[1], self.out_dim), input_shape]


class ScheduledSamplingGRULayer(keras.layers.Layer):
    def __init__(self, name, gru, output_dense, out_dim, beta, ts):
        super(ScheduledSamplingGRULayer, self).__init__(name=name)
        self.gru = gru
        self.output_dense = output_dense
        self.out_dim = out_dim
        self.beta = beta
        self.ts = ts

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "name": self.name,
            }
        )
        return config

    def build(self, input_shape):
        super(ScheduledSamplingGRULayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        gru_output_seq = inputs
        gru_output_seq_shape = tf.shape(inputs)
        if verbose:
            print("gru_output_seq>", gru_output_seq.shape)

        def gru_step(inputs, states):
            """Step function for 1 step inference for a single time state"""
            # set the time dimension
            x_input = K.expand_dims(inputs, 1)
            pred = self.output_dense(x_input)
            if verbose:
                print(f"pred>{pred.shape}")  # => (batch_size, 1, dim)
            gru_outs, state = self.gru(
                pred, initial_state=inputs
            )  # => (batch_size, hidden)
            # concatenate the prediction and hidden state as the finial output
            out = K.concatenate(
                (pred[:, 0], state), axis=-1
            )  # => (batch_size, dim + hidden)
            if verbose:
                print(f"out shape is {out.shape}")
            return out, [out]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(
                inputs
            )  # <= (batch_size, enc_seq_len, hidden_size)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(
                fake_state, [1, hidden_size]
            )  # <= (batch_size, hidden_size)
            return fake_state

        fake_state = create_inital_state(
            gru_output_seq, self.out_dim + gru_output_seq_shape[-1]
        )

        """ Computing 1 step inference outputs """
        _, outputs, _ = K.rnn(gru_step, gru_output_seq, [fake_state])
        preds1, gru_states = (
            outputs[:, :, : self.out_dim],
            outputs[:, :, self.out_dim :],
        )

        _, outputs, _ = K.rnn(gru_step, gru_states[:, :-1], [fake_state])
        preds2, _ = outputs[:, :, : self.out_dim], outputs[:, :, self.out_dim :]

        beta = K.get_value(self.beta)
        print(f"beta value is {beta}")
        # uniform_distribution = tf.random.uniform(
        #     shape=[self.ts-1],
        #     minval=0,
        #     maxval=None,
        #     dtype=tf.float32,
        #     seed=None,
        #     name=None
        # )
        sampling_size = int(beta * self.ts)
        # _, indices_to_keep = tf.nn.top_k(uniform_distribution, sampling_size)
        # n = gru_output_seq_shape[0]
        # # Make tensor of indices for the first dimension
        # ii = tf.tile(tf.range(n)[:, tf.newaxis], (1, sampling_size))
        # print(f"ii is {ii}")
        # # Stack indices
        # idx = tf.stack([ii, indices_to_keep], axis=-1)
        # print(f"idx is {idx}")
        # sorted_indices_to_keep = tf.contrib.framework.sort(indices_to_keep)
        # masking_idx = np.random.choice(self.ts-1, size=sampling_size, replace=False)#.tolist()
        # print (f"masking_idx is {masking_idx}")
        # print (f"masking_idx is {type(masking_idx)}")
        # masking_idx = tf.constant(masking_idx, dtype=tf.int32)
        # print(f"preds1 is {preds1.shape}")
        # print (f"masking_idx is {masking_idx}")
        # preds =tf.transpose(preds1, [1, 0, 2])
        # preds[sampling_size]= tf.transpose(preds2, [1, 0, 2])[sampling_size]
        # preds = tf.transpose(preds, [1, 0, 2])
        print(
            f"preds1 shape {preds1.shape}, preds2 shape {preds2.shape}, sampling_size is {sampling_size}"
        )
        preds = tf.concat(
            [preds1[:, : sampling_size + 1], preds2[:, sampling_size:]], axis=1
        )
        # print(f"preds1[0,-1] is {preds1[0,-1]}; preds2[0,-1] is {preds2[0,-1]}")
        # K.print_tensor(preds1[0,-1], message="preds1 is: ")
        # K.print_tensor(preds2[0,-1], message="preds2 is: ")
        # preds = preds1
        return preds, gru_states

    def compute_output_shape(self, input_shape):
        """Outputs produced by the layer"""
        return [(input_shape[0], input_shape[1], self.out_dim), input_shape]


class ScheduledSamplingLayer(keras.layers.Layer):
    def __init__(self, name, gru, output_dense, out_dim, latent_dim, beta, batch_size):
        super(ScheduledSamplingLayer, self).__init__(name=name)
        self.gru = gru
        self.output_dense = output_dense
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.batch_size = batch_size
        self.sampling = False

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "name": self.name,
            }
        )
        return config

    def build(self, input_shape):
        super(ScheduledSamplingLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        if verbose:
            print("inputs shape ->", tf.shape(inputs))

        def get_next_input(true, estimate):
            # Return -1s where we do not sample, and sample_ids elsewhere
            current_batch_size = tf.shape(true)[0]
            select_sampler = tfp.distributions.Bernoulli(
                probs=self.beta, dtype=tf.bool
            )  # bernoulli.Bernoulli(probs=self.beta, dtype=tf.bool)
            select_sample = select_sampler.sample(sample_shape=current_batch_size)
            sample_ids = tf.where(
                select_sample,
                tf.range(current_batch_size),
                tf.fill([current_batch_size], -1),
            )

            where_not_sampling = tf.cast(tf.where(sample_ids > -1), tf.int32)
            where_sampling = tf.cast(tf.where(sample_ids <= -1), tf.int32)
            # if self.i == 1:
            #     print(f"where_sampling shape {where_sampling.shape}; {where_sampling[:10]}")
            #     print(f"where_not_sampling shape {where_not_sampling.shape}; {where_not_sampling[:10]}")
            # print(
            #     f"where_not_sampling {where_not_sampling}; true shape {true.shape}; current_batch_size {tf.shape(true)[0]}"
            # )
            _estimate = tf.gather_nd(estimate, where_sampling)
            _true = tf.gather_nd(true, where_not_sampling)
            # print(f"_estimate shape {_estimate.shape}; _true shape {_true.shape}")

            base_shape = tf.shape(true)  # tf.constant([self.batch_size, 1, self.dim])
            result1 = tf.scatter_nd(
                indices=where_sampling, updates=_estimate, shape=base_shape
            )
            result2 = tf.scatter_nd(
                indices=where_not_sampling, updates=_true, shape=base_shape
            )
            # print(f"result1 type {type(result1)}; result2 type {type(result2)}")
            # print(f"result1 shape {result1.shape}; result2 shape {result2.shape}")
            result = result1 + result2  # Add()([result1, result2]) #
            # print(f"result type {type(result)}; {result.shape}")
            return result

        def gru_step(inputs, states):
            """Step function for  a single time state"""
            # set the time dimension
            next_input = K.expand_dims(inputs, 1)
            raw_input = K.expand_dims(states[0], 1)
            if self.sampling:
                estimate = self.output_dense(raw_input)
                next_input = get_next_input(next_input, estimate)
                gru_outs, state = self.gru(
                    next_input, initial_state=states
                )  # => (batch_size, hidden)
            else:
                estimate = self.output_dense(raw_input)
                gru_outs, state = self.gru(next_input)  # => (batch_size, hidden)
            # concatenate the prediction and hidden state as the finial output
            # out = K.concatenate((pred[:,0], state), axis=-1) #=> (batch_size, dim + hidden)
            if self.sampling:
                return estimate[:, 0], [state]
            else:
                self.sampling = True
                return estimate[:, 0], [state]
            # return inputs, states

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(
                inputs
            )  # <= (batch_size, enc_seq_len, hidden_size)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(
                fake_state, [1, hidden_size]
            )  # <= (batch_size, hidden_size)
            return fake_state

        fake_state = create_inital_state(inputs, self.latent_dim)
        """ Computing 1 step inference outputs """
        last_output, outputs, state = K.rnn(gru_step, inputs, [fake_state])
        state = K.expand_dims(state[0], -2)
        last_pred = self.output_dense(state)
        ss_preds = tf.concat([outputs[:, 1:], last_pred], axis=1)
        return ss_preds

    def compute_output_shape(self, input_shape):
        """Outputs produced by the layer"""
        return (input_shape[0], input_shape[1], self.out_dim)
