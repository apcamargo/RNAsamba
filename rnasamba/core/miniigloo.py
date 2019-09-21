# -*- coding: utf-8 -*-
#
#   This file is part of the rnasamba package, available at:
#   https://github.com/apcamargo/RNAsamba
#
#   Rnasamba is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#   Contact: antoniop.camargo@gmail.com

import sys

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Concatenate, LeakyReLU, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2


class RNAsambaAttention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(RNAsambaAttention, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        if input_mask is not None:
            return input_mask
        else:
            return None

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.num_channels_input = input_shape[0][1]
        self.fulloutput = input_shape[2][1]
        self.w_1 = self.add_weight(
            shape=(self.num_channels_input, 2),
            initializer='glorot_normal',
            trainable=True,
            name='w_1',
        )
        self.w_2 = self.add_weight(
            shape=(1, 2), initializer='glorot_normal', trainable=True, name='w_2'
        )
        super(RNAsambaAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        orf_length_branch = inputs[0]
        first_branch = inputs[1]
        second_branch = inputs[2]
        alpha = tf.matmul(orf_length_branch, self.w_1) + self.w_2
        alpha = tf.nn.softmax(alpha)
        weighted_first_branch = tf.multiply(
            tf.expand_dims(alpha[:, 0], axis=-1), first_branch
        )
        weighted_second_branch = tf.multiply(
            tf.expand_dims(alpha[:, 1], axis=-1), second_branch
        )
        attention_output = weighted_first_branch + weighted_second_branch
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.fulloutput


def IGLOO1D(
    input_layer,
    nb_patches,
    nb_filters_conv1d,
    patch_size=4,
    padding_style='causal',
    add_batchnorm=False,
    nb_stacks=1,
    l2reg=0.00001,
    conv1d_kernel=3,
    max_pooling_kernel=1,
    dropout=0.0,
    build_backbone=False,
    dilf_factor=3,
):
    igloo1d = IGLOO(
        input_layer,
        nb_patches,
        nb_filters_conv1d,
        patch_size=patch_size,
        padding_style=padding_style,
        add_batchnorm=add_batchnorm,
        nb_stacks=nb_stacks,
        l2reg=l2reg,
        conv1d_kernel=conv1d_kernel,
        max_pooling_kernel=max_pooling_kernel,
        dropout=dropout,
        build_backbone=build_backbone,
        dilf_factor=dilf_factor,
    )
    return igloo1d


def IGLOO(
    input_layer,
    nb_patches,
    nb_filters_conv1d,
    patch_size=4,
    padding_style='causal',
    add_batchnorm=False,
    nb_stacks=1,
    l2reg=0.00001,
    conv1d_kernel=3,
    max_pooling_kernel=1,
    dropout=0.0,
    build_backbone=True,
    dilf_factor=3,
):
    layers = []
    x = Conv1D(nb_filters_conv1d, conv1d_kernel, padding=padding_style)(input_layer)
    if add_batchnorm:
        x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = SpatialDropout1D(dropout)(x)
    x = MaxPooling1D(pool_size=max_pooling_kernel, strides=None, padding='valid')(x)
    layers.append(
        PatchyLayerCNNTopLast(
            patch_size, nb_patches, dropout, build_backbone=build_backbone, l2reg=l2reg
        )(x)
    )
    if nb_stacks > 1:
        for extra_l in range(nb_stacks - 1):
            x = Conv1D(
                nb_filters_conv1d,
                conv1d_kernel,
                padding=padding_style,
                dilation_rate=dilf_factor,
            )(x)
            if add_batchnorm:
                x = BatchNormalization(axis=-1)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = SpatialDropout1D(dropout)(x)
            layers.append(
                PatchyLayerCNNTopLast(
                    patch_size,
                    nb_patches,
                    dropout,
                    build_backbone=build_backbone,
                    l2reg=l2reg,
                )(x)
            )
    if nb_stacks > 1:
        mpi = Concatenate()(layers)
    else:
        mpi = layers[0]
    return mpi


class PatchyLayerCNNTopLast(Layer):
    def __init__(
        self,
        patch_size,
        nb_patches,
        dropout,
        initializer='glorot_normal',
        build_backbone=True,
        l2reg=0.000001,
        activation='relu',
        **kwargs
    ):
        self.supports_masking = True
        self.nb_patches = nb_patches
        self.patch_size = patch_size
        self.dropout = dropout
        self.initializer = initializer
        self.kernel_causal = 4
        self.activation = activation
        self.l2reg = l2reg
        self.outsize = 100
        self.build_backbone = build_backbone
        super(PatchyLayerCNNTopLast, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        if input_mask is not None:
            return input_mask
        else:
            return None

    def PatchyLayerCNNTopLast_initializer(self, shape, dtype=None):
        m = gen_filters_igloo_newstyle1Donly(
            self.patch_size,
            self.nb_patches,
            self.vector_size,
            self.num_channels_input,
            build_backbone=self.build_backbone,
        )
        m.astype(int)
        return m

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.vector_size = input_shape[1]
        self.num_channels_input = input_shape[2]
        self.mshapy = (
            int(self.nb_patches),
            int(self.patch_size * self.num_channels_input),
            2,
        )
        self.patches = self.add_weight(
            shape=(int(self.nb_patches), self.patch_size, 1),
            initializer=self.PatchyLayerCNNTopLast_initializer,
            trainable=False,
            name='random_patches',
            dtype=np.int32,
        )
        self.w_mult = self.add_weight(
            shape=(1, self.nb_patches, self.patch_size, self.num_channels_input),
            initializer=self.initializer,
            trainable=True,
            regularizer=l2(self.l2reg),
            name='w_mult',
        )
        self.w_bias = self.add_weight(
            shape=(1, int(self.nb_patches / 1)),
            initializer=self.initializer,
            trainable=True,
            regularizer=l2(self.l2reg),
            name='w_bias',
        )
        super(PatchyLayerCNNTopLast, self).build(input_shape)

    def call(self, y, mask=None):
        y = tf.transpose(y, [1, 2, 0])
        M = tf.gather_nd(y, self.patches)
        patch_tensor = tf.transpose(M, [3, 0, 1, 2])
        mpi = tf.multiply(self.w_mult, patch_tensor)
        mpi = tf.reduce_sum(mpi, axis=[2, 3])
        mpi = tf.add(mpi, self.w_bias)
        mpi = LeakyReLU(alpha=0.1)(mpi)
        mpi = Dropout(self.dropout)(mpi)
        return mpi

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.nb_patches


def gen_filters_igloo_newstyle1Donly(
    patch_size,
    nb_patches,
    vector_size,
    num_channels_input_reduced,
    nb_stacks=1,
    build_backbone=True,
    consecutive=False,
    nb_sequences=-1,
):
    out = []
    vector_size = int(vector_size)
    for step in range(vector_size):
        if step != vector_size - 1:
            continue
        collect = []
        if step < patch_size:
            for kk in range(nb_patches):
                randy_H = np.random.choice(range(step + 1), patch_size, replace=True)
                first = []
                for pp in randy_H:
                    first.append([pp])
                collect.append(first)
            out.append(collect)
        else:
            if build_backbone:
                maximum_its = int((step / (patch_size - 1)) + 1)
                if maximum_its > nb_patches:
                    print('nb_patches too small, recommende above:', maximum_its)
                    sys.exit()
                for jj in range(maximum_its):
                    if iter == 0:
                        randy_H = [step - pp for pp in range(patch_size)]
                    else:
                        randy_H = [
                            max(step - (jj * (patch_size - 1)) - pp, 0)
                            for pp in range(patch_size)
                        ]
                    first = []
                    for pp in randy_H:
                        first.append([pp])
                    collect.append(first)
                rest_iters = max(nb_patches - maximum_its, 0)
                for itero in range(rest_iters):
                    if not consecutive:
                        randy_B = np.random.choice(
                            range(step + 1), patch_size, replace=False
                        )
                    else:
                        uniq = np.random.choice(
                            range(max(0, step + 1 - patch_size + 1)), 1, replace=False
                        )
                        randy_B = [uniq[0] + pp for pp in range(patch_size)]
                    first = []
                    sorting = True
                    if sorting:
                        randy_B = sorted(randy_B)
                    for pp in randy_B:
                        first.append([pp])
                    collect.append(first)
                collect = np.stack(collect)
                out.append(collect)
            else:
                for itero in range(nb_patches):
                    if not consecutive:
                        randy_B = np.random.choice(
                            range(step + 1), patch_size, replace=False
                        )
                    else:
                        uniq = np.random.choice(
                            range(max(0, step + 1 - patch_size + 1)), 1, replace=False
                        )
                        randy_B = [uniq[0] + pp for pp in range(patch_size)]
                    first = []
                    sorting = True
                    if sorting:
                        randy_B = sorted(randy_B)
                    for pp in randy_B:
                        first.append([pp])
                    collect.append(first)
                collect = np.stack(collect)
                out.append(collect)
    out = np.stack(out)
    out = np.squeeze(out, axis=0)
    return out

    def compute_output_shape(self, input_shape):
        return input_shape
