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

import logging

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Concatenate, Dense, Dropout, Embedding,
                          Input, Lambda)
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from rnasamba.core.inputs import RNAsambaInput
from rnasamba.core.miniigloo import IGLOO1D, RNAsambaAttention


class RNAsambaClassificationModel:
    def __init__(self, fasta_file, weights, verbose=0):
        if verbose > 0:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
        logger = logging.getLogger()
        logger.info('- Computing network inputs.')
        self.input = RNAsambaInput(fasta_file)
        self.maxlen = self.input.maxlen
        self.protein_maxlen = self.input.protein_maxlen
        self.sequence_name = self.input.sequence_name
        self.protein_seqs = self.input.protein_seqs
        self.input_dict = {'nucleotide_layer': self.input.nucleotide_input,
                           'orf_indicator_layer': self.input.orf_indicator_input,
                           'kmer_frequency_layer': self.input.kmer_frequency_input,
                           'protein_layer': self.input.protein_input,
                           'aa_frequency_layer': self.input.aa_frequency_input}
        if len(weights) == 1:
            logger.info('- Building the model.')
            model = get_rnasamba_model(self.maxlen, self.protein_maxlen)
            logger.info('- Loading network weights.')
            model.load_weights(weights[0])
            logger.info('- Classifying sequences.')
            self.predictions = model.predict(self.input_dict)
            self.coding_score = self.predictions[:, 1]
            self.classification_label = np.argmax(self.predictions, axis=1)
            self.classification_label = ['coding' if i == 1 else 'noncoding' for i in self.classification_label]
        else:
            logger.info('- Building the models.')
            n_models = len(weights)
            models = [get_rnasamba_model(self.maxlen, self.protein_maxlen) for i in range(n_models)]
            logger.info('- Loading network weights.')
            for i in range(n_models):
                models[i].load_weights(weights[i])
            logger.info('- Classifying sequences using an ensemble of {} models.'.format(n_models))
            self.predictions = np.average([models[i].predict(self.input_dict)
                                           for i in range(n_models)], axis=0)
            self.coding_score = self.predictions[:, 1]
            self.classification_label = np.argmax(self.predictions, axis=1)
            self.classification_label = ['coding' if i == 1 else 'noncoding' for i in self.classification_label]

    def write_classification_output(self, output_file):
        with open(output_file, 'w') as handle:
            handle.write('sequence_name\tcoding_score\tclassification\n')
            for i in range(len(self.classification_label)):
                handle.write(self.sequence_name[i])
                handle.write('\t')
                handle.write('{:.5f}'.format(self.coding_score[i]))
                handle.write('\t')
                handle.write(self.classification_label[i])
                handle.write('\n')

    def output_protein_fasta(self, protein_fasta):
        with open(protein_fasta, 'w') as handle:
            for i in range(len(self.classification_label)):
                if self.classification_label[i] == 'coding':
                    if self.protein_seqs[i]:
                        handle.write('>')
                        handle.write(self.sequence_name[i])
                        handle.write('\n')
                        handle.write(self.protein_seqs[i])
                        handle.write('\n')


class RNAsambaTrainModel:
    def __init__(self, coding_file, noncoding_file, early_stopping=0, batch_size=128, epochs=40, verbose=0):
        if verbose > 0:
            verbose_keras = verbose - 1
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        else:
            verbose_keras = verbose
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
        logger = logging.getLogger()
        logger.info('- Computing network inputs.')
        self.coding_input = RNAsambaInput(coding_file)
        self.noncoding_input = RNAsambaInput(noncoding_file)
        self.maxlen = self.coding_input.maxlen
        self.protein_maxlen = self.coding_input.protein_maxlen
        self.labels = np.repeat([[0, 1], [1, 0]], [len(self.coding_input.sequence_name), len(
            self.noncoding_input.sequence_name)], axis=0)
        self.input_dict = {'nucleotide_layer': np.concatenate(
            [self.coding_input.nucleotide_input, self.noncoding_input.nucleotide_input]),
            'orf_indicator_layer': np.concatenate(
                [self.coding_input.orf_indicator_input, self.noncoding_input.orf_indicator_input]),
            'kmer_frequency_layer': np.concatenate(
                [self.coding_input.kmer_frequency_input, self.noncoding_input.kmer_frequency_input]),
            'protein_layer': np.concatenate(
                [self.coding_input.protein_input, self.noncoding_input.protein_input]),
            'aa_frequency_layer': np.concatenate(
                [self.coding_input.aa_frequency_input, self.noncoding_input.aa_frequency_input])}
        logger.info('- Building the model.')
        self.model = get_rnasamba_model(self.maxlen, self.protein_maxlen)
        logger.info('- Training the network.')
        if early_stopping > 0:
            logger.info('- Using early stopping. 10% of the training data will be set aside for validation.')
            seed = np.random.randint(0, 50)
            np.random.seed(seed)
            np.random.shuffle(self.labels)
            for i in self.input_dict:
                np.random.seed(seed)
                np.random.shuffle(self.input_dict[i])
            early_stop_call = EarlyStopping(monitor='val_loss', patience=early_stopping,
                                            restore_best_weights=True, verbose=verbose_keras)
            self.model.fit(self.input_dict, self.labels, callbacks=[early_stop_call],
                           validation_split=0.1, shuffle=True, batch_size=batch_size, epochs=epochs,
                           verbose=verbose_keras)
        else:
            self.model.fit(self.input_dict, self.labels, shuffle=True,
                           batch_size=batch_size, epochs=epochs, verbose=verbose_keras)


def get_rnasamba_model(maxlen, protein_maxlen):
    nucleotide_layer = Input(name='nucleotide_layer', shape=(maxlen,))
    kmer_frequency_layer = Input(name='kmer_frequency_layer', shape=(336,))
    orf_indicator_layer = Input(name='orf_indicator_layer', shape=(maxlen, 2))
    protein_layer = Input(name='protein_layer', shape=(protein_maxlen,))
    aa_frequency_layer = Input(name='aa_frequency_layer', shape=(21,))

    # Nucleotide branch (first branch):
    emb_mat_nuc = np.random.normal(0, 1, (5, 4))
    for i in range(1, 5):
        vector = np.zeros(4)
        vector[i-1] = 1
        emb_mat_nuc[i] = vector
    embedding_layer_nucleotide = Embedding(input_dim=emb_mat_nuc.shape[0], output_dim=4,
                                           weights=[emb_mat_nuc], trainable=False, mask_zero=False)
    emb_nuc = embedding_layer_nucleotide(nucleotide_layer)
    nucleotide_branch_1 = IGLOO1D(
        emb_nuc, nb_patches=900, nb_filters_conv1d=6, padding_style='same',
        add_batchnorm=True, nb_stacks=6, conv1d_kernel=1, l2reg=0.01, DR=0.30, max_pooling_kernel=8)
    nucleotide_branch_2 = IGLOO1D(
        emb_nuc, nb_patches=900, nb_filters_conv1d=6, padding_style='same',
        add_batchnorm=True, nb_stacks=6, conv1d_kernel=3, l2reg=0.01, DR=0.30, max_pooling_kernel=8)
    first_branch = Concatenate(axis=-1)([nucleotide_branch_1, nucleotide_branch_2])
    first_branch = Dense(128)(first_branch)
    first_branch = BatchNormalization()(first_branch)
    first_branch = Activation('relu')(first_branch)
    first_branch = Dropout(0.30)(first_branch)

    # k-mer frequency branch:
    kmer_branch = Dense(128)(kmer_frequency_layer)
    kmer_branch = BatchNormalization()(kmer_branch)
    kmer_branch = Activation('relu')(kmer_branch)
    kmer_branch = Dropout(0.30)(kmer_branch)
    kmer_branch = Dense(64)(kmer_branch)
    kmer_branch = BatchNormalization()(kmer_branch)
    kmer_branch = Activation('relu')(kmer_branch)
    kmer_branch = Dropout(0.30)(kmer_branch)

    # ORF branch:
    orf_length_branch = Lambda(lambda d: tf.reduce_sum(
        d[:, :, 1]/maxlen, axis=-1))(orf_indicator_layer)
    orf_length_branch = Lambda(lambda d: tf.expand_dims(d, axis=-1))(orf_length_branch)
    orf_length_branch = Lambda(lambda d: tf.tile(d, [1, 64]))(orf_length_branch)

    # Protein branch:
    emb_mat_prot = np.random.random((22, 21))
    emb_mat_prot[0] = np.zeros(21)
    for i in range(1, 22):
        vector = np.zeros(21)
        vector[i-1] = 1
        emb_mat_prot[i] = vector
    embedding_layer_protein = Embedding(input_dim=emb_mat_prot.shape[0], output_dim=21,
                                        weights=[emb_mat_prot], trainable=False, mask_zero=False)
    emb_prot = embedding_layer_protein(protein_layer)
    protein_branch = IGLOO1D(
        emb_prot, nb_patches=600, nb_filters_conv1d=8, padding_style='same',
        add_batchnorm=True, nb_stacks=6, conv1d_kernel=3, l2reg=0.13, DR=0.30, max_pooling_kernel=2)
    protein_branch = Dense(64)(protein_branch)
    protein_branch = BatchNormalization()(protein_branch)
    protein_branch = Activation('relu')(protein_branch)
    protein_branch = Dropout(0.30)(protein_branch)

    # Aminoacid frequency branch:
    aa_branch = Dense(36)(aa_frequency_layer)
    aa_branch = BatchNormalization()(aa_branch)
    aa_branch = Activation('relu')(aa_branch)
    aa_branch = Dropout(0.30)(aa_branch)
    aa_branch = Dense(36)(aa_branch)
    aa_branch = BatchNormalization()(aa_branch)
    aa_branch = Activation('relu')(aa_branch)
    aa_branch = Dropout(0.30)(aa_branch)

    # Second branch:
    second_branch = Concatenate(
        axis=-1)([kmer_branch, orf_length_branch, protein_branch, aa_branch])
    second_branch = Dense(128)(second_branch)
    second_branch = BatchNormalization()(second_branch)
    second_branch = Activation('relu')(second_branch)
    second_branch = Dropout(0.30)(second_branch)

    # Attention layer:
    attention = RNAsambaAttention()([orf_length_branch, first_branch, second_branch])
    output_layer = Dense(2, activation='softmax')(attention)
    model = Model([nucleotide_layer, orf_indicator_layer, kmer_frequency_layer,
                   protein_layer, aa_frequency_layer], output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0025, clipnorm=1., decay=0.01),
                  metrics=['accuracy'])
    return model
