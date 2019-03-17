import logging

import numpy as np

import tensorflow as tf
from keras import optimizers
from keras.layers import (Activation, Concatenate, Dense, Dropout, Embedding,
                          Input, Lambda)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from rnasamba.inputs import RNAsambaInput
from rnasamba.miniigloo import IGLOO1D, RNAsambaAttention


class RNAsambaClassificationModel:
    def __init__(self, fasta_file, weights_file, verbose=0):
        if verbose > 0:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
        logger = logging.getLogger()
        logger.info('1. Computing network inputs.')
        self.input = RNAsambaInput(fasta_file)
        self.maxlen = self.input.maxlen
        self.protein_maxlen = self.input.protein_maxlen
        self.sequence_name = self.input.sequence_name
        self.input_dict = {'nucleotide_layer': self.input.nucleotide_input,
                           'orf_indicator_layer': self.input.orf_indicator_input,
                           'kmer_frequency_layer': self.input.kmer_frequency_input,
                           'protein_layer': self.input.protein_input,
                           'aa_frequency_layer': self.input.aa_frequency_input}
        logger.info('2. Building the model.')
        self.model = get_rnasamba_model(self.maxlen, self.protein_maxlen)
        logger.info('3. Loading network weights.')
        self.model.load_weights(weights_file)
        logger.info('4. Classifying sequences.')
        self.predictions = self.model.predict(self.input_dict)

    def write_classification_output(self, output_file):
        coding_score = self.predictions[:, 1]
        classification_label = np.argmax(self.predictions, axis=1)
        classification_label = ['coding' if i == 1 else 'noncoding' for i in classification_label]
        with open(output_file, 'w') as handle:
            handle.write('sequence_name\tcoding_score\tclassification\n')
            for i in range(len(classification_label)):
                handle.write(self.sequence_name[i])
                handle.write('\t')
                handle.write('{:.5f}'.format(coding_score[i]))
                handle.write('\t')
                handle.write(classification_label[i])
                handle.write('\n')


class RNAsambaTrainModel:
    def __init__(self, coding_file, noncoding_file, batch_size=128, epochs=40, verbose=0):
        if verbose > 0:
            verbose_keras = verbose - 1
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        else:
            verbose_keras = verbose
            logging.basicConfig(level=logging.WARNING, format='%(message)s')
        logger = logging.getLogger()
        logger.info('1. Computing network inputs.')
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
        logger.info('2. Building the model.')
        self.model = get_rnasamba_model(self.maxlen, self.protein_maxlen)
        logger.info('3. Training the network.')
        self.model.fit(self.input_dict, self.labels,
                       shuffle=True, batch_size=batch_size, epochs=epochs, verbose=verbose_keras)


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
