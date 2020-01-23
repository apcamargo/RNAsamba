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

from collections import Counter

import numpy as np
from Bio import SeqIO
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def read_fasta(filename):
    seqs = []
    seqs_tokenized = []
    seqs_names = []
    with open(filename) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            sequence_str = str(record.seq).upper().replace('U', 'T')
            if len(sequence_str) < 4:
                continue
            else:
                sequence_name = record.description
                seqs.append(sequence_str)
                seqs_tokenized.append(tokenize_dna(sequence_str))
                seqs_names.append(sequence_name)
    return seqs, seqs_tokenized, seqs_names


def tokenize_dna(sequence):
    lookup = {'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
    if not sequence:
        token = [0]
    else:
        token = [lookup[c] for c in sequence if c in lookup]
    return token


def orf_indicator(orfs, maxlen):
    orf_indicator = []
    for orf in orfs:
        orf_vector = np.zeros(maxlen)
        if orf[0] > 1:
            orf_vector[orf[1] : orf[1] + orf[0] * 3] = 1
        orf_indicator.append(orf_vector)
    orf_indicator = pad_sequences(orf_indicator, padding='post', maxlen=maxlen)
    orf_indicator = to_categorical(orf_indicator, num_classes=2)
    orf_indicator = np.stack(orf_indicator)
    return orf_indicator


def aa_frequency(aa_dict, orfs):
    aa_numeric = list(range(1, 22))
    aa_frequency = []
    for orf in orfs:
        protein_seq = orf[2]
        protein_numeric = [aa_dict[aa] for aa in protein_seq]
        aa_count = Counter(protein_numeric)
        protein_len = max(len(protein_numeric), 1)
        freq = []
        for aa in aa_numeric:
            if aa in aa_count:
                freq.append(aa_count[aa] / protein_len)
            else:
                freq.append(0.0)
        aa_frequency.append(freq)
    aa_frequency = np.stack(aa_frequency)
    return aa_frequency
