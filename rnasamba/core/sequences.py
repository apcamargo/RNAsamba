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

import itertools
import re
from collections import Counter

import numpy as np
from Bio import Seq, SeqIO
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def read_fasta(filename, tokenize=False):
    seqs = []
    with open(filename) as handle:
        if tokenize:
            for record in SeqIO.parse(handle, 'fasta'):
                sequence_str = str(record.seq).upper()
                sequence_name = record.description
                seqs.append((tokenize_dna(sequence_str), sequence_name))
        else:
            for record in SeqIO.parse(handle, 'fasta'):
                sequence_str = str(record.seq).upper()
                sequence_name = record.description
                seqs.append((sequence_str, sequence_name))
    return seqs


def tokenize_dna(seq):
    lookup = dict(zip('NATCG', range(5)))
    if not seq:
        token = [0]
    else:
        token = [lookup[c] for c in seq if c in lookup]
    return token


def longest_orf(input_seq):
    start_codon = re.compile('ATG')
    longest = (0, 0, '')
    for m in start_codon.finditer(input_seq):
        putative_orf = input_seq[m.start():]
        # Add trailing Ns to make the sequence length a multiple of three:
        putative_orf = putative_orf + 'N'*(3-len(putative_orf) % 3)
        protein = Seq.Seq(putative_orf).translate(to_stop=True)
        if len(protein) > longest[0]:
            longest = (len(protein),
                       m.start(),
                       str(protein))
    return longest


def orf_indicator(orfs, maxlen):
    orf_indicator = []
    for orf in orfs:
        orf_vector = np.zeros(maxlen)
        if orf[0] > 1:
            orf_vector[orf[1]:orf[1]+orf[0]*3] = 1
        orf_indicator.append(orf_vector)
    orf_indicator = pad_sequences(orf_indicator, padding='post', maxlen=maxlen)
    orf_indicator = to_categorical(orf_indicator, num_classes=2)
    orf_indicator = np.stack(orf_indicator)
    return orf_indicator


def count_kmers(read, k):
    counts = {}
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
        kmer = read[i:i+k]
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts


def kmer_frequency(nucleotide_sequences):
    kmer_frequency = []
    bases = ['A', 'T', 'C', 'G']
    kmer_lengths = [2, 3, 4]
    for nucleotide_seq in nucleotide_sequences:
        matches = [bases, bases]
        sequence_kmer_frequency = []
        for current_length in kmer_lengths:
            current_seq = nucleotide_seq[0]
            total_kmers = len(current_seq)-(current_length-1)
            kmer_count = count_kmers(current_seq, current_length)
            for match in itertools.product(*matches):
                current_kmer = ''.join(match)
                if current_kmer in kmer_count:
                    sequence_kmer_frequency.append(kmer_count[current_kmer]/(total_kmers))
                else:
                    sequence_kmer_frequency.append(0)
            matches.append(bases)
        kmer_frequency.append(sequence_kmer_frequency)
    kmer_frequency = np.stack(kmer_frequency)
    return kmer_frequency


def aa_frequency(aa_dict, orfs):
    aa_numeric = list(aa_dict.values())
    aa_numeric.sort()
    aa_frequency = []
    for orf in orfs:
        protein_seq = orf[2].lower()
        protein_numeric = [aa_dict[aa] for aa in protein_seq]
        aa_count = Counter(protein_numeric)
        protein_len = max(len(protein_numeric), 1)
        freq = []
        for aa in aa_numeric:
            if aa in aa_count:
                freq.append(aa_count[aa]/protein_len)
            else:
                freq.append(0.)
        aa_frequency.append(freq)
    aa_frequency = np.stack(aa_frequency)
    return aa_frequency
