import itertools
from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from rnasamba import sequences


class RNAsambaInput:
    def __init__(self, fasta_file, maxlen=3000):
        self._tokenized_sequences = sequences.read_fasta(fasta_file, tokenize=True)
        self._nucleotide_sequences = sequences.read_fasta(fasta_file, tokenize=False)
        self._aa_dict = {
            'a': 4, 'c': 18, 'd': 12, 'e': 3, 'f': 14, 'g': 5, 'h': 16, 'i': 13, 'k': 9, 'l': 1,
            'm': 19, 'n': 15, 'p': 6, 'q': 11, 'r': 8, 's': 2, 't': 10, 'v': 7, 'w': 20, 'x': 21,
            'y': 17
        }
        self._orfs = self.get_orfs()
        self.maxlen = maxlen
        self.protein_maxlen = int(maxlen/3)
        self.nucleotide_input = self.get_nucleotide_input()
        self.kmer_frequency_input = self.get_kmer_frequency_input()
        self.orf_indicator_input = self.get_orf_indicator_input()
        self.protein_input = self.get_protein_input()
        self.aa_frequency_input = self.get_aa_frequency_input()
        self.sequence_name = [seq[1] for seq in self._nucleotide_sequences]

    def get_orfs(self):
        orfs = [sequences.get_longest_orf(seq[0]) for seq in self._nucleotide_sequences]
        return orfs

    def get_nucleotide_input(self):
        nucleotide_input = [i[0] for i in self._tokenized_sequences]
        nucleotide_input = pad_sequences(nucleotide_input, padding='post', maxlen=self.maxlen)
        return nucleotide_input

    def get_kmer_frequency_input(self):
        kmer_frequency_input = []
        bases = ['A', 'T', 'C', 'G']
        kmer_lengths = [2, 3, 4]
        for nucleotide_seq in self._nucleotide_sequences:
            matches = [bases, bases]
            sequence_kmer_frequency = []
            for current_length in kmer_lengths:
                current_seq = nucleotide_seq[0]
                total_kmers = len(current_seq)-(current_length-1)
                kmer_count = sequences.count_kmers(current_seq, current_length)
                for match in itertools.product(*matches):
                    current_kmer = ''.join(match)
                    if current_kmer in kmer_count:
                        sequence_kmer_frequency.append(kmer_count[current_kmer]/(total_kmers))
                    else:
                        sequence_kmer_frequency.append(0)
                matches.append(bases)
            kmer_frequency_input.append(sequence_kmer_frequency)
        kmer_frequency_input = np.stack(kmer_frequency_input)
        return kmer_frequency_input

    def get_orf_indicator_input(self):
        orf_indicator_input = []
        for orf in self._orfs:
            orf_vector = np.zeros(self.maxlen)
            if orf[0] > 1:
                orf_vector[orf[1]:orf[1]+orf[0]*3] = 1
            orf_indicator_input.append(orf_vector)
        orf_indicator_input = pad_sequences(orf_indicator_input, padding='post', maxlen=self.maxlen)
        orf_indicator_input = to_categorical(orf_indicator_input, num_classes=2)
        orf_indicator_input = np.stack(orf_indicator_input)
        return orf_indicator_input

    def get_protein_input(self):
        protein_input = []
        for orf in self._orfs:
            protein_seq = orf[2].lower()
            protein_numeric = [self._aa_dict[aa] for aa in protein_seq]
            protein_input.append(protein_numeric)
        protein_input = pad_sequences(protein_input, padding='post', maxlen=self.protein_maxlen)
        return protein_input

    def get_aa_frequency_input(self):
        aa_numeric = list(self._aa_dict.values())
        aa_numeric.sort()
        aa_frequency_input = []
        for orf in self._orfs:
            protein_seq = orf[2].lower()
            protein_numeric = [self._aa_dict[aa] for aa in protein_seq]
            aa_count = Counter(protein_numeric)
            protein_len = max(len(protein_numeric), 1)
            aa_frequency = []
            for aa in aa_numeric:
                if aa in aa_count:
                    aa_frequency.append(aa_count[aa]/protein_len)
                else:
                    aa_frequency.append(0.)
            aa_frequency_input.append(aa_frequency)
        aa_frequency_input = np.stack(aa_frequency_input)
        return aa_frequency_input
