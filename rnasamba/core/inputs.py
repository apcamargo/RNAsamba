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

from keras.preprocessing.sequence import pad_sequences

from rnasamba.core import sequences


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
        self.protein_seqs = [orf[2] for orf in self._orfs]
        self.maxlen = maxlen
        self.protein_maxlen = int(maxlen/3)
        self.nucleotide_input = self.get_nucleotide_input()
        self.kmer_frequency_input = self.get_kmer_frequency_input()
        self.orf_indicator_input = self.get_orf_indicator_input()
        self.protein_input = self.get_protein_input()
        self.aa_frequency_input = self.get_aa_frequency_input()
        self.sequence_name = [seq[1] for seq in self._nucleotide_sequences]

    def get_orfs(self):
        orfs = [sequences.longest_orf(seq[0]) for seq in self._nucleotide_sequences]
        return orfs

    def get_nucleotide_input(self):
        nucleotide_input = [i[0] for i in self._tokenized_sequences]
        nucleotide_input = pad_sequences(nucleotide_input, padding='post', maxlen=self.maxlen)
        return nucleotide_input

    def get_kmer_frequency_input(self):
        kmer_frequency_input = sequences.kmer_frequency(self._nucleotide_sequences)
        return kmer_frequency_input

    def get_orf_indicator_input(self):
        orf_indicator_input = sequences.orf_indicator(self._orfs, self.maxlen)
        return orf_indicator_input

    def get_protein_input(self):
        protein_input = []
        for protein_seq in self.protein_seqs:
            protein_seq = protein_seq.lower()
            protein_numeric = [self._aa_dict[aa] for aa in protein_seq]
            protein_input.append(protein_numeric)
        protein_input = pad_sequences(protein_input, padding='post', maxlen=self.protein_maxlen)
        return protein_input

    def get_aa_frequency_input(self):
        aa_frequency_input = sequences.aa_frequency(self._aa_dict, self._orfs)
        return aa_frequency_input
