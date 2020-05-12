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
from rnasamba.core import kmer, orf, sequences


class RNAsambaInput:
    def __init__(self, fasta_file, maxlen=3000):
        self._nucleotide_seqs, self._token_seqs, self.seqs_names = sequences.read_fasta(
            fasta_file
        )
        self._aa_dict = {
            'A': 4,
            'C': 18,
            'D': 12,
            'E': 3,
            'F': 14,
            'G': 5,
            'H': 16,
            'I': 13,
            'K': 9,
            'L': 1,
            'M': 19,
            'N': 15,
            'P': 6,
            'Q': 11,
            'R': 8,
            'S': 2,
            'T': 10,
            'V': 7,
            'W': 20,
            'X': 21,
            'Y': 17,
        }
        self._orfs = self.get_orfs()
        self.protein_seqs = [orf[2] for orf in self._orfs]
        self.maxlen = maxlen
        self.protein_maxlen = int(maxlen / 3)
        self.nucleotide_input = self.get_nucleotide_input()
        self.kmer_frequency_input = self.get_kmer_frequency_input()
        self.orf_indicator_input = self.get_orf_indicator_input()
        self.protein_input = self.get_protein_input()
        self.aa_frequency_input = self.get_aa_frequency_input()

    def get_orfs(self):
        return orf.longest_orf_array(self._nucleotide_seqs)

    def get_nucleotide_input(self):
        return pad_sequences(
                self._token_seqs, padding='post', maxlen=self.maxlen
            )

    def get_kmer_frequency_input(self):
        return kmer.kmer_frequencies_array(self._nucleotide_seqs)

    def get_orf_indicator_input(self):
        return sequences.orf_indicator(self._orfs, self.maxlen)

    def get_protein_input(self):
        protein_input = []
        for protein_seq in self.protein_seqs:
            protein_numeric = [self._aa_dict[aa] for aa in protein_seq]
            protein_input.append(protein_numeric)
        protein_input = pad_sequences(
            protein_input, padding='post', maxlen=self.protein_maxlen
        )
        return protein_input

    def get_aa_frequency_input(self):
        return sequences.aa_frequency(self._aa_dict, self._orfs)
