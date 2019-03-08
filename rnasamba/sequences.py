import re

from Bio import Seq, SeqIO


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


def get_longest_orf(input_seq):
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


def count_kmers(read, k):
    counts = {}
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
        kmer = read[i:i+k]
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts
