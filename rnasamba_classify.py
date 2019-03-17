#!/usr/bin/env python

import argparse
import sys

from rnasamba import RNAsambaClassificationModel


def main(fasta_file, weights_file, output_file, verbose):
    """Classify sequences from a input FASTA file."""
    classification = RNAsambaClassificationModel(fasta_file, weights_file, verbose=verbose)
    classification.write_classification_output(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify sequences from a input FASTA file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fasta_file',
                        help='input FASTA file containing complete sequences of protein-coding transcripts.')
    parser.add_argument('weights_file',
                        help='input HDF5 file containing weights of a trained RNAsamba network.')
    parser.add_argument('output_file',
                        help='output TSV file containing the results of the classification.')
    parser.add_argument('-v', '--verbose',
                        default=0, type=int, choices=[0, 1],
                        help='print the progress of the classification. 0 = silent, 1 = current step.')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(**vars(args))
