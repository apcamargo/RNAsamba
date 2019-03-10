#!/usr/bin/env python

import argparse
import sys

from rnasamba import RNAsambaTrainModel


def main(coding_file, noncoding_file, output_file, batch_size, epochs, verbose):
    """Train a classification model from training data and saves the weights into a HDF5 file."""
    trained = RNAsambaTrainModel(coding_file, noncoding_file,
                                 batch_size=batch_size, epochs=epochs, verbose=verbose)
    trained.model.save_weights(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify sequences from a input FASTA file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('coding_file',
                        help='input FASTA file containing sequences of protein-coding transcripts.')
    parser.add_argument('noncoding_file',
                        help='input FASTA file containing sequences of noncoding transcripts.')
    parser.add_argument('output_file',
                        help='output HDF5 file containing weights of the newly trained RNAsamba network.')
    parser.add_argument('--batch_size',
                        default=128, type=int, help='number of samples per gradient update.')
    parser.add_argument('--epochs',
                        default=40, type=int, help='number of epochs to train the model.')
    parser.add_argument('--verbose',
                        default=0, type=int, help='show the progress of the training.')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(**vars(args))
