#!/usr/bin/env python

import argparse
import sys

from rnasamba import RNAsambaTrainModel


def main(output_file, coding_file, noncoding_file, early_stop, batch_size, epochs, verbose):
    """Train a classification model from training data and saves the weights into a HDF5 file."""
    trained = RNAsambaTrainModel(coding_file, noncoding_file, early_stop=early_stop,
                                 batch_size=batch_size, epochs=epochs, verbose=verbose)
    trained.model.save_weights(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify sequences from a input FASTA file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_file',
                        help='output HDF5 file containing weights of the newly trained RNAsamba network.')
    parser.add_argument('coding_file',
                        help='input FASTA file containing sequences of protein-coding transcripts.')
    parser.add_argument('noncoding_file',
                        help='input FASTA file containing sequences of noncoding transcripts.')
    parser.add_argument('-s', '--early_stop',
                        default=0, type=int, help='number of epochs after lowest validation loss before stopping training (a fraction of 0.1 of the train set is set apart for validation and the model with the lowest validation loss will be saved).')
    parser.add_argument('-b', '--batch_size',
                        default=128, type=int, help='number of samples per gradient update.')
    parser.add_argument('-e', '--epochs',
                        default=40, type=int, help='number of epochs to train the model.')
    parser.add_argument('-v', '--verbose',
                        default=0, type=int, choices=[0, 1, 2, 3],
                        help='print the progress of the training. 0 = silent, 1 = current step, 2 = progress bar, 3 = one line per epoch.')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(**vars(args))
