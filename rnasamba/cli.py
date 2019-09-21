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

import argparse
import sys

from rnasamba import RNAsambaClassificationModel, RNAsambaTrainModel


def classify(args):
    """Classify sequences from a input FASTA file."""
    classification = RNAsambaClassificationModel(
        args.fasta_file, args.weights, verbose=args.verbose
    )
    classification.write_classification_output(args.output_file)
    if args.protein_fasta:
        classification.output_protein_fasta(args.protein_fasta)


def train(args):
    """Train a classification model from training data and saves the weights into a HDF5 file."""
    trained = RNAsambaTrainModel(
        args.coding_file,
        args.noncoding_file,
        early_stopping=args.early_stopping,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
    )
    trained.model.save_weights(args.output_file)


def classify_cli(parser):
    parser.set_defaults(func=classify)
    parser.add_argument(
        'output_file',
        help='output TSV file containing the results of the classification.',
    )
    parser.add_argument(
        'fasta_file', help='input FASTA file containing transcript sequences.'
    )
    parser.add_argument(
        'weights',
        nargs='+',
        help='input HDF5 file(s) containing weights of a trained RNAsamba network (if more than a file is provided, an ensembling of the models will be performed).',
    )
    parser.add_argument(
        '-p',
        '--protein_fasta',
        help='output FASTA file containing translated sequences for the predicted coding ORFs.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=0,
        type=int,
        choices=[0, 1],
        help='print the progress of the classification. 0 = silent, 1 = current step.',
    )


def train_cli(parser):
    parser.set_defaults(func=train)
    parser.add_argument(
        'output_file',
        help='output HDF5 file containing weights of the newly trained RNAsamba network.',
    )
    parser.add_argument(
        'coding_file',
        help='input FASTA file containing sequences of protein-coding transcripts.',
    )
    parser.add_argument(
        'noncoding_file',
        help='input FASTA file containing sequences of noncoding transcripts.',
    )
    parser.add_argument(
        '-s',
        '--early_stopping',
        default=0,
        type=int,
        help='number of epochs after lowest validation loss before stopping training (a fraction of 0.1 of the training set is set apart for validation and the model with the lowest validation loss will be saved).',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=128,
        type=int,
        help='number of samples per gradient update.',
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=40,
        type=int,
        help='number of epochs to train the model.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help='print the progress of the training. 0 = silent, 1 = current step, 2 = progress bar, 3 = one line per epoch.',
    )


def cli():
    parser = argparse.ArgumentParser(
        description='Coding potential calculation using deep learning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers()
    classify_parser = subparsers.add_parser(
        'classify',
        help='classify sequences from a input FASTA file.',
        description='Classify sequences from a input FASTA file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    classify_cli(classify_parser)
    train_parser = subparsers.add_parser(
        'train',
        help='train a new classification model.',
        description='Train a new classification model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(train_parser)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'classify':
            classify_parser.print_help()
            sys.exit(0)
        elif sys.argv[1] == 'train':
            train_parser.print_help()
            sys.exit(0)
    args = parser.parse_args()
    args.func(args)
