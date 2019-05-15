# RNAsamba

- [Overview](#overview)
- [Installation](#installation)
- [Download the pre-trained model](#download-the-pre-trained-model)
- [Usage](#usage)
  - [`rnasamba-train`](#rnasamba-train)
  - [`rnasamba-classify`](#rnasamba-classify)
- [Examples](#examples)
- [Citation](#citation)

## Overview

RNAsamba is a tool for computing the coding potential of RNA sequences using a neural network classification model. A description of the algorithm and benchmarks comparing RNAsamba to other tools can be found in our [artipapercle](#citation).

## Installation

1. Using pip:

```
$ pip install rnasamba
```

2. Using conda:

```
$ conda install -c bioconda rnasamba
```

## Download the pre-trained model

We provide a HDF5 file containing the weights of a classification model trained with human trancript sequences. This model achieves high classification performance even in transcripts of distant species (see [reference](#citation)). You can download the file by executing the following command:

```
$ curl -O https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/weights_master_model.hdf5
```

In case you want to train your own model, you can follow the steps shown in the [Examples](#examples) section.

## Usage

RNAsamba provides two commands: `rnasamba-train` and `rnasamba-classify`.

### `rnasamba-train`

`rnasamba-train` is the command for training a new classification model from a training dataset and saving the network weights into an HDF5 file. The user can specify the batch size (`--batch_size`) and the number of training epochs (`--epochs`). The user can also choose to activate early stopping, which reduces training time and can help avoiding overfitting.

```
usage: rnasamba-train [-h] [-s EARLY_STOPPING] [-b BATCH_SIZE] [-e EPOCHS]
                      [-v {0,1,2,3}]
                      output_file coding_file noncoding_file

Train a new classification model.

positional arguments:
  output_file           output HDF5 file containing weights of the newly
                        trained RNAsamba network.
  coding_file           input FASTA file containing sequences of protein-
                        coding transcripts.
  noncoding_file        input FASTA file containing sequences of noncoding
                        transcripts.

optional arguments:
  -h, --help            show this help message and exit
  -s EARLY_STOPPING, --early_stopping EARLY_STOPPING
                        number of epochs after lowest validation loss before
                        stopping training (a fraction of 0.1 of the training
                        set is set apart for validation and the model with the
                        lowest validation loss will be saved). (default: 0)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of samples per gradient update. (default: 128)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the model. (default: 40)
  -v {0,1,2,3}, --verbose {0,1,2,3}
                        print the progress of the training. 0 = silent, 1 =
                        current step, 2 = progress bar, 3 = one line per
                        epoch. (default: 0)
```

### `rnasamba-classify`

`rnasamba-classify` is the command for computing the coding potential of transcripts contained in an input FASTA file and classifying them into coding or non-coding. Optionally, the user can specify an output FASTA file (`--protein_fasta`) in which RNAsamba will write the translated sequences of the predicted coding ORFs. If multiple weight files are provided, RNAsamba will ensemble their predictions into a single output.

```
usage: rnasamba-classify [-h] [-p PROTEIN_FASTA] [-v {0,1}]
                         output_file fasta_file weights [weights ...]

Classify sequences from a input FASTA file.

positional arguments:
  output_file           output TSV file containing the results of the
                        classification.
  fasta_file            input FASTA file containing transcript sequences.
  weights               input HDF5 file(s) containing weights of a trained
                        RNAsamba network (if more than a file is provided, an
                        ensembling of the models will be performed).

optional arguments:
  -h, --help            show this help message and exit
  -p PROTEIN_FASTA, --protein_fasta PROTEIN_FASTA
                        output FASTA file containing translated sequences for
                        the predicted coding ORFs. (default: None)
  -v {0,1}, --verbose {0,1}
                        print the progress of the classification. 0 = silent,
                        1 = current step. (default: 0)
```

## Examples

1. Training a new classification model using *M. musculus* data downloaded from GENCODE:

```
$ rnasamba-train mouse_model.hdf5 -v 2 gencode.vM21.pc_transcripts.fa gencode.vM21.lncRNA_transcripts.fa

```

2. Classifying sequences using our pre-trained model (`weights_master_model.hdf5`) and saving the predicted proteins into a FASTA file:

```
$ rnasamba-classify -p predicted_proteins.fa classification.tsv input.fa weights_master_model.hdf5
$ head classification.tsv

sequence_name	coding_score	classification
ENSMUST00000054910	0.99022	coding
ENSMUST00000059648	0.84718	coding
ENSMUST00000055537	0.99713	coding
ENSMUST00000030975	0.85189	coding
ENSMUST00000050754	0.02638	noncoding
ENSMUST00000008011	0.14949	noncoding
ENSMUST00000061643	0.03456	noncoding
ENSMUST00000059704	0.89232	coding
ENSMUST00000036304	0.03782	noncoding
```

## Citation

> Camargo, Antonio P., Vsevolod Sourkov, and Marcelo F. Carazzolle. "RNAsamba: coding potential assessment using ORF and whole transcript sequence information." *BioRxiv* (2019).