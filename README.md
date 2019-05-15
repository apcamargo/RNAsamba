# RNAsamba

A tool for computing the coding potential of RNA transcript sequences using deep learning classification model

## Overview


## Installation

```
# Using pip
$ pip install rnasamba

# using conda
$ conda install -c bioconda rnasamba
```

## Download the pre-trained model

We provide a HDF5 file containing the weights of a model trained with human data. This model achieves high classification performance even in distant species (see reference). You can download the file by executing the following command:

```
$ curl -O https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/weights_master_model.hdf5
```

In case you want to train your model, you can follow the steps described in the section below.

## Usage



## Reference
> Camargo, Antonio P., Vsevolod Sourkov, and Marcelo F. Carazzolle. "RNAsamba: coding potential assessment using ORF and whole transcript sequence information." *BioRxiv* (2019).