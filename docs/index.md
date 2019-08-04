<p align="center"><img src="https://raw.githubusercontent.com/apcamargo/RNAsamba/master/logo.png" width="300rem"></p>

## Overview

RNAsamba is a tool for computing the coding potential of RNA sequences using a neural network classification model. It can be used to identify mRNAs and lncRNAs without relying on database searches. For details of RNAsamba's methods, check the [Theory](theory.md) section or read our [article](https://www.biorxiv.org/content/10.1101/620880v1).

## Web version

RNAsamba can be used through a minimal web interface that is freely [available online](https://rnasamba.lge.ibi.unicamp.br/).

## Source code

RNAsamba is an open source package distributed under the GPL-3.0 license. Its source code can be found in the [GitHub repository](https://github.com/apcamargo/RNAsamba/). The source code of the web version is also available in [GitHub](https://github.com/apcamargo/rnasamba-webapp/).

## Installation

There are two ways to install RNAsamba:

- Using pip:

```
pip install rnasamba
```

- Using conda:

```
conda install -c bioconda rnasamba
```

!!! warning ""
    If you want to use RNAsamba with a GPU, you first need to install `tensorflow-gpu>=1.5.0,<2.0`, `keras>=2.1.0`, `numpy` and `bioconda` with pip and then proceed to install `rnasamba` using the `--no-deps` argument.

## Citation

Camargo, Antonio P., Vsevolod Sourkov, and Marcelo F. Carazzolle. "[RNAsamba: coding potential assessment using ORF and whole transcript sequence information](https://www.biorxiv.org/content/10.1101/620880v1)" *BioRxiv* (2019).