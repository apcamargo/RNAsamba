# Installation

## Installation methods

There are two methods to install RNAsamba in your computer:

- Using pip:

```
pip install rnasamba
```

- Using conda:

```
conda install -c bioconda rnasamba
```

!!! info "Using RNAsamba with a GPU"
    If you want to use RNAsamba with a GPU, you first need to install `tensorflow-gpu>=1.5.0,<2.0`, `keras>=2.1.0,<2.3.0`, `numpy<=1.16.5` and `biopython` with pip and then proceed to install `rnasamba` using the `--no-deps` argument.

## Docker

We provide a Docker image for RNAsamba. You can check how to use it in the [usage documentation](usage.md#using-the-docker-image).