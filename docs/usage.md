# Using RNAsamba locally

## Download the pre-trained models

We provide two HDF5 files containing the weights of classification models trained with human trancript sequences. The first model (`full_length_weights.hdf5`) was trained exclusively with full-length transcripts and can be used in datasets comprised mostly or exclusively of complete transcript sequences. The second model (`partial_length_weights.hdf5`) was trained with both complete and truncated transcripts and is prefered in cases where there is a significant fraction of partial-length sequences, such as transcriptomes assembled using *de novo* approaches.

<center>
  <button onclick="location.href='https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/full_length_weights.hdf5'" class="pure-material-button-contained">Full-length transcripts</button>
  <button onclick="location.href='https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/partial_length_weights.hdf5'" class="pure-material-button-contained">Partial-length transcripts</button>
</center>

Alternatively, you can download the weight files in an command-line interface by executing the following commands:

```
curl -O https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/full_length_weights.hdf5
curl -O https://raw.githubusercontent.com/apcamargo/RNAsamba/master/data/partial_length_weights.hdf5
```

Both models achieves high classification performance in transcripts from a variety of different species (see our [article](https://www.biorxiv.org/content/10.1101/620880v1)).

!!! warning ""
    In case you want to train your own model, you should follow the steps described in the [`rnasamba train`](#rnasamba-train) section.


## `rnasamba train`

`rnasamba train` is the command for training a new classification model from a training dataset and saving the network weights into an HDF5 file. The user can specify the batch size (`--batch_size`) and the number of training epochs (`--epochs`). The user can also choose to activate early stopping (`--early_stopping`), which reduces training time and can help avoiding overfitting.

```
usage: rnasamba train [-h] [-s EARLY_STOPPING] [-b BATCH_SIZE] [-e EPOCHS]
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

## `rnasamba classify`

`rnasamba classify` is the command for computing the coding potential of transcripts contained in an input FASTA file and classifying them into coding or non-coding. Optionally, the user can specify an output FASTA file (`--protein_fasta`) in which RNAsamba will write the translated sequences of the predicted coding ORFs. If multiple weight files are provided, RNAsamba will ensemble their predictions into a single output.

```
usage: rnasamba classify [-h] [-p PROTEIN_FASTA] [-v {0,1}]
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

- Training a new classification model using *Mus musculus* data downloaded from GENCODE:

```
rnasamba train mouse_model.hdf5 -v 2 gencode.vM21.pc_transcripts.fa gencode.vM21.lncRNA_transcripts.fa

```

- Classifying sequences using our pre-trained model (`full_length_weights.hdf5`) and saving the predicted proteins into a FASTA file:

```
rnasamba classify -p predicted_proteins.fa classification.tsv input.fa full_length_weights.hdf5
```

```
head classification.tsv

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