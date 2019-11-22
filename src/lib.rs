use itertools::iproduct;
use ndarray::Array2;
use numpy::{convert::ToPyArray, PyArray2};
use pyo3::{prelude::*, wrap_pyfunction};
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;

static AA_TABLE_CANONICAL: [[[char; 4]; 4]; 4] = [
    [
        ['K', 'N', 'K', 'N'],
        ['T', 'T', 'T', 'T'],
        ['R', 'S', 'R', 'S'],
        ['I', 'I', 'M', 'I'],
    ],
    [
        ['Q', 'H', 'Q', 'H'],
        ['P', 'P', 'P', 'P'],
        ['R', 'R', 'R', 'R'],
        ['L', 'L', 'L', 'L'],
    ],
    [
        ['E', 'D', 'E', 'D'],
        ['A', 'A', 'A', 'A'],
        ['G', 'G', 'G', 'G'],
        ['V', 'V', 'V', 'V'],
    ],
    [
        ['*', 'Y', '*', 'Y'],
        ['S', 'S', 'S', 'S'],
        ['*', 'C', 'W', 'C'],
        ['L', 'F', 'L', 'F'],
    ],
];

static ASCII_TO_INDEX: [usize; 128] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

// Function adapted from the protein-translate crate
fn translate(seq: &[u8]) -> String {
    let mut peptide = String::new();
    for triplet in seq.chunks_exact(3) {
        for c in triplet {
            if !c.is_ascii() {
                peptide.push('X');
                continue;
            }
        }
        let c1 = ASCII_TO_INDEX[triplet[0] as usize];
        let c2 = ASCII_TO_INDEX[triplet[1] as usize];
        let c3 = ASCII_TO_INDEX[triplet[2] as usize];
        let amino_acid = if c1 == 4 || c2 == 4 || c3 == 4 {
            'X'
        } else {
            AA_TABLE_CANONICAL[c1][c2][c3]
        };
        if amino_acid == '*' {
            break;
        } else {
            peptide.push(amino_acid);
        }
    }
    peptide
}

fn sequence_longest_orf(sequence: &str) -> (usize, usize, String) {
    let re = Regex::new("ATG").unwrap();
    let mut longest = (0, 0, "".to_string());
    for m in re.find_iter(sequence) {
        let putative_sequence = &sequence[m.start()..];
        let putative_protein = translate(putative_sequence.as_bytes());
        if putative_protein.len() > longest.0 {
            longest = (putative_protein.len(), m.start(), putative_protein);
        }
    }
    longest
}

#[pyfunction]
fn longest_orf_array(sequences: Vec<&str>) -> PyResult<Vec<(usize, usize, String)>> {
    Ok(sequences
        .par_iter()
        .map(|sequence| sequence_longest_orf(sequence))
        .collect())
}

#[pymodule]
fn orf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(longest_orf_array))?;
    Ok(())
}

fn sequence_kmer_counts(sequence: &str, k: usize) -> HashMap<&str, u16> {
    let mut counts = HashMap::new();
    let n_kmers = sequence.len() - k + 1;
    for i in 0..n_kmers {
        let kmer = &sequence[i..i + k];
        *counts.entry(kmer).or_insert(0) += 1;
    }
    counts
}

fn kmer_generator(alphabet: String, k: usize) -> Vec<String> {
    match k {
        0 => vec![],
        1 => alphabet.chars().map(|c| c.to_string()).collect(),
        2 => iproduct!(alphabet.chars(), alphabet.chars())
            .map(|(a, b)| format!("{}{}", a, b))
            .collect(),
        _ => iproduct!(kmer_generator(alphabet.clone(), k - 1), alphabet.chars())
            .map(|(a, b)| format!("{}{}", a, b))
            .collect(),
    }
}

fn sequence_kmer_frequencies(sequence: &str) -> Vec<f32> {
    let mut kmer_frequency_array: Vec<f32> = Vec::new();
    for k in 2..5 {
        let sequence_total_kmers = sequence.len() - k + 1;
        let sequence_kmer_count = sequence_kmer_counts(sequence, k);
        for kmer in kmer_generator(String::from("ATCG"), k).into_iter() {
            let kmer_count: u16;
            match sequence_kmer_count.get(&kmer[..]) {
                Some(n) => kmer_count = *n,
                _ => kmer_count = 0,
            };
            let kmer_frequency = kmer_count as f32 / sequence_total_kmers as f32;
            kmer_frequency_array.push(kmer_frequency);
        }
    }
    kmer_frequency_array
}

#[pyfunction]
fn kmer_frequencies_array(sequences: Vec<&str>) -> Py<PyArray2<f32>> {
    Array2::from_shape_vec(
        (sequences.len(), 336),
        sequences
            .par_iter()
            .map(|sequence| sequence_kmer_frequencies(sequence))
            .flatten()
            .collect(),
    )
    .unwrap()
    .to_pyarray(Python::acquire_gil().python())
    .to_owned()
}

#[pymodule]
fn kmer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(kmer_frequencies_array))?;
    Ok(())
}
