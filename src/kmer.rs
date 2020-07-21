//   This file is part of the rnasamba package, available at:
//   https://github.com/apcamargo/RNAsamba
//
//   Rnasamba is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with this program. If not, see <https://www.gnu.org/licenses/>.
//
//   Contact: antoniop.camargo@gmail.com

use itertools::iproduct;
use ndarray::Array2;
use numpy::{convert::ToPyArray, PyArray2};
use pyo3::{prelude::*, wrap_pyfunction};
use rayon::prelude::*;
use std::collections::HashMap;

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
        for kmer in kmer_generator(String::from("ATCG"), k).iter() {
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

/// kmer_frequencies_array(sequences)
/// --
///
/// Computes the relative frequencies of 2-, 3- and 4-mers.
///
/// Parameters
/// ----------
/// sequences : list
///    List containing n nucleotide sequences.
///
/// Returns
/// -------
/// ndarray
///    A n * 336 numpy array containing the relative frequencies of each
///    sequences the 2-, 3-, and 4-mer.
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
