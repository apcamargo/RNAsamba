use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::str;

#[pyfunction]
fn count_kmers(sequence: &str, k: usize) -> PyResult<HashMap<&str, u16>> {
    let mut counts = HashMap::new();
    let n_kmers = sequence.len() - k + 1;
    for i in 0..n_kmers {
        let kmer = &sequence[i..i + k];
        *counts.entry(kmer).or_insert(0) += 1;
    }
    Ok(counts)
}

#[pymodule]
fn kmer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(count_kmers))?;
    Ok(())
}
