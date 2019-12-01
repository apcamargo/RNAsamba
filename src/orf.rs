use pyo3::{prelude::*, wrap_pyfunction};
use rayon::prelude::*;
use regex::Regex;

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
