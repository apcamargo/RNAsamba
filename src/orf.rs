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

/// longest_orf_array(sequences)
/// --
///
/// Finds the ORF within transcript sequences.
///
/// Parameters
/// ----------
/// sequences : list
///    List containing n nucleotide sequences.
///
/// Returns
/// -------
/// list
///    A list of n tuples, each containing the length of the translated protein,
///    the position of the ORF start and the sequence of the putative protein.
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
