// microservices/gospel/src/lib.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a DNA sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    /// Identifier for the sequence
    pub id: String,
    /// The actual nucleotide sequence (A, C, G, T)
    pub sequence: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Sequence {
    /// Create a new DNA sequence
    pub fn new(id: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            sequence: sequence.into(),
            metadata: HashMap::new(),
        }
    }
    
    /// Calculate GC content (percentage of G and C bases)
    pub fn gc_content(&self) -> f64 {
        let gc_count = self.sequence.chars()
            .filter(|c| *c == 'G' || *c == 'C' || *c == 'g' || *c == 'c')
            .count();
            
        if self.sequence.is_empty() {
            return 0.0;
        }
        
        (gc_count as f64) / (self.sequence.len() as f64) * 100.0
    }
    
    /// Count occurrences of each nucleotide
    pub fn nucleotide_counts(&self) -> HashMap<char, usize> {
        let mut counts = HashMap::new();
        
        for c in self.sequence.chars().map(|c| c.to_ascii_uppercase()) {
            *counts.entry(c).or_insert(0) += 1;
        }
        
        counts
    }
    
    /// Find all occurrences of a specific motif
    pub fn find_motif(&self, motif: &str) -> Vec<usize> {
        let seq = self.sequence.as_bytes();
        let motif = motif.as_bytes();
        let mut positions = Vec::new();
        
        // Simple pattern matching (in production, you'd use faster algorithms)
        for i in 0..=seq.len().saturating_sub(motif.len()) {
            if &seq[i..i+motif.len()] == motif {
                positions.push(i);
            }
        }
        
        positions
    }
}

/// Represents a genomic variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Chromosome name
    pub chromosome: String,
    /// Position (1-based)
    pub position: u64,
    /// Reference allele
    pub reference: String,
    /// Alternate allele
    pub alternate: String,
    /// Quality score
    pub quality: f64,
    /// Additional information
    pub info: HashMap<String, String>,
}

impl Variant {
    /// Create a new variant
    pub fn new(
        chromosome: impl Into<String>,
        position: u64,
        reference: impl Into<String>,
        alternate: impl Into<String>,
        quality: f64,
    ) -> Self {
        Self {
            chromosome: chromosome.into(),
            position,
            reference: reference.into(),
            alternate: alternate.into(),
            quality,
            info: HashMap::new(),
        }
    }
    
    /// Check if this is a SNP (Single Nucleotide Polymorphism)
    pub fn is_snp(&self) -> bool {
        self.reference.len() == 1 && self.alternate.len() == 1
    }
    
    /// Check if this is an insertion
    pub fn is_insertion(&self) -> bool {
        self.reference.len() < self.alternate.len()
    }
    
    /// Check if this is a deletion
    pub fn is_deletion(&self) -> bool {
        self.reference.len() > self.alternate.len()
    }
    
    /// Get variant length difference (positive for insertions, negative for deletions)
    pub fn length_diff(&self) -> i64 {
        self.alternate.len() as i64 - self.reference.len() as i64
    }
}

/// Genomics analyzer for sequence and variant analysis
pub struct GenomicsAnalyzer;

impl GenomicsAnalyzer {
    /// Analyze a DNA sequence and extract features
    pub fn analyze_sequence(sequence: &Sequence) -> Result<SequenceAnalysis> {
        let gc = sequence.gc_content();
        let counts = sequence.nucleotide_counts();
        
        // Calculate basic sequence properties
        let length = sequence.sequence.len();
        let has_n = sequence.sequence.contains('N') || sequence.sequence.contains('n');
        
        Ok(SequenceAnalysis {
            sequence_id: sequence.id.clone(),
            length,
            gc_content: gc,
            nucleotide_counts: counts,
            has_ambiguous_bases: has_n,
        })
    }
    
    /// Batch analyze variants
    pub fn analyze_variants(variants: &[Variant]) -> Result<VariantAnalysis> {
        let total = variants.len();
        if total == 0 {
            return Ok(VariantAnalysis::default());
        }
        
        let snps = variants.iter().filter(|v| v.is_snp()).count();
        let insertions = variants.iter().filter(|v| v.is_insertion()).count();
        let deletions = variants.iter().filter(|v| v.is_deletion()).count();
        
        // Calculate statistics
        let snp_rate = snps as f64 / total as f64;
        let indel_rate = (insertions + deletions) as f64 / total as f64;
        
        // Calculate quality distribution
        let mean_quality = variants.iter()
            .map(|v| v.quality)
            .sum::<f64>() / total as f64;
            
        let median_quality = {
            let mut qualities: Vec<f64> = variants.iter().map(|v| v.quality).collect();
            qualities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if total % 2 == 0 {
                (qualities[total/2 - 1] + qualities[total/2]) / 2.0
            } else {
                qualities[total/2]
            }
        };
        
        Ok(VariantAnalysis {
            total_variants: total,
            snp_count: snps,
            insertion_count: insertions,
            deletion_count: deletions,
            snp_rate,
            indel_rate,
            mean_quality,
            median_quality,
        })
    }
}

/// Results of sequence analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct SequenceAnalysis {
    /// ID of the analyzed sequence
    pub sequence_id: String,
    /// Length of the sequence
    pub length: usize,
    /// GC content percentage
    pub gc_content: f64,
    /// Counts of each nucleotide
    pub nucleotide_counts: HashMap<char, usize>,
    /// Whether sequence contains ambiguous bases (N)
    pub has_ambiguous_bases: bool,
}

/// Results of variant analysis
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VariantAnalysis {
    /// Total number of variants
    pub total_variants: usize,
    /// Number of SNPs
    pub snp_count: usize,
    /// Number of insertions
    pub insertion_count: usize,
    /// Number of deletion count
    pub deletion_count: usize,
    /// Ratio of SNPs to total variants
    pub snp_rate: f64,
    /// Ratio of indels to total variants
    pub indel_rate: f64,
    /// Mean quality score
    pub mean_quality: f64,
    /// Median quality score
    pub median_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_content() {
        let seq = Sequence::new("test", "ACGTACGT");
        assert_eq!(seq.gc_content(), 50.0);
        
        let seq2 = Sequence::new("gc_rich", "GCGCGCGC");
        assert_eq!(seq2.gc_content(), 100.0);
    }

    #[test]
    fn test_nucleotide_counts() {
        let seq = Sequence::new("test", "ACGTACGT");
        let counts = seq.nucleotide_counts();
        assert_eq!(counts.get(&'A'), Some(&2));
        assert_eq!(counts.get(&'C'), Some(&2));
        assert_eq!(counts.get(&'G'), Some(&2));
        assert_eq!(counts.get(&'T'), Some(&2));
    }

    #[test]
    fn test_is_snp() {
        let snp = Variant::new("chr1", 100, "A", "G", 30.0);
        assert!(snp.is_snp());
        
        let insertion = Variant::new("chr1", 100, "A", "AGT", 30.0);
        assert!(!insertion.is_snp());
    }
}