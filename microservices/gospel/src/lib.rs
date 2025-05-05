// microservices/gospel/src/lib.rs
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use std::path::Path;
use std::fs;

/// Text document for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document
    pub id: String,
    /// Document title
    pub title: String,
    /// Document content
    pub content: String,
    /// Language of the document
    pub language: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document
    pub fn new(id: impl Into<String>, title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            language: "en".to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// Get the word count of the document
    pub fn word_count(&self) -> usize {
        self.content
            .split_whitespace()
            .count()
    }
    
    /// Get the character count of the document
    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
    
    /// Get the sentence count (approximate) of the document
    pub fn sentence_count(&self) -> usize {
        // Simple sentence splitting on ., !, ?
        self.content
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .count()
    }
    
    /// Extract keywords using a simple TF algorithm
    pub fn extract_keywords(&self, stop_words: &HashSet<String>, max_keywords: usize) -> Vec<(String, usize)> {
        let words: Vec<String> = self.content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && !stop_words.contains(*s))
            .map(String::from)
            .collect();
            
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in words {
            *word_counts.entry(word).or_insert(0) += 1;
        }
        
        let mut keyword_counts: Vec<(String, usize)> = word_counts.into_iter().collect();
        keyword_counts.sort_by(|a, b| b.1.cmp(&a.1));
        
        keyword_counts.truncate(max_keywords);
        keyword_counts
    }
    
    /// Calculate similarity with another document using Jaccard similarity
    pub fn similarity(&self, other: &Document) -> f64 {
        let words_self: HashSet<&str> = self.content
            .to_lowercase()
            .split_whitespace()
            .collect();
            
        let words_other: HashSet<&str> = other.content
            .to_lowercase()
            .split_whitespace()
            .collect();
            
        let intersection = words_self.intersection(&words_other).count();
        let union = words_self.union(&words_other).count();
        
        if union == 0 {
            return 0.0;
        }
        
        intersection as f64 / union as f64
    }
}

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sentiment {
    /// Positive sentiment
    Positive,
    /// Neutral sentiment
    Neutral,
    /// Negative sentiment
    Negative,
}

/// Language detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetection {
    /// Detected language code (ISO 639-1)
    pub language: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Text analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAnalysis {
    /// Document ID
    pub document_id: String,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
    /// Sentence count
    pub sentence_count: usize,
    /// Average words per sentence
    pub avg_words_per_sentence: f64,
    /// Detected language
    pub language: LanguageDetection,
    /// Detected sentiment
    pub sentiment: Sentiment,
    /// Extracted keywords with frequency
    pub keywords: Vec<(String, usize)>,
    /// Readability score (Flesch-Kincaid)
    pub readability_score: f64,
}

/// NLP processor for text analysis and processing
pub struct NlpProcessor {
    /// Stop words for different languages
    stop_words: HashMap<String, HashSet<String>>,
    /// Sentiment lexicon (word -> score)
    sentiment_lexicon: HashMap<String, f64>,
}

impl NlpProcessor {
    /// Create a new NLP processor with default lexicons
    pub fn new() -> Self {
        let mut stop_words = HashMap::new();
        let mut en_stop_words = HashSet::new();
        
        // Just a few common English stop words for demonstration
        for word in &["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"] {
            en_stop_words.insert(word.to_string());
        }
        
        stop_words.insert("en".to_string(), en_stop_words);
        
        // Simple sentiment lexicon
        let mut sentiment_lexicon = HashMap::new();
        
        // Positive words
        for word in &["good", "great", "excellent", "happy", "best", "love", "wonderful"] {
            sentiment_lexicon.insert(word.to_string(), 1.0);
        }
        
        // Negative words
        for word in &["bad", "worst", "terrible", "sad", "hate", "awful", "poor"] {
            sentiment_lexicon.insert(word.to_string(), -1.0);
        }
        
        Self {
            stop_words,
            sentiment_lexicon,
        }
    }
    
    /// Load a custom stop words file
    pub fn load_stop_words<P: AsRef<Path>>(&mut self, language: &str, file_path: P) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let words: HashSet<String> = content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();
            
        self.stop_words.insert(language.to_string(), words);
        Ok(())
    }
    
    /// Load a custom sentiment lexicon
    pub fn load_sentiment_lexicon<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        
        for line in content.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                if let Ok(score) = parts[1].parse::<f64>() {
                    self.sentiment_lexicon.insert(parts[0].to_string(), score);
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect the language of a document (simplified)
    pub fn detect_language(&self, document: &Document) -> LanguageDetection {
        // This is a simplified implementation
        // In production, you would use a more sophisticated algorithm or library
        
        // For demonstration, we'll just assume English
        LanguageDetection {
            language: "en".to_string(),
            confidence: 0.95,
        }
    }
    
    /// Analyze the sentiment of a document
    pub fn analyze_sentiment(&self, document: &Document) -> Sentiment {
        let words: Vec<String> = document.content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
            
        let mut score = 0.0;
        let mut count = 0;
        
        for word in words {
            if let Some(word_score) = self.sentiment_lexicon.get(&word) {
                score += word_score;
                count += 1;
            }
        }
        
        if count == 0 {
            return Sentiment::Neutral;
        }
        
        let avg_score = score / count as f64;
        
        if avg_score > 0.2 {
            Sentiment::Positive
        } else if avg_score < -0.2 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        }
    }
    
    /// Calculate readability score (simplified Flesch-Kincaid)
    pub fn calculate_readability(&self, document: &Document) -> f64 {
        let word_count = document.word_count() as f64;
        let sentence_count = document.sentence_count() as f64;
        
        if sentence_count == 0.0 || word_count == 0.0 {
            return 0.0;
        }
        
        // Simplified Flesch-Kincaid formula
        206.835 - 1.015 * (word_count / sentence_count)
    }
    
    /// Perform full text analysis on a document
    pub fn analyze(&self, document: &Document) -> Result<TextAnalysis> {
        let word_count = document.word_count();
        let char_count = document.char_count();
        let sentence_count = document.sentence_count();
        
        let avg_words_per_sentence = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };
        
        let language = self.detect_language(document);
        let sentiment = self.analyze_sentiment(document);
        
        let stop_words = self.stop_words.get(&language.language)
            .ok_or_else(|| anyhow!("Stop words not available for language: {}", language.language))?;
            
        let keywords = document.extract_keywords(stop_words, 10);
        let readability_score = self.calculate_readability(document);
        
        Ok(TextAnalysis {
            document_id: document.id.clone(),
            word_count,
            char_count,
            sentence_count,
            avg_words_per_sentence,
            language,
            sentiment,
            keywords,
            readability_score,
        })
    }
    
    /// Batch analyze multiple documents in parallel
    pub fn batch_analyze(&self, documents: &[Document]) -> Result<Vec<TextAnalysis>> {
        documents.par_iter()
            .map(|doc| self.analyze(doc))
            .collect()
    }
}

/// NLP service to manage documents and analysis
pub struct NlpService {
    /// NLP processor for analysis
    processor: NlpProcessor,
    /// Document storage
    documents: RwLock<HashMap<String, Arc<Document>>>,
    /// Analysis cache
    analysis_cache: RwLock<HashMap<String, Arc<TextAnalysis>>>,
}

impl NlpService {
    /// Create a new NLP service
    pub fn new() -> Self {
        Self {
            processor: NlpProcessor::new(),
            documents: RwLock::new(HashMap::new()),
            analysis_cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// Add or update a document
    pub async fn add_document(&self, document: Document) -> Result<()> {
        let id = document.id.clone();
        
        // Update document storage
        {
            let mut documents = self.documents.write().await;
            documents.insert(id.clone(), Arc::new(document));
        }
        
        // Invalidate cached analysis
        {
            let mut cache = self.analysis_cache.write().await;
            cache.remove(&id);
        }
        
        Ok(())
    }
    
    /// Get a document by ID
    pub async fn get_document(&self, id: &str) -> Result<Arc<Document>> {
        let documents = self.documents.read().await;
        documents.get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Document not found: {}", id))
    }
    
    /// Analyze a document by ID
    pub async fn analyze_document(&self, id: &str) -> Result<Arc<TextAnalysis>> {
        // Check cache first
        {
            let cache = self.analysis_cache.read().await;
            if let Some(analysis) = cache.get(id) {
                return Ok(analysis.clone());
            }
        }
        
        // Get the document
        let document = self.get_document(id).await?;
        
        // Perform analysis
        let analysis = self.processor.analyze(&document)?;
        let analysis_arc = Arc::new(analysis);
        
        // Update cache
        {
            let mut cache = self.analysis_cache.write().await;
            cache.insert(id.to_string(), analysis_arc.clone());
        }
        
        Ok(analysis_arc)
    }
    
    /// Find similar documents for a given document
    pub async fn find_similar(&self, id: &str, min_similarity: f64) -> Result<Vec<(Arc<Document>, f64)>> {
        let target_doc = self.get_document(id).await?;
        let documents = self.documents.read().await;
        
        let mut similar: Vec<(Arc<Document>, f64)> = Vec::new();
        
        for (doc_id, doc) in documents.iter() {
            if doc_id == id {
                continue;
            }
            
            let similarity = target_doc.similarity(doc);
            if similarity >= min_similarity {
                similar.push((doc.clone(), similarity));
            }
        }
        
        // Sort by similarity (descending)
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(similar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_stats() {
        let doc = Document::new(
            "test", 
            "Test Document", 
            "This is a test document. It has three sentences! How many words does it contain?"
        );
        
        assert_eq!(doc.word_count(), 15);
        assert_eq!(doc.sentence_count(), 3);
    }

    #[test]
    fn test_similarity() {
        let doc1 = Document::new("doc1", "Similar", "The quick brown fox jumps over the lazy dog");
        let doc2 = Document::new("doc2", "Similar too", "The fox jumps over the lazy brown dog");
        let doc3 = Document::new("doc3", "Different", "Lorem ipsum dolor sit amet");
        
        assert!(doc1.similarity(&doc2) > 0.7); // Similar documents
        assert!(doc1.similarity(&doc3) < 0.1); // Different documents
    }

    #[tokio::test]
    async fn test_nlp_service() {
        let service = NlpService::new();
        
        let doc = Document::new(
            "test1", 
            "Positive Document", 
            "This is a great document with happy content. It's excellent!"
        );
        
        service.add_document(doc).await.unwrap();
        let analysis = service.analyze_document("test1").await.unwrap();
        
        assert_eq!(analysis.sentiment, Sentiment::Positive);
        assert!(analysis.readability_score > 0.0);
    }
}