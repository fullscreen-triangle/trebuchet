use anyhow::{anyhow, Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use csv::ReaderBuilder;
use dashmap::DashMap;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use indexmap::IndexMap;
use ndarray::{Array2, Axis};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Record identifier type
pub type RecordId = Uuid;

/// Data source identifier type
pub type SourceId = String;

/// Field identifier type
pub type FieldId = String;

/// Supported data types for fields
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Decimal number type
    Decimal,
    /// Boolean type
    Boolean,
    /// Date/time type
    DateTime,
    /// JSON type
    Json,
}

/// Data field definition
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Field {
    /// Field ID
    pub id: FieldId,
    /// Field name
    pub name: String,
    /// Field description
    pub description: Option<String>,
    /// Field data type
    pub field_type: FieldType,
    /// Whether the field is required
    pub required: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Validation rules
    pub validation: Option<HashMap<String, serde_json::Value>>,
}

/// Data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    /// CSV file
    Csv,
    /// JSON file
    Json,
    /// Database
    Database,
    /// API
    Api,
    /// Memory
    Memory,
}

/// Data source definition
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataSource {
    /// Source ID
    pub id: SourceId,
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: SourceType,
    /// Connection details
    pub connection: HashMap<String, String>,
    /// Field mappings (source field -> canonical field)
    pub field_mappings: HashMap<String, FieldId>,
    /// Schema fields
    pub fields: Vec<Field>,
}

/// Integration mode for combining data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IntegrationMode {
    /// Union of all records
    Union,
    /// Intersection of records (only matching IDs)
    Intersection,
    /// Left join (keep all records from primary source)
    LeftJoin,
    /// Right join (keep all records from secondary source)
    RightJoin,
    /// Full outer join (union with matching)
    FullOuterJoin,
}

/// Key matching strategy for record linkage
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyMatchingStrategy {
    /// Exact matching
    Exact,
    /// Fuzzy matching with threshold
    Fuzzy { threshold: f64 },
    /// Custom matching function
    Custom { name: String },
}

/// Integration configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IntegrationConfig {
    /// Primary data source ID
    pub primary_source: SourceId,
    /// Secondary data sources
    pub secondary_sources: Vec<SourceId>,
    /// Integration mode
    pub mode: IntegrationMode,
    /// Key fields for matching records across sources
    pub key_fields: Vec<FieldId>,
    /// Key matching strategy
    pub matching_strategy: KeyMatchingStrategy,
    /// Conflict resolution strategy for overlapping fields
    pub conflict_strategy: ConflictStrategy,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Value representation for fields
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Decimal value
    Decimal(Decimal),
    /// Boolean value
    Boolean(bool),
    /// Date/time value
    DateTime(DateTime<Utc>),
    /// JSON value
    Json(serde_json::Value),
    /// Null value
    Null,
}

impl FieldValue {
    /// Get the type of this field value
    pub fn get_type(&self) -> FieldType {
        match self {
            FieldValue::String(_) => FieldType::String,
            FieldValue::Integer(_) => FieldType::Integer,
            FieldValue::Decimal(_) => FieldType::Decimal,
            FieldValue::Boolean(_) => FieldType::Boolean,
            FieldValue::DateTime(_) => FieldType::DateTime,
            FieldValue::Json(_) => FieldType::Json,
            FieldValue::Null => FieldType::String, // Default to string for null
        }
    }
    
    /// Parse a string into a field value based on the field type
    pub fn parse(value: &str, field_type: FieldType) -> Result<Self> {
        if value.is_empty() {
            return Ok(FieldValue::Null);
        }
        
        match field_type {
            FieldType::String => Ok(FieldValue::String(value.to_string())),
            FieldType::Integer => {
                let parsed = value.parse::<i64>()
                    .context("Failed to parse integer value")?;
                Ok(FieldValue::Integer(parsed))
            }
            FieldType::Decimal => {
                let parsed = value.parse::<Decimal>()
                    .context("Failed to parse decimal value")?;
                Ok(FieldValue::Decimal(parsed))
            }
            FieldType::Boolean => {
                let parsed = match value.to_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    _ => return Err(anyhow!("Failed to parse boolean value: {}", value)),
                };
                Ok(FieldValue::Boolean(parsed))
            }
            FieldType::DateTime => {
                // Try parsing with different formats
                if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
                    Ok(FieldValue::DateTime(dt.with_timezone(&Utc)))
                } else if let Ok(dt) = DateTime::parse_from_rfc2822(value) {
                    Ok(FieldValue::DateTime(dt.with_timezone(&Utc)))
                } else {
                    Err(anyhow!("Failed to parse date/time value: {}", value))
                }
            }
            FieldType::Json => {
                let parsed = serde_json::from_str(value)
                    .context("Failed to parse JSON value")?;
                Ok(FieldValue::Json(parsed))
            }
        }
    }
    
    /// Convert the field value to a string
    pub fn to_string_value(&self) -> String {
        match self {
            FieldValue::String(s) => s.clone(),
            FieldValue::Integer(i) => i.to_string(),
            FieldValue::Decimal(d) => d.to_string(),
            FieldValue::Boolean(b) => b.to_string(),
            FieldValue::DateTime(dt) => dt.to_rfc3339(),
            FieldValue::Json(j) => j.to_string(),
            FieldValue::Null => "".to_string(),
        }
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ConflictStrategy {
    /// Prefer primary source
    PreferPrimary,
    /// Prefer secondary source
    PreferSecondary,
    /// Most recent value based on timestamp field
    MostRecent { timestamp_field: FieldId },
    /// Combine values using a custom function
    Custom { function: String },
}

/// Data record
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Record {
    /// Record ID
    pub id: RecordId,
    /// Source ID
    pub source_id: SourceId,
    /// Record data
    pub data: HashMap<FieldId, FieldValue>,
    /// Record metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when the record was created
    pub created_at: DateTime<Utc>,
    /// Timestamp when the record was last updated
    pub updated_at: DateTime<Utc>,
}

impl Record {
    /// Create a new record
    pub fn new(source_id: SourceId, data: HashMap<FieldId, FieldValue>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            source_id,
            data,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Get a field value
    pub fn get(&self, field_id: &FieldId) -> Option<&FieldValue> {
        self.data.get(field_id)
    }
    
    /// Set a field value
    pub fn set(&mut self, field_id: FieldId, value: FieldValue) {
        self.data.insert(field_id, value);
        self.updated_at = Utc::now();
    }
}

/// Data reader interface
#[async_trait]
pub trait DataReader: Send + Sync {
    /// Read data as a stream of records
    async fn read_records(&self, source: &DataSource) -> Result<Pin<Box<dyn Stream<Item = Result<Record>> + Send>>>;
    
    /// Read all records into a vector
    async fn read_all(&self, source: &DataSource) -> Result<Vec<Record>> {
        let stream = self.read_records(source).await?;
        stream.try_collect().await
    }
    
    /// Count the number of records
    async fn count(&self, source: &DataSource) -> Result<usize> {
        let records = self.read_all(source).await?;
        Ok(records.len())
    }
}

/// Data writer interface
#[async_trait]
pub trait DataWriter: Send + Sync {
    /// Write records to the data source
    async fn write_records<S>(&self, source: &DataSource, records: S) -> Result<usize>
    where
        S: Stream<Item = Result<Record>> + Send + 'static;
        
    /// Write a single record
    async fn write_record(&self, source: &DataSource, record: Record) -> Result<()> {
        let stream = stream::once(async { Ok(record) });
        let count = self.write_records(source, stream).await?;
        
        if count != 1 {
            return Err(anyhow!("Failed to write record"));
        }
        
        Ok(())
    }
}

/// CSV data reader implementation
pub struct CsvReader;

impl CsvReader {
    /// Create a new CSV reader
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DataReader for CsvReader {
    async fn read_records(&self, source: &DataSource) -> Result<Pin<Box<dyn Stream<Item = Result<Record>> + Send>>> {
        // Check source type
        if source.source_type != SourceType::Csv {
            return Err(anyhow!("Invalid source type for CSV reader: {:?}", source.source_type));
        }
        
        // Get file path from connection
        let file_path = source.connection.get("path")
            .ok_or_else(|| anyhow!("Missing 'path' in CSV source connection"))?;
            
        // Read file
        let content = fs::read_to_string(file_path).await
            .context(format!("Failed to read CSV file: {}", file_path))?;
            
        // Parse CSV
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(content.as_bytes());
            
        // Get headers
        let headers = reader.headers()
            .context("Failed to read CSV headers")?
            .iter()
            .map(String::from)
            .collect::<Vec<_>>();
            
        // Convert field mappings for faster lookup
        let field_mappings: HashMap<String, FieldId> = source.field_mappings.clone();
        
        // Create records
        let mut records = Vec::new();
        
        for result in reader.records() {
            let record = result.context("Failed to read CSV record")?;
            
            let mut data = HashMap::new();
            
            for (i, header) in headers.iter().enumerate() {
                // Skip if header is not in field mappings
                let field_id = match field_mappings.get(header) {
                    Some(id) => id.clone(),
                    None => continue,
                };
                
                // Get field definition
                let field = source.fields.iter()
                    .find(|f| f.id == field_id)
                    .ok_or_else(|| anyhow!("Field not found in schema: {}", field_id))?;
                    
                // Get value
                let value = if i < record.len() {
                    let raw_value = record.get(i).unwrap_or("");
                    
                    if raw_value.is_empty() && !field.required {
                        FieldValue::Null
                    } else if raw_value.is_empty() && field.required && field.default_value.is_some() {
                        // Use default value
                        FieldValue::parse(field.default_value.as_ref().unwrap(), field.field_type)?
                    } else {
                        FieldValue::parse(raw_value, field.field_type)?
                    }
                } else if field.required && field.default_value.is_some() {
                    // Use default value
                    FieldValue::parse(field.default_value.as_ref().unwrap(), field.field_type)?
                } else {
                    FieldValue::Null
                };
                
                data.insert(field_id, value);
            }
            
            // Create record
            let record = Record::new(source.id.clone(), data);
            records.push(Ok(record));
        }
        
        // Create stream from records
        let stream = stream::iter(records);
        
        Ok(Box::pin(stream))
    }
}

/// Model input data to be sent to an AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInput {
    /// Unique ID for this input
    pub id: String,
    /// Input text content for text-based models
    pub text: Option<String>,
    /// Input binary data for image, audio, etc. models
    pub binary_data: Option<Vec<u8>>,
    /// Additional parameters to control model behavior
    pub parameters: HashMap<String, serde_json::Value>,
}

impl ModelInput {
    /// Create a new text input with random ID
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            text: Some(content.into()),
            binary_data: None,
            parameters: HashMap::new(),
        }
    }
    
    /// Create a new binary input with random ID
    pub fn binary(data: Vec<u8>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            text: None,
            binary_data: Some(data),
            parameters: HashMap::new(),
        }
    }
    
    /// Add a parameter
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

/// Model output data returned from an AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// ID matching the original input
    pub id: String,
    /// Output text content for text-based models
    pub text: Option<String>,
    /// Output binary data for image, audio, etc. models
    pub binary_data: Option<Vec<u8>>,
    /// Model metadata about the generation
    pub metadata: HashMap<String, serde_json::Value>,
    /// Whether the model generation was successful
    pub success: bool,
    /// Error message if not successful
    pub error: Option<String>,
}

impl ModelOutput {
    /// Create a successful text output
    pub fn text_success(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: Some(content.into()),
            binary_data: None,
            metadata: HashMap::new(),
            success: true,
            error: None,
        }
    }
    
    /// Create a successful binary output
    pub fn binary_success(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            text: None,
            binary_data: Some(data),
            metadata: HashMap::new(),
            success: true,
            error: None,
        }
    }
    
    /// Create an error output
    pub fn error(id: impl Into<String>, error_msg: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: None,
            binary_data: None,
            metadata: HashMap::new(),
            success: false,
            error: Some(error_msg.into()),
        }
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Model capability flags to describe what a model can do
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    /// Generate text
    TextGeneration,
    /// Answer questions with factual information
    QuestionAnswering,
    /// Summarize text
    Summarization,
    /// Classify text into categories
    Classification,
    /// Extract information from text
    InformationExtraction,
    /// Translate between languages
    Translation,
    /// Generate embeddings/vectors from text
    Embeddings,
    /// Generate or edit images
    ImageGeneration,
    /// Understand image content
    ImageUnderstanding,
    /// Process or generate audio
    Audio,
    /// Generate code
    CodeGeneration,
    /// Code completion
    CodeCompletion,
    /// General reasoning and problem-solving
    Reasoning,
}

/// Model definition with capabilities, costs, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Unique model identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model provider (e.g., "openai", "anthropic", "huggingface")
    pub provider: String,
    /// Capabilities this model supports
    pub capabilities: Vec<ModelCapability>,
    /// Whether this model is currently available
    pub available: bool,
    /// Relative cost to run (1.0 = standard cost)
    pub cost_factor: f64,
    /// Maximum tokens/units this model can process in one call
    pub max_input_size: usize,
    /// Maximum tokens/units this model can generate in one call
    pub max_output_size: usize,
    /// Additional model-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Model backend interface for processing inputs
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Process an input and produce an output
    async fn process(&self, input: ModelInput) -> Result<ModelOutput>;
    
    /// Get the model definition
    fn get_definition(&self) -> &ModelDefinition;
    
    /// Check if this backend can handle the given input
    fn can_handle(&self, input: &ModelInput) -> bool;
}

/// OpenAI-compatible model backend
pub struct OpenAIBackend {
    /// Model definition
    definition: ModelDefinition,
    /// API key
    api_key: String,
    /// Base URL (allows using OpenAI-compatible APIs)
    base_url: String,
}

impl OpenAIBackend {
    /// Create a new OpenAI backend
    pub fn new(definition: ModelDefinition, api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            definition,
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for OpenAIBackend {
    async fn process(&self, input: ModelInput) -> Result<ModelOutput> {
        // In a real implementation, this would call the OpenAI API
        // For this example, we'll just simulate a response
        
        if let Some(text) = &input.text {
            // Simulate delay for processing
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            let response = format!("OpenAI response to: {}", text);
            Ok(ModelOutput::text_success(input.id, response)
                .with_metadata("model", self.definition.id.clone())
                .with_metadata("tokens_used", 15))
        } else {
            Err(anyhow!("OpenAI backend only supports text input"))
        }
    }
    
    fn get_definition(&self) -> &ModelDefinition {
        &self.definition
    }
    
    fn can_handle(&self, input: &ModelInput) -> bool {
        // Check if this model can handle the input based on its capabilities
        input.text.is_some() && self.definition.capabilities.contains(&ModelCapability::TextGeneration)
    }
}

/// Local model backend for models running on the same machine
pub struct LocalModelBackend {
    /// Model definition
    definition: ModelDefinition,
    /// Model path
    model_path: String,
}

impl LocalModelBackend {
    /// Create a new local model backend
    pub fn new(definition: ModelDefinition, model_path: impl Into<String>) -> Self {
        Self {
            definition,
            model_path: model_path.into(),
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for LocalModelBackend {
    async fn process(&self, input: ModelInput) -> Result<ModelOutput> {
        // In a real implementation, this would load and run a local model
        // For this example, we'll just simulate a response
        
        if let Some(text) = &input.text {
            // Simulate delay for processing
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            
            let response = format!("Local model response using model at {}: {}", self.model_path, text);
            Ok(ModelOutput::text_success(input.id, response)
                .with_metadata("model", self.definition.id.clone())
                .with_metadata("processing_time_ms", 200))
        } else if let Some(data) = &input.binary_data {
            // Simulate processing binary data
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
            
            Ok(ModelOutput::text_success(input.id, format!("Processed {} bytes of binary data", data.len()))
                .with_metadata("model", self.definition.id.clone())
                .with_metadata("processing_time_ms", 300))
        } else {
            Err(anyhow!("Local model backend received empty input"))
        }
    }
    
    fn get_definition(&self) -> &ModelDefinition {
        &self.definition
    }
    
    fn can_handle(&self, input: &ModelInput) -> bool {
        // Check if this model can handle the input based on its capabilities and the input type
        let has_valid_input = input.text.is_some() || input.binary_data.is_some();
        has_valid_input && self.definition.available
    }
}

/// Routing strategy for selecting models
pub enum RoutingStrategy {
    /// Use a specific model
    Specific(String),
    /// Choose the cheapest model that satisfies requirements
    LowestCost,
    /// Choose the fastest model based on historical performance
    Fastest,
    /// Balance between cost and performance
    Balanced,
}

/// ModelRouter service for routing requests to models
pub struct ModelRouter {
    /// Registered model backends
    backends: RwLock<HashMap<String, Arc<dyn ModelBackend>>>,
    /// Performance metrics for models
    metrics: RwLock<HashMap<String, ModelMetrics>>,
}

/// Performance metrics for a model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Number of requests processed
    pub request_count: usize,
    /// Number of successful requests
    pub success_count: usize,
    /// Number of failed requests
    pub failure_count: usize,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

impl ModelRouter {
    /// Create a new model router
    pub fn new() -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
            metrics: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a model backend
    pub async fn register_model(&self, backend: impl ModelBackend + 'static) -> Result<()> {
        let backend_arc = Arc::new(backend);
        let model_id = backend_arc.get_definition().id.clone();
        
        let mut backends = self.backends.write().await;
        backends.insert(model_id.clone(), backend_arc);
        
        // Initialize metrics
        let mut metrics = self.metrics.write().await;
        metrics.entry(model_id).or_insert_with(ModelMetrics::default);
        
        Ok(())
    }
    
    /// Unregister a model backend
    pub async fn unregister_model(&self, model_id: &str) -> Result<()> {
        let mut backends = self.backends.write().await;
        backends.remove(model_id).ok_or_else(|| anyhow!("Model '{}' not found", model_id))?;
        Ok(())
    }
    
    /// Get available models with specified capabilities
    pub async fn get_available_models(&self, capabilities: &[ModelCapability]) -> Vec<ModelDefinition> {
        let backends = self.backends.read().await;
        
        backends.values()
            .filter(|backend| {
                let def = backend.get_definition();
                def.available && capabilities.iter().all(|cap| def.capabilities.contains(cap))
            })
            .map(|backend| backend.get_definition().clone())
            .collect()
    }
    
    /// Route a request to the appropriate model using the specified strategy
    pub async fn route_request(
        &self, 
        input: ModelInput, 
        strategy: RoutingStrategy,
        required_capabilities: &[ModelCapability],
    ) -> Result<ModelOutput> {
        let backends = self.backends.read().await;
        
        // Find candidates that can handle this input
        let candidates: Vec<Arc<dyn ModelBackend>> = backends.values()
            .filter(|backend| {
                let def = backend.get_definition();
                def.available && 
                backend.can_handle(&input) && 
                required_capabilities.iter().all(|cap| def.capabilities.contains(cap))
            })
            .cloned()
            .collect();
            
        if candidates.is_empty() {
            return Err(anyhow!("No suitable model found for the request"));
        }
        
        // Select a model based on strategy
        let selected = match strategy {
            RoutingStrategy::Specific(model_id) => {
                candidates.iter()
                    .find(|backend| backend.get_definition().id == model_id)
                    .ok_or_else(|| anyhow!("Specified model '{}' not available", model_id))?
                    .clone()
            },
            RoutingStrategy::LowestCost => {
                candidates.iter()
                    .min_by(|a, b| a.get_definition().cost_factor.partial_cmp(&b.get_definition().cost_factor).unwrap())
                    .unwrap()
                    .clone()
            },
            RoutingStrategy::Fastest => {
                let metrics = self.metrics.read().await;
                candidates.iter()
                    .min_by(|a, b| {
                        let a_time = metrics.get(&a.get_definition().id)
                            .map(|m| m.avg_processing_time_ms)
                            .unwrap_or(f64::MAX);
                        let b_time = metrics.get(&b.get_definition().id)
                            .map(|m| m.avg_processing_time_ms)
                            .unwrap_or(f64::MAX);
                        a_time.partial_cmp(&b_time).unwrap()
                    })
                    .unwrap()
                    .clone()
            },
            RoutingStrategy::Balanced => {
                let metrics = self.metrics.read().await;
                candidates.iter()
                    .min_by(|a, b| {
                        let a_def = a.get_definition();
                        let b_def = b.get_definition();
                        
                        let a_metric = metrics.get(&a_def.id).unwrap_or(&ModelMetrics::default());
                        let b_metric = metrics.get(&b_def.id).unwrap_or(&ModelMetrics::default());
                        
                        // Balance score = processing_time * cost_factor * (1 / success_rate)
                        let a_score = a_metric.avg_processing_time_ms * a_def.cost_factor / (a_metric.success_rate.max(0.1));
                        let b_score = b_metric.avg_processing_time_ms * b_def.cost_factor / (b_metric.success_rate.max(0.1));
                        
                        a_score.partial_cmp(&b_score).unwrap()
                    })
                    .unwrap()
                    .clone()
            }
        };
        
        // Process the request with the selected model
        let start_time = std::time::Instant::now();
        let result = selected.process(input.clone()).await;
        let elapsed = start_time.elapsed().as_millis() as f64;
        
        // Update metrics
        self.update_metrics(&selected.get_definition().id, result.is_ok(), elapsed).await;
        
        // Return result or error
        result.map_err(|e| anyhow!("Model processing error: {}", e))
    }
    
    /// Update metrics for a model
    async fn update_metrics(&self, model_id: &str, success: bool, processing_time_ms: f64) {
        let mut metrics = self.metrics.write().await;
        
        let model_metrics = metrics.entry(model_id.to_string()).or_insert_with(ModelMetrics::default);
        
        // Update counts
        model_metrics.request_count += 1;
        if success {
            model_metrics.success_count += 1;
        } else {
            model_metrics.failure_count += 1;
        }
        
        // Update averages
        let old_total_time = model_metrics.avg_processing_time_ms * (model_metrics.request_count - 1) as f64;
        let new_total_time = old_total_time + processing_time_ms;
        model_metrics.avg_processing_time_ms = new_total_time / model_metrics.request_count as f64;
        
        // Update success rate
        model_metrics.success_rate = model_metrics.success_count as f64 / model_metrics.request_count as f64;
    }
    
    /// Get metrics for a specific model
    pub async fn get_model_metrics(&self, model_id: &str) -> Option<ModelMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(model_id).cloned()
    }
}

/// Concurrent model processing with parallel requests
pub struct ParallelProcessor {
    /// Router for routing requests
    router: Arc<ModelRouter>,
}

impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new(router: Arc<ModelRouter>) -> Self {
        Self {
            router,
        }
    }
    
    /// Process multiple inputs in parallel, returning results as they complete
    pub async fn process_parallel(
        &self,
        inputs: Vec<ModelInput>,
        strategy: RoutingStrategy,
        required_capabilities: &[ModelCapability],
    ) -> Result<Vec<ModelOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        
        let (tx, mut rx) = mpsc::channel(inputs.len());
        let mut handles = Vec::with_capacity(inputs.len());
        
        // Process each input in parallel
        for input in inputs {
            let tx = tx.clone();
            let router = self.router.clone();
            let strategy = strategy.clone();
            let capabilities = required_capabilities.to_vec();
            
            let handle = tokio::spawn(async move {
                let result = router.route_request(input, strategy, &capabilities).await;
                let _ = tx.send(result).await;
            });
            
            handles.push(handle);
        }
        
        // Drop original sender to allow channel to close when all tasks complete
        drop(tx);
        
        // Collect results as they arrive
        let mut outputs = Vec::with_capacity(handles.len());
        while let Some(result) = rx.recv().await {
            outputs.push(result?);
        }
        
        // Ensure all tasks complete
        for handle in handles {
            handle.await?;
        }
        
        Ok(outputs)
    }
}

/// Streaming output for models that support it
pub struct StreamingOutput {
    /// Receiver for streaming tokens
    pub receiver: mpsc::Receiver<Result<String>>,
}

/// Model backend supporting streaming output
#[async_trait::async_trait]
pub trait StreamingModelBackend: ModelBackend {
    /// Process with streaming output
    async fn process_streaming(&self, input: ModelInput) -> Result<StreamingOutput>;
}

/// Example streaming OpenAI backend
pub struct StreamingOpenAIBackend {
    /// Base backend
    backend: OpenAIBackend,
}

impl StreamingOpenAIBackend {
    /// Create a new streaming OpenAI backend
    pub fn new(definition: ModelDefinition, api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            backend: OpenAIBackend::new(definition, api_key, base_url),
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for StreamingOpenAIBackend {
    async fn process(&self, input: ModelInput) -> Result<ModelOutput> {
        self.backend.process(input).await
    }
    
    fn get_definition(&self) -> &ModelDefinition {
        self.backend.get_definition()
    }
    
    fn can_handle(&self, input: &ModelInput) -> bool {
        self.backend.can_handle(input)
    }
}

#[async_trait::async_trait]
impl StreamingModelBackend for StreamingOpenAIBackend {
    async fn process_streaming(&self, input: ModelInput) -> Result<StreamingOutput> {
        if let Some(text) = &input.text {
            let (tx, rx) = mpsc::channel(100);
            
            // Split input into "tokens" (just words for this example)
            let words = text.split_whitespace().collect::<Vec<_>>();
            
            // Spawn a task to simulate streaming responses
            tokio::spawn(async move {
                for word in words {
                    // Simulate thinking time
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    
                    // Send the next token
                    let response = format!("Streaming: {}", word);
                    if tx.send(Ok(response)).await.is_err() {
                        break;
                    }
                }
            });
            
            Ok(StreamingOutput { receiver: rx })
        } else {
            Err(anyhow!("Streaming backend only supports text input"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_routing() {
        // Create a router
        let router = ModelRouter::new();
        
        // Create model definitions
        let gpt4_def = ModelDefinition {
            id: "gpt-4".to_string(),
            name: "GPT-4".to_string(),
            version: "1.0".to_string(),
            provider: "openai".to_string(),
            capabilities: vec![
                ModelCapability::TextGeneration, 
                ModelCapability::QuestionAnswering,
                ModelCapability::Reasoning,
            ],
            available: true,
            cost_factor: 5.0,
            max_input_size: 8000,
            max_output_size: 4000,
            config: HashMap::new(),
        };
        
        let gpt35_def = ModelDefinition {
            id: "gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            version: "1.0".to_string(),
            provider: "openai".to_string(),
            capabilities: vec![
                ModelCapability::TextGeneration, 
                ModelCapability::QuestionAnswering,
            ],
            available: true,
            cost_factor: 1.0,
            max_input_size: 4000,
            max_output_size: 2000,
            config: HashMap::new(),
        };
        
        let local_def = ModelDefinition {
            id: "local-llama".to_string(),
            name: "Local Llama".to_string(),
            version: "2.0".to_string(),
            provider: "local".to_string(),
            capabilities: vec![
                ModelCapability::TextGeneration,
            ],
            available: true,
            cost_factor: 0.1,
            max_input_size: 2000,
            max_output_size: 1000,
            config: HashMap::new(),
        };
        
        // Register models
        router.register_model(OpenAIBackend::new(
            gpt4_def, 
            "fake-api-key", 
            "https://api.openai.com/v1"
        )).await.unwrap();
        
        router.register_model(OpenAIBackend::new(
            gpt35_def,
            "fake-api-key",
            "https://api.openai.com/v1"
        )).await.unwrap();
        
        router.register_model(LocalModelBackend::new(
            local_def,
            "/path/to/model.bin"
        )).await.unwrap();
        
        // Test lowest cost routing - should select the local model
        let input = ModelInput::text("Hello, how are you?");
        let result = router.route_request(
            input,
            RoutingStrategy::LowestCost,
            &[ModelCapability::TextGeneration],
        ).await.unwrap();
        
        assert!(result.success);
        assert!(result.text.unwrap().contains("Local model"));
        
        // Test specific model routing - select GPT-4
        let input = ModelInput::text("What is the meaning of life?");
        let result = router.route_request(
            input,
            RoutingStrategy::Specific("gpt-4".to_string()),
            &[ModelCapability::TextGeneration, ModelCapability::Reasoning],
        ).await.unwrap();
        
        assert!(result.success);
        assert!(result.text.unwrap().contains("OpenAI response"));
        
        // Test capability filtering - need QA capability, should not select local model
        let available = router.get_available_models(&[
            ModelCapability::TextGeneration,
            ModelCapability::QuestionAnswering,
        ]).await;
        
        assert_eq!(available.len(), 2);
        assert!(available.iter().any(|m| m.id == "gpt-4"));
        assert!(available.iter().any(|m| m.id == "gpt-3.5-turbo"));
    }

    // Helper to create a CSV file with test data
    fn create_test_csv(dir: &TempDir, filename: &str, content: &str) -> Result<String> {
        let file_path = dir.path().join(filename);
        let mut file = File::create(&file_path)?;
        file.write_all(content.as_bytes())?;
        Ok(file_path.to_string_lossy().to_string())
    }

    #[test]
    fn test_integration_service() -> Result<()> {
        // Create runtime
        let rt = Runtime::new()?;
        
        // Create temp directory for test files
        let temp_dir = TempDir::new()?;
        
        // Create primary CSV
        let primary_csv = create_test_csv(&temp_dir, "primary.csv", r#"id,name,age,city
1,John,30,New York
2,Jane,25,San Francisco
3,Bob,40,Chicago
"#)?;

        // Create secondary CSV
        let secondary_csv = create_test_csv(&temp_dir, "secondary.csv", r#"id,name,salary,department
1,John,75000,Engineering
2,Jane,85000,Marketing
4,Alice,90000,HR
"#)?;

        // Run the test
        rt.block_on(async {
            // Create service
            let service = create_data_integration_service();
            
            // Define primary source
            let primary_source = DataSource {
                id: "primary".to_string(),
                name: "Primary Source".to_string(),
                source_type: SourceType::Csv,
                connection: {
                    let mut conn = HashMap::new();
                    conn.insert("path".to_string(), primary_csv);
                    conn
                },
                field_mappings: {
                    let mut mappings = HashMap::new();
                    mappings.insert("id".to_string(), "id".to_string());
                    mappings.insert("name".to_string(), "name".to_string());
                    mappings.insert("age".to_string(), "age".to_string());
                    mappings.insert("city".to_string(), "city".to_string());
                    mappings
                },
                fields: vec![
                    Field {
                        id: "id".to_string(),
                        name: "ID".to_string(),
                        description: Some("Unique identifier".to_string()),
                        field_type: FieldType::String,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "name".to_string(),
                        name: "Name".to_string(),
                        description: Some("Person's name".to_string()),
                        field_type: FieldType::String,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "age".to_string(),
                        name: "Age".to_string(),
                        description: Some("Person's age".to_string()),
                        field_type: FieldType::Integer,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "city".to_string(),
                        name: "City".to_string(),
                        description: Some("Person's city".to_string()),
                        field_type: FieldType::String,
                        required: false,
                        default_value: None,
                        validation: None,
                    },
                ],
            };
            
            // Define secondary source
            let secondary_source = DataSource {
                id: "secondary".to_string(),
                name: "Secondary Source".to_string(),
                source_type: SourceType::Csv,
                connection: {
                    let mut conn = HashMap::new();
                    conn.insert("path".to_string(), secondary_csv);
                    conn
                },
                field_mappings: {
                    let mut mappings = HashMap::new();
                    mappings.insert("id".to_string(), "id".to_string());
                    mappings.insert("name".to_string(), "name".to_string());
                    mappings.insert("salary".to_string(), "salary".to_string());
                    mappings.insert("department".to_string(), "department".to_string());
                    mappings
                },
                fields: vec![
                    Field {
                        id: "id".to_string(),
                        name: "ID".to_string(),
                        description: Some("Unique identifier".to_string()),
                        field_type: FieldType::String,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "name".to_string(),
                        name: "Name".to_string(),
                        description: Some("Person's name".to_string()),
                        field_type: FieldType::String,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "salary".to_string(),
                        name: "Salary".to_string(),
                        description: Some("Person's salary".to_string()),
                        field_type: FieldType::Integer,
                        required: true,
                        default_value: None,
                        validation: None,
                    },
                    Field {
                        id: "department".to_string(),
                        name: "Department".to_string(),
                        description: Some("Person's department".to_string()),
                        field_type: FieldType::String,
                        required: false,
                        default_value: None,
                        validation: None,
                    },
                ],
            };
            
            // Register sources
            service.register_source(primary_source).await?;
            service.register_source(secondary_source).await?;
            
            // Create integration config
            let config = IntegrationConfig {
                primary_source: "primary".to_string(),
                secondary_sources: vec!["secondary".to_string()],
                mode: IntegrationMode::FullOuterJoin,
                key_fields: vec!["id".to_string()],
                matching_strategy: KeyMatchingStrategy::Exact,
                conflict_strategy: ConflictStrategy::PreferPrimary,
                timeout_ms: None,
            };
            
            // Integrate data
            let result = service.integrate(&config).await?;
            
            // Verify results
            assert_eq!(result.len(), 4, "Should have 4 records after integration");
            
            // Check each record
            for record in &result {
                match record.get(&"id".to_string()) {
                    Some(FieldValue::String(id)) => {
                        match id.as_str() {
                            "1" => {
                                // John should have data from both sources
                                assert_eq!(record.get(&"name".to_string()).map(|v| v.to_string_value()), Some("John".to_string()));
                                assert_eq!(record.get(&"age".to_string()).map(|v| v.to_string_value()), Some("30".to_string()));
                                assert_eq!(record.get(&"city".to_string()).map(|v| v.to_string_value()), Some("New York".to_string()));
                                assert_eq!(record.get(&"salary".to_string()).map(|v| v.to_string_value()), Some("75000".to_string()));
                                assert_eq!(record.get(&"department".to_string()).map(|v| v.to_string_value()), Some("Engineering".to_string()));
                            },
                            "2" => {
                                // Jane should have data from both sources
                                assert_eq!(record.get(&"name".to_string()).map(|v| v.to_string_value()), Some("Jane".to_string()));
                                assert_eq!(record.get(&"age".to_string()).map(|v| v.to_string_value()), Some("25".to_string()));
                                assert_eq!(record.get(&"city".to_string()).map(|v| v.to_string_value()), Some("San Francisco".to_string()));
                                assert_eq!(record.get(&"salary".to_string()).map(|v| v.to_string_value()), Some("85000".to_string()));
                                assert_eq!(record.get(&"department".to_string()).map(|v| v.to_string_value()), Some("Marketing".to_string()));
                            },
                            "3" => {
                                // Bob should only have primary data
                                assert_eq!(record.get(&"name".to_string()).map(|v| v.to_string_value()), Some("Bob".to_string()));
                                assert_eq!(record.get(&"age".to_string()).map(|v| v.to_string_value()), Some("40".to_string()));
                                assert_eq!(record.get(&"city".to_string()).map(|v| v.to_string_value()), Some("Chicago".to_string()));
                                assert_eq!(record.get(&"salary".to_string()), None);
                                assert_eq!(record.get(&"department".to_string()), None);
                            },
                            "4" => {
                                // Alice should only have secondary data
                                assert_eq!(record.get(&"name".to_string()).map(|v| v.to_string_value()), Some("Alice".to_string()));
                                assert_eq!(record.get(&"age".to_string()), None);
                                assert_eq!(record.get(&"city".to_string()), None);
                                assert_eq!(record.get(&"salary".to_string()).map(|v| v.to_string_value()), Some("90000".to_string()));
                                assert_eq!(record.get(&"department".to_string()).map(|v| v.to_string_value()), Some("HR".to_string()));
                            },
                            _ => panic!("Unexpected ID: {}", id),
                        }
                    },
                    _ => panic!("Invalid ID field"),
                }
            }
            
            // Try different integration mode
            let intersection_config = IntegrationConfig {
                primary_source: "primary".to_string(),
                secondary_sources: vec!["secondary".to_string()],
                mode: IntegrationMode::Intersection,
                key_fields: vec!["id".to_string()],
                matching_strategy: KeyMatchingStrategy::Exact,
                conflict_strategy: ConflictStrategy::PreferPrimary,
                timeout_ms: None,
            };
            
            let intersection_result = service.integrate(&intersection_config).await?;
            
            // Intersection should only return records in both sources
            assert_eq!(intersection_result.len(), 2, "Intersection should only have John and Jane");
            
            Ok(())
        })
    }
}

/// Data integration service for combining data from multiple sources
pub struct DataIntegrationService {
    /// Data sources by ID
    sources: Arc<RwLock<HashMap<SourceId, DataSource>>>,
    /// Readers by source type
    readers: HashMap<SourceType, Arc<dyn DataReader>>,
    /// Writers by source type
    writers: HashMap<SourceType, Arc<dyn DataWriter>>,
}

impl DataIntegrationService {
    /// Create a new data integration service
    pub fn new() -> Self {
        let mut readers = HashMap::new();
        readers.insert(SourceType::Csv, Arc::new(CsvReader::new()) as Arc<dyn DataReader>);
        
        // In a real implementation, we would have readers for other source types
        
        Self {
            sources: Arc::new(RwLock::new(HashMap::new())),
            readers,
            writers: HashMap::new(),
        }
    }
    
    /// Register a data source
    pub async fn register_source(&self, source: DataSource) -> Result<()> {
        let mut sources = self.sources.write().await;
        sources.insert(source.id.clone(), source);
        Ok(())
    }
    
    /// Get a data source by ID
    pub async fn get_source(&self, source_id: &SourceId) -> Result<DataSource> {
        let sources = self.sources.read().await;
        sources.get(source_id)
            .cloned()
            .ok_or_else(|| anyhow!("Data source not found: {}", source_id))
    }
    
    /// Get a reader for a source type
    fn get_reader(&self, source_type: SourceType) -> Result<&Arc<dyn DataReader>> {
        self.readers.get(&source_type)
            .ok_or_else(|| anyhow!("Reader not available for source type: {:?}", source_type))
    }
    
    /// Get a writer for a source type
    fn get_writer(&self, source_type: SourceType) -> Result<&Arc<dyn DataWriter>> {
        self.writers.get(&source_type)
            .ok_or_else(|| anyhow!("Writer not available for source type: {:?}", source_type))
    }
    
    /// Read records from a data source
    pub async fn read_records(&self, source_id: &SourceId) -> Result<Vec<Record>> {
        let source = self.get_source(source_id).await?;
        let reader = self.get_reader(source.source_type)?;
        reader.read_all(&source).await
    }
    
    /// Create a key for record matching
    fn create_key(&self, record: &Record, key_fields: &[FieldId]) -> String {
        let mut key_parts = Vec::new();
        
        for field_id in key_fields {
            let value = match record.get(field_id) {
                Some(v) => v.to_string_value(),
                None => String::new(),
            };
            
            key_parts.push(value);
        }
        
        key_parts.join("|")
    }
    
    /// Match records based on key fields and matching strategy
    fn match_records(
        &self,
        primary_record: &Record,
        secondary_record: &Record,
        key_fields: &[FieldId],
        strategy: &KeyMatchingStrategy,
    ) -> bool {
        match strategy {
            KeyMatchingStrategy::Exact => {
                // Create keys and compare
                let primary_key = self.create_key(primary_record, key_fields);
                let secondary_key = self.create_key(secondary_record, key_fields);
                
                primary_key == secondary_key
            }
            KeyMatchingStrategy::Fuzzy { threshold } => {
                // For fuzzy matching, we'll use a simple similarity score
                // In a real implementation, you would use a more sophisticated algorithm
                
                let primary_key = self.create_key(primary_record, key_fields);
                let secondary_key = self.create_key(secondary_record, key_fields);
                
                if primary_key.is_empty() || secondary_key.is_empty() {
                    return false;
                }
                
                // Calculate similarity (simple implementation)
                let max_len = primary_key.len().max(secondary_key.len()) as f64;
                let common_prefix_len = primary_key.chars()
                    .zip(secondary_key.chars())
                    .take_while(|(a, b)| a == b)
                    .count() as f64;
                    
                let similarity = common_prefix_len / max_len;
                
                similarity >= *threshold
            }
            KeyMatchingStrategy::Custom { name } => {
                // In a real implementation, you would have a registry of custom matching functions
                warn!("Custom matching function not implemented: {}", name);
                false
            }
        }
    }
    
    /// Resolve conflicting field values
    fn resolve_conflict(
        &self,
        field_id: &FieldId,
        primary_value: &FieldValue,
        secondary_value: &FieldValue,
        strategy: &ConflictStrategy,
        primary_record: &Record,
        secondary_record: &Record,
    ) -> Result<FieldValue> {
        match strategy {
            ConflictStrategy::PreferPrimary => {
                Ok(primary_value.clone())
            }
            ConflictStrategy::PreferSecondary => {
                Ok(secondary_value.clone())
            }
            ConflictStrategy::MostRecent { timestamp_field } => {
                // Get timestamp values
                let primary_timestamp = match primary_record.get(timestamp_field) {
                    Some(FieldValue::DateTime(dt)) => Some(*dt),
                    _ => None,
                };
                
                let secondary_timestamp = match secondary_record.get(timestamp_field) {
                    Some(FieldValue::DateTime(dt)) => Some(*dt),
                    _ => None,
                };
                
                // Compare timestamps
                match (primary_timestamp, secondary_timestamp) {
                    (Some(pt), Some(st)) if pt > st => Ok(primary_value.clone()),
                    (Some(_), Some(_)) => Ok(secondary_value.clone()),
                    (Some(_), None) => Ok(primary_value.clone()),
                    (None, Some(_)) => Ok(secondary_value.clone()),
                    (None, None) => Ok(primary_value.clone()), // Default to primary
                }
            }
            ConflictStrategy::Custom { function } => {
                // In a real implementation, you would have a registry of custom functions
                warn!("Custom conflict resolution function not implemented: {}", function);
                Ok(primary_value.clone())
            }
        }
    }
    
    /// Merge two records
    fn merge_records(
        &self,
        primary_record: &Record,
        secondary_record: &Record,
        conflict_strategy: &ConflictStrategy,
    ) -> Result<Record> {
        let mut merged_data = primary_record.data.clone();
        
        // For each field in the secondary record
        for (field_id, secondary_value) in &secondary_record.data {
            match primary_record.get(field_id) {
                Some(primary_value) => {
                    // Resolve conflict
                    let resolved_value = self.resolve_conflict(
                        field_id,
                        primary_value,
                        secondary_value,
                        conflict_strategy,
                        primary_record,
                        secondary_record,
                    )?;
                    
                    merged_data.insert(field_id.clone(), resolved_value);
                }
                None => {
                    // No conflict, just add the secondary value
                    merged_data.insert(field_id.clone(), secondary_value.clone());
                }
            }
        }
        
        // Create merged record
        let mut merged_record = Record::new(primary_record.source_id.clone(), merged_data);
        
        // Merge metadata
        for (key, value) in &secondary_record.metadata {
            if !merged_record.metadata.contains_key(key) {
                merged_record.metadata.insert(key.clone(), value.clone());
            }
        }
        
        Ok(merged_record)
    }
    
    /// Integrate data from multiple sources
    #[instrument(skip(self))]
    pub async fn integrate(&self, config: &IntegrationConfig) -> Result<Vec<Record>> {
        // Get primary source
        let primary_source = self.get_source(&config.primary_source).await?;
        let primary_reader = self.get_reader(primary_source.source_type)?;
        
        // Read primary records
        info!("Reading records from primary source: {}", primary_source.id);
        let primary_records = primary_reader.read_all(&primary_source).await?;
        info!("Read {} records from primary source", primary_records.len());
        
        // Initialize result
        let mut result = match config.mode {
            IntegrationMode::Union | IntegrationMode::LeftJoin | IntegrationMode::FullOuterJoin => {
                // Start with all primary records
                primary_records.clone()
            }
            IntegrationMode::Intersection | IntegrationMode::RightJoin => {
                // Start with empty set
                Vec::new()
            }
        };
        
        // Index primary records by key for faster lookup
        let mut primary_by_key = HashMap::new();
        for record in &primary_records {
            let key = self.create_key(record, &config.key_fields);
            primary_by_key.insert(key, record);
        }
        
        // Process each secondary source
        for secondary_source_id in &config.secondary_sources {
            let secondary_source = self.get_source(secondary_source_id).await?;
            let secondary_reader = self.get_reader(secondary_source.source_type)?;
            
            // Read secondary records
            info!("Reading records from secondary source: {}", secondary_source.id);
            let secondary_records = secondary_reader.read_all(&secondary_source).await?;
            info!("Read {} records from secondary source", secondary_records.len());
            
            // Track matched primary records
            let mut matched_primary = HashSet::new();
            
            // Process each secondary record
            for secondary_record in &secondary_records {
                let secondary_key = self.create_key(secondary_record, &config.key_fields);
                
                // Find matching primary record
                let matched = primary_by_key.iter()
                    .find(|(primary_key, primary_record)| {
                        // Check if keys match or use custom matching
                        primary_key == &&secondary_key ||
                        self.match_records(
                            primary_record,
                            secondary_record,
                            &config.key_fields,
                            &config.matching_strategy,
                        )
                    })
                    .map(|(key, record)| (key.clone(), record));
                
                match (matched, config.mode) {
                    // Matched record in Union, Intersection, LeftJoin, or FullOuterJoin mode
                    (Some((primary_key, primary_record)), IntegrationMode::Union | 
                                                          IntegrationMode::Intersection | 
                                                          IntegrationMode::LeftJoin |
                                                          IntegrationMode::FullOuterJoin) => {
                        // Merge records
                        let merged = self.merge_records(
                            primary_record,
                            secondary_record,
                            &config.conflict_strategy,
                        )?;
                        
                        // Replace in result
                        if let Some(idx) = result.iter().position(|r| r.id == primary_record.id) {
                            result[idx] = merged;
                        }
                        
                        // Mark as matched
                        matched_primary.insert(primary_key);
                    }
                    // Matched record in RightJoin mode
                    (Some((primary_key, primary_record)), IntegrationMode::RightJoin) => {
                        // Merge records
                        let merged = self.merge_records(
                            primary_record,
                            secondary_record,
                            &config.conflict_strategy,
                        )?;
                        
                        // Add to result
                        result.push(merged);
                        
                        // Mark as matched
                        matched_primary.insert(primary_key);
                    }
                    // Unmatched record in Union or FullOuterJoin mode
                    (None, IntegrationMode::Union | IntegrationMode::FullOuterJoin | IntegrationMode::RightJoin) => {
                        // Add secondary record
                        result.push(secondary_record.clone());
                    }
                    // Unmatched record in other modes
                    _ => {
                        // Skip
                    }
                }
            }
            
            // Handle unmatched primary records
            if config.mode == IntegrationMode::Intersection || config.mode == IntegrationMode::RightJoin {
                // Remove unmatched primary records
                result.retain(|r| {
                    let key = self.create_key(r, &config.key_fields);
                    matched_primary.contains(&key)
                });
            }
        }
        
        info!("Integration complete, produced {} records", result.len());
        Ok(result)
    }
}

/// Factory function to create a data integration service
pub fn create_data_integration_service() -> DataIntegrationService {
    DataIntegrationService::new()
}