use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

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
}