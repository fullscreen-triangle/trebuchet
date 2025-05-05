use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use trebuchet_core::prelude::*;

/// Represents a security event that needs to be analyzed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    /// Unique identifier for the event
    pub id: String,
    /// Timestamp when the event occurred
    pub timestamp: u64,
    /// Source of the event (IP address, service name, etc.)
    pub source: String,
    /// Type of event (login, access, error, etc.)
    pub event_type: String,
    /// Severity level (info, warning, error, critical)
    pub severity: EventSeverity,
    /// Additional event-specific data
    pub data: HashMap<String, String>,
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Informational event, no action needed
    Info,
    /// Warning event, might need attention
    Warning,
    /// Error event, requires attention
    Error,
    /// Critical event, immediate action required
    Critical,
}

impl SecurityEvent {
    /// Create a new security event
    pub fn new(
        id: impl Into<String>,
        timestamp: u64,
        source: impl Into<String>,
        event_type: impl Into<String>,
        severity: EventSeverity,
    ) -> Self {
        Self {
            id: id.into(),
            timestamp,
            source: source.into(),
            event_type: event_type.into(),
            severity,
            data: HashMap::new(),
        }
    }
    
    /// Add data to the event
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
}

/// Represents an encryption key pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPair {
    /// Key identifier
    pub id: String,
    /// Public key (PEM format)
    pub public_key: String,
    /// Private key (PEM format)
    pub private_key: Option<String>,
    /// Key algorithm (RSA, EC, etc.)
    pub algorithm: String,
    /// Key strength (bits)
    pub strength: usize,
    /// Metadata about the key
    pub metadata: HashMap<String, String>,
}

/// Crypto service for encryption/decryption operations
pub struct CryptoService {
    /// Available key pairs
    keys: RwLock<HashMap<String, KeyPair>>,
}

impl CryptoService {
    /// Create a new crypto service
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a key pair
    pub async fn register_key(&self, key_pair: KeyPair) -> Result<()> {
        let mut keys = self.keys.write().await;
        keys.insert(key_pair.id.clone(), key_pair);
        Ok(())
    }
    
    /// Get a key pair by ID
    pub async fn get_key(&self, id: &str) -> Result<KeyPair> {
        let keys = self.keys.read().await;
        keys.get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Key '{}' not found", id))
    }
    
    /// Encrypt data using a specified key
    pub async fn encrypt(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>> {
        let key = self.get_key(key_id).await?;
        
        // In a real implementation, this would use actual cryptography libraries
        // For this example, we'll just return a dummy result
        let mut result = Vec::with_capacity(data.len() + 16);
        result.extend_from_slice(b"ENCRYPTED:"); // Just a prefix to simulate encryption
        result.extend_from_slice(data);
        
        Ok(result)
    }
    
    /// Decrypt data using a specified key
    pub async fn decrypt(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>> {
        let key = self.get_key(key_id).await?;
        
        // Check if we have the private key
        if key.private_key.is_none() {
            return Err(anyhow!("Private key not available for '{}'", key_id));
        }
        
        // In a real implementation, this would use actual cryptography libraries
        // For this example, we'll just strip the prefix we added in encrypt()
        if data.len() < 10 || !data.starts_with(b"ENCRYPTED:") {
            return Err(anyhow!("Invalid encrypted data format"));
        }
        
        Ok(data[10..].to_vec())
    }
    
    /// Generate a secure hash of data
    pub fn hash_data(&self, data: &[u8], algorithm: &str) -> Result<String> {
        // In a real implementation, this would use a proper hashing library
        // For this example, we'll just return a dummy hash
        let hash = format!("{}:{}", algorithm, hex::encode(data)[..16].to_string());
        Ok(hash)
    }
}

/// Security analyzer for threat detection
pub struct SecurityAnalyzer {
    /// Event pattern rules for detection
    rules: RwLock<HashMap<String, SecurityRule>>,
    /// Event cache for correlation
    event_cache: RwLock<Vec<SecurityEvent>>,
}

/// Security detection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Event types this rule applies to
    pub event_types: Vec<String>,
    /// Minimum severity to trigger this rule
    pub min_severity: EventSeverity,
    /// Pattern to match in event data
    pub pattern: HashMap<String, String>,
}

impl SecurityAnalyzer {
    /// Create a new security analyzer
    pub fn new() -> Self {
        Self {
            rules: RwLock::new(HashMap::new()),
            event_cache: RwLock::new(Vec::new()),
        }
    }
    
    /// Register a security rule
    pub async fn register_rule(&self, rule: SecurityRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }
    
    /// Analyze a security event
    pub async fn analyze_event(&self, event: SecurityEvent) -> Result<Vec<SecurityAlert>> {
        let rules = self.rules.read().await;
        let mut alerts = Vec::new();
        
        // Add event to cache for correlation
        {
            let mut cache = self.event_cache.write().await;
            cache.push(event.clone());
            
            // Limit cache size
            if cache.len() > 1000 {
                cache.remove(0);
            }
        }
        
        // Check each rule for matches
        for rule in rules.values() {
            // Skip if event type doesn't match any in the rule
            if !rule.event_types.is_empty() && !rule.event_types.contains(&event.event_type) {
                continue;
            }
            
            // Skip if severity is below the minimum for this rule
            if event.severity as u8 < rule.min_severity as u8 {
                continue;
            }
            
            // Check if pattern matches
            let mut matches = true;
            for (key, value) in &rule.pattern {
                if let Some(event_value) = event.data.get(key) {
                    if event_value != value {
                        matches = false;
                        break;
                    }
                } else {
                    matches = false;
                    break;
                }
            }
            
            if matches {
                alerts.push(SecurityAlert {
                    rule_id: rule.id.clone(),
                    rule_name: rule.name.clone(),
                    event_id: event.id.clone(),
                    timestamp: event.timestamp,
                    description: rule.description.clone(),
                    severity: rule.min_severity,
                });
            }
        }
        
        Ok(alerts)
    }
    
    /// Find correlated events based on source and time window
    pub async fn find_correlated_events(&self, source: &str, time_window: u64, now: u64) -> Result<Vec<SecurityEvent>> {
        let cache = self.event_cache.read().await;
        
        // Filter events by source and time window
        let correlated: Vec<SecurityEvent> = cache.iter()
            .filter(|event| {
                event.source == source && 
                now >= event.timestamp && 
                now - event.timestamp <= time_window
            })
            .cloned()
            .collect();
            
        Ok(correlated)
    }
}

/// Security alert generated when a rule matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    /// ID of the rule that generated this alert
    pub rule_id: String,
    /// Name of the rule
    pub rule_name: String,
    /// ID of the event that triggered the alert
    pub event_id: String,
    /// When the alert was generated
    pub timestamp: u64,
    /// Description of the alert
    pub description: String,
    /// Severity of the alert
    pub severity: EventSeverity,
}

#[derive(Error, Debug)]
pub enum MoriartyError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("OpenCV error: {0}")]
    OpenCvError(String),

    #[error("Video processing error: {0}")]
    VideoProcessingError(String),

    #[error("Pose detection error: {0}")]
    PoseDetectionError(String),

    #[error("Analysis error: {0}")]
    AnalysisError(String),

    #[error("Resource error: {0}")]
    ResourceError(String),

    #[error("Model error: {0}")]
    ModelError(String),
}

// Core data structures for pose analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skeleton {
    pub keypoints: Vec<Point3D>,
    pub person_id: Option<u32>,
    pub confidence: f32,
    pub timestamp: f64,
    pub frame_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomechanicalData {
    pub joint_angles: Vec<JointAngle>,
    pub velocities: Vec<JointVelocity>,
    pub accelerations: Vec<JointAcceleration>,
    pub forces: Vec<JointForce>,
    pub stride_metrics: Option<StrideMetrics>,
    pub center_of_mass: Option<Point3D>,
    pub timestamp: f64,
    pub frame_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointAngle {
    pub joint_name: String,
    pub angle: f32,          // in degrees
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointVelocity {
    pub joint_name: String,
    pub velocity: Point3D,   // in m/s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointAcceleration {
    pub joint_name: String,
    pub acceleration: Point3D, // in m/s²
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointForce {
    pub joint_name: String,
    pub force: Point3D,      // in N
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideMetrics {
    pub stride_length: f32,  // in meters
    pub stride_rate: f32,    // in strides/second
    pub contact_time: f32,   // in seconds
    pub flight_time: f32,    // in seconds
    pub stance_phase: f32,   // percentage
    pub swing_phase: f32,    // percentage
    pub vertical_oscillation: f32, // in cm
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub use_gpu: bool,
    pub pose_complexity: u32, // 0=fast, 1=balanced, 2=accurate
    pub tracking_enabled: bool,
    pub worker_count: Option<usize>,
    pub memory_limit_mb: Option<usize>,
    pub confidence_threshold: f32,
    pub gpu_memory_limit_mb: Option<usize>,
    pub video_resolution: Option<(u32, u32)>, // (width, height)
    pub fps_target: Option<f32>,
    pub output_visualizations: bool,
    pub enable_llm_analysis: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            pose_complexity: 1,
            tracking_enabled: true,
            worker_count: None,
            memory_limit_mb: None,
            confidence_threshold: 0.6,
            gpu_memory_limit_mb: None,
            video_resolution: None,
            fps_target: None,
            output_visualizations: true,
            enable_llm_analysis: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub video_path: String,
    pub skeletons: Vec<Skeleton>,
    pub biomechanical_data: Vec<BiomechanicalData>,
    pub metrics: PerformanceMetrics,
    pub llm_insights: Option<String>,
    pub visualization_paths: Option<Vec<String>>,
    pub processing_stats: ProcessingStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub overall_score: Option<f32>,
    pub metric_scores: std::collections::HashMap<String, f32>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub processing_time_ms: u64,
    pub frame_count: u32,
    pub detected_persons: u32,
    pub average_confidence: f32,
    pub memory_usage_mb: u64,
    pub gpu_memory_usage_mb: Option<u64>,
}

pub struct MoriartyService {
    config: ProcessingConfig,
    pose_detector: PoseDetector,
    dynamics_analyzer: DynamicsAnalyzer,
    motion_analyzer: MotionAnalyzer,
    scene_analyzer: SceneAnalyzer,
    llm_integration: Option<Arc<RwLock<LlmIntegration>>>,
    memory_monitor: MemoryMonitor,
}

impl MoriartyService {
    pub async fn new(config: ProcessingConfig) -> Result<Self> {
        let worker_count = config.worker_count.unwrap_or_else(|| {
            std::cmp::max(1, num_cpus::get().saturating_sub(1))
        });
        
        info!("Initializing MoriartyService with {} workers, GPU: {}", 
              worker_count, config.use_gpu);
        
        let pose_detector = PoseDetector::new(config.use_gpu, config.pose_complexity)?;
        let dynamics_analyzer = DynamicsAnalyzer::new(config.use_gpu)?;
        let motion_analyzer = MotionAnalyzer::new(config.use_gpu)?;
        let scene_analyzer = SceneAnalyzer::new(config.use_gpu)?;
        
        let llm_integration = if config.enable_llm_analysis {
            Some(Arc::new(RwLock::new(LlmIntegration::new().await?)))
        } else {
            None
        };
        
        let memory_monitor = MemoryMonitor::new(config.memory_limit_mb, config.gpu_memory_limit_mb)?;
        
        Ok(Self {
            config,
            pose_detector,
            dynamics_analyzer,
            motion_analyzer,
            scene_analyzer,
            llm_integration,
            memory_monitor,
        })
    }
    
    pub async fn analyze_video(&self, video_path: impl AsRef<Path>) -> Result<AnalysisResult> {
        let start_time = std::time::Instant::now();
        let video_path = video_path.as_ref().to_path_buf();
        
        info!("Analyzing video: {}", video_path.display());
        self.memory_monitor.check_resources()?;
        
        // 1. Load and preprocess video
        let video_frames = self.scene_analyzer.extract_frames(&video_path, self.config.video_resolution, self.config.fps_target).await?;
        debug!("Extracted {} frames from video", video_frames.len());
        
        // 2. Detect poses in frames
        let skeletons = self.pose_detector.detect_poses(&video_frames, self.config.confidence_threshold, self.config.tracking_enabled).await?;
        debug!("Detected {} skeletons", skeletons.len());
        
        // 3. Analyze biomechanics
        let biomechanical_data = self.dynamics_analyzer.analyze_biomechanics(&skeletons).await?;
        
        // 4. Calculate performance metrics
        let metrics = self.motion_analyzer.calculate_metrics(&skeletons, &biomechanical_data).await?;
        
        // 5. Generate LLM insights if enabled
        let llm_insights = if let Some(llm) = &self.llm_integration {
            let llm = llm.read().await;
            match llm.generate_insights(&skeletons, &biomechanical_data, &metrics).await {
                Ok(insights) => Some(insights),
                Err(e) => {
                    warn!("Failed to generate LLM insights: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // 6. Generate visualizations if enabled
        let visualization_paths = if self.config.output_visualizations {
            let output_dir = video_path.parent().unwrap_or(Path::new(".")).join("output");
            std::fs::create_dir_all(&output_dir)?;
            
            match self.scene_analyzer.generate_visualizations(
                &video_frames, &skeletons, &biomechanical_data, &output_dir
            ).await {
                Ok(paths) => Some(paths),
                Err(e) => {
                    warn!("Failed to generate visualizations: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // 7. Collect processing stats
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        let memory_usage = self.memory_monitor.get_memory_usage()?;
        let gpu_memory_usage = if self.config.use_gpu {
            self.memory_monitor.get_gpu_memory_usage().ok()
        } else {
            None
        };
        
        let frame_count = video_frames.len() as u32;
        let detected_persons = if skeletons.is_empty() {
            0
        } else {
            // Count unique person IDs across all frames
            let mut person_ids = std::collections::HashSet::new();
            for skeleton in &skeletons {
                if let Some(id) = skeleton.person_id {
                    person_ids.insert(id);
                }
            }
            person_ids.len() as u32
        };
        
        let average_confidence = if skeletons.is_empty() {
            0.0
        } else {
            skeletons.iter().map(|s| s.confidence).sum::<f32>() / skeletons.len() as f32
        };
        
        let processing_stats = ProcessingStats {
            processing_time_ms,
            frame_count,
            detected_persons,
            average_confidence,
            memory_usage_mb: memory_usage,
            gpu_memory_usage_mb: gpu_memory_usage,
        };
        
        info!("Video analysis complete in {}ms, {} frames processed", 
              processing_time_ms, frame_count);
        
        Ok(AnalysisResult {
            video_path: video_path.to_string_lossy().to_string(),
            skeletons,
            biomechanical_data,
            metrics,
            llm_insights,
            visualization_paths: visualization_paths.map(|paths| {
                paths.iter().map(|p| p.to_string_lossy().to_string()).collect()
            }),
            processing_stats,
        })
    }
    
    pub async fn batch_analyze(&self, video_paths: Vec<PathBuf>) -> Result<Vec<AnalysisResult>> {
        info!("Batch analyzing {} videos", video_paths.len());
        
        // In a full implementation, this would use parallel processing with proper resource management
        // For simplicity, we're just processing videos sequentially
        let mut results = Vec::with_capacity(video_paths.len());
        
        for path in video_paths {
            match self.analyze_video(&path).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Failed to analyze video {}: {}", path.display(), e);
                    // Continue with other videos
                }
            }
        }
        
        Ok(results)
    }
    
    pub async fn compare_videos(&self, video_paths: Vec<PathBuf>) -> Result<ComparisonResult> {
        // Analyze all videos
        let analyses = self.batch_analyze(video_paths).await?;
        
        if analyses.len() < 2 {
            return Err(anyhow::anyhow!("At least two videos are required for comparison"));
        }
        
        // For a real implementation, this would perform sophisticated comparison
        // This is just a placeholder
        let comparisons = self.motion_analyzer.compare_analyses(&analyses).await?;
        
        Ok(comparisons)
    }
}

// Core component implementations
struct PoseDetector {
    use_gpu: bool,
    complexity: u32,
}

impl PoseDetector {
    fn new(use_gpu: bool, complexity: u32) -> Result<Self> {
        Ok(Self { use_gpu, complexity })
    }
    
    async fn detect_poses(&self, frames: &[Vec<u8>], confidence_threshold: f32, tracking_enabled: bool) -> Result<Vec<Skeleton>> {
        // In a real implementation, this would use OpenCV and ML models to detect poses
        info!("Detecting poses in {} frames (GPU: {}, tracking: {})", 
              frames.len(), self.use_gpu, tracking_enabled);
        
        // Placeholder for actual implementation
        let skeletons = Vec::new();
        
        Ok(skeletons)
    }
}

struct DynamicsAnalyzer {
    use_gpu: bool,
}

impl DynamicsAnalyzer {
    fn new(use_gpu: bool) -> Result<Self> {
        Ok(Self { use_gpu })
    }
    
    async fn analyze_biomechanics(&self, skeletons: &[Skeleton]) -> Result<Vec<BiomechanicalData>> {
        info!("Analyzing biomechanics for {} skeletons", skeletons.len());
        
        // Placeholder for actual implementation
        let biomechanical_data = Vec::new();
        
        Ok(biomechanical_data)
    }
}

struct MotionAnalyzer {
    use_gpu: bool,
}

impl MotionAnalyzer {
    fn new(use_gpu: bool) -> Result<Self> {
        Ok(Self { use_gpu })
    }
    
    async fn calculate_metrics(&self, skeletons: &[Skeleton], biomechanics: &[BiomechanicalData]) -> Result<PerformanceMetrics> {
        info!("Calculating performance metrics");
        
        // Placeholder for actual implementation
        let metrics = PerformanceMetrics {
            overall_score: Some(0.0),
            metric_scores: std::collections::HashMap::new(),
            recommendations: Vec::new(),
        };
        
        Ok(metrics)
    }
    
    async fn compare_analyses(&self, analyses: &[AnalysisResult]) -> Result<ComparisonResult> {
        info!("Comparing {} analyses", analyses.len());
        
        // Placeholder for actual implementation
        let comparison = ComparisonResult {
            video_paths: analyses.iter().map(|a| a.video_path.clone()).collect(),
            metric_comparisons: std::collections::HashMap::new(),
            joint_angle_comparisons: Vec::new(),
            stride_comparisons: None,
            visualizations: None,
        };
        
        Ok(comparison)
    }
}

struct SceneAnalyzer {
    use_gpu: bool,
}

impl SceneAnalyzer {
    fn new(use_gpu: bool) -> Result<Self> {
        Ok(Self { use_gpu })
    }
    
    async fn extract_frames(&self, video_path: &Path, resolution: Option<(u32, u32)>, fps_target: Option<f32>) -> Result<Vec<Vec<u8>>> {
        info!("Extracting frames from {}", video_path.display());
        
        // Placeholder for actual implementation
        let frames = Vec::new();
        
        Ok(frames)
    }
    
    async fn generate_visualizations(
        &self,
        frames: &[Vec<u8>],
        skeletons: &[Skeleton],
        biomechanics: &[BiomechanicalData],
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        info!("Generating visualizations in {}", output_dir.display());
        
        // Placeholder for actual implementation
        let visualization_paths = Vec::new();
        
        Ok(visualization_paths)
    }
}

struct LlmIntegration {
    // Could hold model references, API configs, etc.
}

impl LlmIntegration {
    async fn new() -> Result<Self> {
        Ok(Self {})
    }
    
    async fn generate_insights(
        &self,
        skeletons: &[Skeleton],
        biomechanics: &[BiomechanicalData],
        metrics: &PerformanceMetrics,
    ) -> Result<String> {
        info!("Generating LLM insights");
        
        // Placeholder for actual implementation
        let insights = "Analysis insights would appear here".to_string();
        
        Ok(insights)
    }
}

struct MemoryMonitor {
    memory_limit_mb: Option<usize>,
    gpu_memory_limit_mb: Option<usize>,
}

impl MemoryMonitor {
    fn new(memory_limit_mb: Option<usize>, gpu_memory_limit_mb: Option<usize>) -> Result<Self> {
        Ok(Self { memory_limit_mb, gpu_memory_limit_mb })
    }
    
    fn check_resources(&self) -> Result<()> {
        // In a real implementation, this would check system memory and GPU memory
        // against configured limits and return an error if resources are insufficient
        Ok(())
    }
    
    fn get_memory_usage(&self) -> Result<u64> {
        // Placeholder implementation
        Ok(1000)
    }
    
    fn get_gpu_memory_usage(&self) -> Result<u64> {
        // Placeholder implementation
        Ok(500)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub video_paths: Vec<String>,
    pub metric_comparisons: std::collections::HashMap<String, Vec<f32>>,
    pub joint_angle_comparisons: Vec<JointAngleComparison>,
    pub stride_comparisons: Option<StrideComparison>,
    pub visualizations: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointAngleComparison {
    pub joint_name: String,
    pub angles: Vec<Vec<f32>>, // per athlete, per frame
    pub difference: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideComparison {
    pub stride_lengths: Vec<f32>,
    pub stride_rates: Vec<f32>,
    pub contact_times: Vec<f32>,
    pub flight_times: Vec<f32>,
}

// Implementation of the TrebuchetService trait for MoriartyService
#[async_trait]
impl TrebuchetService for MoriartyService {
    async fn process(&self, request: ServiceRequest) -> ServiceResponse {
        match request.command.as_str() {
            "analyze_video" => {
                if let Some(video_path) = request.params.get("video_path") {
                    match self.analyze_video(video_path).await {
                        Ok(result) => ServiceResponse::success(serde_json::to_value(result).unwrap_or_default()),
                        Err(e) => ServiceResponse::error(format!("Error analyzing video: {}", e)),
                    }
                } else {
                    ServiceResponse::error("Missing video_path parameter")
                }
            },
            "batch_analyze" => {
                if let Some(paths_json) = request.params.get("video_paths") {
                    match serde_json::from_str::<Vec<String>>(paths_json) {
                        Ok(paths) => {
                            let path_bufs: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
                            match self.batch_analyze(path_bufs).await {
                                Ok(results) => ServiceResponse::success(serde_json::to_value(results).unwrap_or_default()),
                                Err(e) => ServiceResponse::error(format!("Error in batch analysis: {}", e)),
                            }
                        },
                        Err(e) => ServiceResponse::error(format!("Invalid video_paths parameter: {}", e)),
                    }
                } else {
                    ServiceResponse::error("Missing video_paths parameter")
                }
            },
            "compare_videos" => {
                if let Some(paths_json) = request.params.get("video_paths") {
                    match serde_json::from_str::<Vec<String>>(paths_json) {
                        Ok(paths) => {
                            let path_bufs: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
                            match self.compare_videos(path_bufs).await {
                                Ok(results) => ServiceResponse::success(serde_json::to_value(results).unwrap_or_default()),
                                Err(e) => ServiceResponse::error(format!("Error comparing videos: {}", e)),
                            }
                        },
                        Err(e) => ServiceResponse::error(format!("Invalid video_paths parameter: {}", e)),
                    }
                } else {
                    ServiceResponse::error("Missing video_paths parameter")
                }
            },
            _ => ServiceResponse::error(format!("Unknown command: {}", request.command)),
        }
    }

    fn name(&self) -> &str {
        "moriarty"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }
}
