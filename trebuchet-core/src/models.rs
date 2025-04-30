use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a model
pub type ModelId = Uuid;

/// Unique identifier for a job
pub type JobId = Uuid;

/// Model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Audio processing models
    Audio,
    
    /// Genomics analysis models
    Genomics,
    
    /// Natural language processing models
    Nlp,
    
    /// Computer vision models
    Vision,
    
    /// Generic ML models
    Generic,
}

/// Model metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelMetadata {
    /// Unique identifier for the model
    pub id: ModelId,
    
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model type
    pub model_type: ModelType,
    
    /// Model description
    pub description: Option<String>,
    
    /// Model creation date
    pub created_at: DateTime<Utc>,
    
    /// Model modification date
    pub updated_at: DateTime<Utc>,
    
    /// Model tags
    pub tags: Vec<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelMetadata {
    /// Create a new model metadata
    pub fn new(name: impl Into<String>, version: impl Into<String>, model_type: ModelType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            version: version.into(),
            model_type,
            description: None,
            created_at: now,
            updated_at: now,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is waiting to be processed
    Pending,
    
    /// Job is currently being processed
    Running,
    
    /// Job has been successfully completed
    Completed,
    
    /// Job has failed
    Failed,
    
    /// Job has been cancelled
    Cancelled,
}

/// Job metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobMetadata {
    /// Unique identifier for the job
    pub id: JobId,
    
    /// Job name
    pub name: String,
    
    /// Job status
    pub status: JobStatus,
    
    /// Model ID associated with the job
    pub model_id: Option<ModelId>,
    
    /// Job creation date
    pub created_at: DateTime<Utc>,
    
    /// Job start date
    pub started_at: Option<DateTime<Utc>>,
    
    /// Job completion date
    pub completed_at: Option<DateTime<Utc>>,
    
    /// Error message if the job failed
    pub error: Option<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl JobMetadata {
    /// Create a new job metadata
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            status: JobStatus::Pending,
            model_id: None,
            created_at: now,
            started_at: None,
            completed_at: None,
            error: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Mark the job as running
    pub fn mark_running(&mut self) {
        self.status = JobStatus::Running;
        self.started_at = Some(Utc::now());
    }
    
    /// Mark the job as completed
    pub fn mark_completed(&mut self) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(Utc::now());
    }
    
    /// Mark the job as failed
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = JobStatus::Failed;
        self.error = Some(error.into());
        self.completed_at = Some(Utc::now());
    }
    
    /// Mark the job as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = JobStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }
}

/// Resource usage metrics
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResourceUsage {
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,
    
    /// Memory usage in bytes
    pub memory_bytes: u64,
    
    /// Disk I/O in bytes
    pub disk_io_bytes: u64,
    
    /// Network I/O in bytes
    pub network_io_bytes: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_time_ms: 0,
            memory_bytes: 0,
            disk_io_bytes: 0,
            network_io_bytes: 0,
        }
    }
} 