use std::path::PathBuf;

use thiserror::Error;

/// Result type for Trebuchet operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for Trebuchet operations
#[derive(Debug, Error)]
pub enum Error {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(#[source] anyhow::Error),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Service errors
    #[error("Service '{service}' error: {message}")]
    ServiceError {
        service: String,
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },
    
    /// Model errors
    #[error("Model error: {0}")]
    ModelError(String),
    
    /// Communication errors
    #[error("Communication error: {0}")]
    CommunicationError(String),
    
    /// Python bridge errors
    #[error("Python bridge error: {0}")]
    PythonError(String),
    
    /// WASM frontend errors
    #[error("WASM frontend error: {0}")]
    WasmError(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Not found errors
    #[error("Not found: {0}")]
    NotFound(String),
    
    /// Permission errors
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    /// Generic errors
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Create a service error
    pub fn service_error<S, M>(service: S, message: M) -> Self
    where
        S: Into<String>,
        M: Into<String>,
    {
        Self::ServiceError {
            service: service.into(),
            message: message.into(),
            source: None,
        }
    }
    
    /// Create a service error with a source
    pub fn service_error_with_source<S, M, E>(service: S, message: M, source: E) -> Self
    where
        S: Into<String>,
        M: Into<String>,
        E: Into<anyhow::Error>,
    {
        Self::ServiceError {
            service: service.into(),
            message: message.into(),
            source: Some(source.into()),
        }
    }
    
    /// Create a not found error
    pub fn not_found<S>(message: S) -> Self
    where
        S: Into<String>,
    {
        Self::NotFound(message.into())
    }
    
    /// Create a validation error
    pub fn validation<S>(message: S) -> Self
    where
        S: Into<String>,
    {
        Self::ValidationError(message.into())
    }
} 