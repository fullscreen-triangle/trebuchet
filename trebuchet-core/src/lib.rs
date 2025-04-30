use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

pub mod config;
pub mod error;
pub mod models;

/// Re-export common types and functions
pub use config::Config;
pub use error::{Error, Result};

/// Trebuchet version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Represents a Trebuchet instance that manages services and resources
#[derive(Debug)]
pub struct Trebuchet {
    config: Arc<RwLock<Config>>,
    // Add more fields here as needed
}

impl Trebuchet {
    /// Create a new Trebuchet instance with the given configuration
    pub async fn new(config: Config) -> Result<Self> {
        // Initialize the Trebuchet instance
        let instance = Self {
            config: Arc::new(RwLock::new(config)),
        };

        // Add initialization logic here

        Ok(instance)
    }

    /// Get a reference to the configuration
    pub async fn config(&self) -> Result<Config> {
        let config = self.config.read().await;
        Ok(config.clone())
    }

    /// Start the Trebuchet instance and its services
    pub async fn start(&self) -> Result<()> {
        // Add startup logic here
        Ok(())
    }

    /// Stop the Trebuchet instance and its services
    pub async fn stop(&self) -> Result<()> {
        // Add shutdown logic here
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trebuchet_new() {
        let config = Config::default();
        let trebuchet = Trebuchet::new(config).await.unwrap();
        
        // Basic test to check that Trebuchet instance can be created
        assert!(trebuchet.config.read().await.environment.is_some());
    }
} 