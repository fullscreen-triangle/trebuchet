use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Environment types for the Trebuchet application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Development,
    Testing,
    Production,
}

impl Default for Environment {
    fn default() -> Self {
        Environment::Development
    }
}

/// Logging configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            file: None,
        }
    }
}

/// Configuration for a specific service
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServiceConfig {
    pub enabled: bool,
    #[serde(flatten)]
    pub settings: HashMap<String, serde_json::Value>,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            settings: HashMap::new(),
        }
    }
}

/// Communication configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CommunicationConfig {
    pub message_bus: MessageBusConfig,
    pub api_gateway: ApiGatewayConfig,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            message_bus: MessageBusConfig::default(),
            api_gateway: ApiGatewayConfig::default(),
        }
    }
}

/// Message bus configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageBusConfig {
    pub r#type: String,
    pub buffer_size: usize,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            r#type: "tokio-channel".to_string(),
            buffer_size: 1000,
        }
    }
}

/// API Gateway configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiGatewayConfig {
    pub host: String,
    pub port: u16,
    pub cors_enabled: bool,
}

impl Default for ApiGatewayConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_enabled: true,
        }
    }
}

/// Python bridge configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PythonBridgeConfig {
    pub enabled: bool,
    pub config_path: String,
}

impl Default for PythonBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            config_path: "python-bridge/config.yaml".to_string(),
        }
    }
}

/// WASM frontend configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WasmFrontendConfig {
    pub enabled: bool,
    pub port: u16,
    pub assets_path: String,
}

impl Default for WasmFrontendConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8081,
            assets_path: "wasm-frontend/dist".to_string(),
        }
    }
}

/// Main configuration structure for Trebuchet
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub version: String,
    pub environment: Option<Environment>,
    pub logging: LoggingConfig,
    pub services: HashMap<String, ServiceConfig>,
    pub communication: CommunicationConfig,
    pub python_bridge: PythonBridgeConfig,
    pub wasm_frontend: WasmFrontendConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            environment: Some(Environment::Development),
            logging: LoggingConfig::default(),
            services: HashMap::new(),
            communication: CommunicationConfig::default(),
            python_bridge: PythonBridgeConfig::default(),
            wasm_frontend: WasmFrontendConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a YAML file
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        
        let config_str = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))
            .map_err(Error::ConfigError)?;
            
        let config: Config = serde_yaml::from_str(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
            .map_err(Error::ConfigError)?;
            
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        // Load .env file if it exists
        dotenv::dotenv().ok();
        
        // Create a default config
        let mut config = Config::default();
        
        // Update from environment variables
        if let Ok(env) = std::env::var("TREBUCHET_ENV") {
            config.environment = match env.to_lowercase().as_str() {
                "development" => Some(Environment::Development),
                "testing" => Some(Environment::Testing),
                "production" => Some(Environment::Production),
                _ => None,
            };
        }
        
        // Add more environment variable parsing here
        
        Ok(config)
    }
    
    /// Merge configurations, with the other config taking precedence
    pub fn merge(&mut self, other: &Config) {
        // Implement merging logic here
        if let Some(env) = other.environment {
            self.environment = Some(env);
        }
        
        // Merge other fields
        // ...
    }
} 