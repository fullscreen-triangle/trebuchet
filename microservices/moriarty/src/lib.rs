use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_security_analyzer() {
        // Create a security analyzer
        let analyzer = SecurityAnalyzer::new();
        
        // Register a rule
        let rule = SecurityRule {
            id: "rule1".to_string(),
            name: "Failed Login".to_string(),
            description: "Detects failed login attempts".to_string(),
            event_types: vec!["login".to_string()],
            min_severity: EventSeverity::Warning,
            pattern: {
                let mut map = HashMap::new();
                map.insert("status".to_string(), "failed".to_string());
                map
            },
        };
        
        analyzer.register_rule(rule).await.unwrap();
        
        // Create an event that should match
        let event = SecurityEvent::new(
            "event1",
            1234567890,
            "192.168.1.1",
            "login",
            EventSeverity::Warning,
        ).with_data("status", "failed");
        
        // Analyze the event
        let alerts = analyzer.analyze_event(event).await.unwrap();
        
        // Verify that an alert was generated
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rule_id, "rule1");
        assert_eq!(alerts[0].event_id, "event1");
    }
    
    #[tokio::test]
    async fn test_crypto_service() {
        // Create a crypto service
        let crypto = CryptoService::new();
        
        // Register a key pair
        let key_pair = KeyPair {
            id: "key1".to_string(),
            public_key: "DUMMY PUBLIC KEY".to_string(),
            private_key: Some("DUMMY PRIVATE KEY".to_string()),
            algorithm: "RSA".to_string(),
            strength: 2048,
            metadata: HashMap::new(),
        };
        
        crypto.register_key(key_pair).await.unwrap();
        
        // Test encryption
        let data = b"Hello, world!";
        let encrypted = crypto.encrypt("key1", data).await.unwrap();
        
        // Test decryption
        let decrypted = crypto.decrypt("key1", &encrypted).await.unwrap();
        
        // Verify that decryption recovers the original data
        assert_eq!(decrypted, data);
    }
}
