use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A record in a dataset, representing a single row or document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Unique identifier for the record
    pub id: String,
    /// Fields and their values
    pub fields: HashMap<String, Value>,
}

/// Represents different data types that can be stored in a record
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// String value
    String(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// Array of values
    Array(Vec<Value>),
    /// Null value
    Null,
}

impl Record {
    /// Create a new record with the given ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: HashMap::new(),
        }
    }

    /// Set a field value
    pub fn set_field(&mut self, key: impl Into<String>, value: Value) -> &mut Self {
        self.fields.insert(key.into(), value);
        self
    }

    /// Get a field value
    pub fn get_field(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// Check if a field exists
    pub fn has_field(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }
}

/// A dataset containing multiple records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Name of the dataset
    pub name: String,
    /// Records in the dataset
    pub records: Vec<Record>,
    /// Schema defining the fields and their types
    pub schema: Option<Schema>,
}

/// Schema defining the structure of a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Fields and their data types
    pub fields: HashMap<String, FieldType>,
}

/// Field type definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// String type
    String,
    /// Numeric type
    Number,
    /// Boolean type
    Boolean,
    /// Array type with element type
    Array(Box<FieldType>),
    /// Record/object type with nested schema
    Object(HashMap<String, FieldType>),
}

/// A transformation to apply to a dataset
pub trait Transform: Send + Sync {
    /// Apply this transformation to a dataset
    fn transform(&self, dataset: &Dataset) -> Result<Dataset>;
    
    /// Get the name of this transformation
    fn name(&self) -> &str;
}

/// Filter transformation that keeps only records matching a condition
pub struct FilterTransform {
    /// Name of this transform
    name: String,
    /// Field to filter on
    field: String,
    /// Operator to apply
    operator: FilterOperator,
    /// Value to compare against
    value: Value,
}

/// Filter operators for comparison
pub enum FilterOperator {
    /// Equal to
    Eq,
    /// Not equal to
    NotEq,
    /// Greater than
    Gt,
    /// Greater than or equal to
    GtEq,
    /// Less than
    Lt,
    /// Less than or equal to
    LtEq,
    /// Contains (for strings and arrays)
    Contains,
}

impl FilterTransform {
    /// Create a new filter transform
    pub fn new(
        name: impl Into<String>, 
        field: impl Into<String>, 
        operator: FilterOperator, 
        value: Value
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            operator,
            value,
        }
    }

    /// Check if a record matches the filter
    fn matches(&self, record: &Record) -> bool {
        let field_value = match record.get_field(&self.field) {
            Some(value) => value,
            None => return false,
        };

        match &self.operator {
            FilterOperator::Eq => self.compare_eq(field_value, &self.value),
            FilterOperator::NotEq => !self.compare_eq(field_value, &self.value),
            FilterOperator::Gt => self.compare_gt(field_value, &self.value),
            FilterOperator::GtEq => self.compare_gt(field_value, &self.value) || self.compare_eq(field_value, &self.value),
            FilterOperator::Lt => self.compare_lt(field_value, &self.value),
            FilterOperator::LtEq => self.compare_lt(field_value, &self.value) || self.compare_eq(field_value, &self.value),
            FilterOperator::Contains => self.compare_contains(field_value, &self.value),
        }
    }

    fn compare_eq(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            _ => false,
        }
    }

    fn compare_gt(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::String(a), Value::String(b)) => a > b,
            (Value::Number(a), Value::Number(b)) => a > b,
            _ => false,
        }
    }

    fn compare_lt(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::String(a), Value::String(b)) => a < b,
            (Value::Number(a), Value::Number(b)) => a < b,
            _ => false,
        }
    }

    fn compare_contains(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::String(a), Value::String(b)) => a.contains(b),
            (Value::Array(arr), val) => arr.iter().any(|item| self.compare_eq(item, val)),
            _ => false,
        }
    }
}

impl Transform for FilterTransform {
    fn transform(&self, dataset: &Dataset) -> Result<Dataset> {
        let filtered_records = dataset.records.iter()
            .filter(|record| self.matches(record))
            .cloned()
            .collect();

        Ok(Dataset {
            name: format!("{}_filtered", dataset.name),
            records: filtered_records,
            schema: dataset.schema.clone(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Map transformation that applies a function to each record
pub struct MapTransform {
    /// Name of this transform
    name: String,
    /// Output field name
    output_field: String,
    /// Input field names
    input_fields: Vec<String>,
    /// Transformation function
    transform_fn: Arc<dyn Fn(Vec<&Value>) -> Result<Value> + Send + Sync>,
}

impl MapTransform {
    /// Create a new map transform with a transformation function
    pub fn new<F>(
        name: impl Into<String>,
        output_field: impl Into<String>,
        input_fields: Vec<String>,
        transform_fn: F,
    ) -> Self
    where
        F: Fn(Vec<&Value>) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            output_field: output_field.into(),
            input_fields,
            transform_fn: Arc::new(transform_fn),
        }
    }
}

impl Transform for MapTransform {
    fn transform(&self, dataset: &Dataset) -> Result<Dataset> {
        let mut transformed_records = Vec::with_capacity(dataset.records.len());

        for record in &dataset.records {
            let mut new_record = record.clone();
            
            // Extract input values
            let input_values: Vec<&Value> = self.input_fields.iter()
                .map(|field| record.get_field(field)
                    .ok_or_else(|| anyhow!("Field '{}' not found in record", field)))
                .collect::<Result<Vec<&Value>>>()?;
            
            // Apply transformation function
            let output_value = (self.transform_fn)(input_values)?;
            
            // Set output field
            new_record.set_field(&self.output_field, output_value);
            transformed_records.push(new_record);
        }

        // Create new schema if needed
        let schema = if let Some(schema) = &dataset.schema {
            let mut new_schema = schema.clone();
            // Add output field to schema - this is simplified; in real code, you'd deduce the type
            new_schema.fields.insert(self.output_field.clone(), FieldType::String);
            Some(new_schema)
        } else {
            None
        };

        Ok(Dataset {
            name: format!("{}_mapped", dataset.name),
            records: transformed_records,
            schema,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A data pipeline consisting of multiple transformations
pub struct Pipeline {
    /// Name of the pipeline
    pub name: String,
    /// Transformations to apply in sequence
    pub transforms: Vec<Arc<dyn Transform>>,
}

impl Pipeline {
    /// Create a new pipeline with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transforms: Vec::new(),
        }
    }

    /// Add a transformation to the pipeline
    pub fn add_transform<T: Transform + 'static>(&mut self, transform: T) -> &mut Self {
        self.transforms.push(Arc::new(transform));
        self
    }

    /// Execute the pipeline on a dataset
    pub async fn execute(&self, dataset: Dataset) -> Result<Dataset> {
        let mut current = dataset;

        for transform in &self.transforms {
            current = transform.transform(&current)?;
        }

        Ok(current)
    }
}

/// Dataset loader interface
pub trait DatasetLoader: Send + Sync {
    /// Load a dataset from a source
    fn load(&self, source: &str) -> Result<Dataset>;
}

/// CSV dataset loader
pub struct CsvLoader {
    /// Delimiter to use
    delimiter: char,
    /// Whether the CSV has a header row
    has_header: bool,
}

impl CsvLoader {
    /// Create a new CSV loader
    pub fn new(delimiter: char, has_header: bool) -> Self {
        Self {
            delimiter,
            has_header,
        }
    }
}

impl DatasetLoader for CsvLoader {
    fn load(&self, source: &str) -> Result<Dataset> {
        // In a real implementation, this would read from a CSV file
        // For this example, we'll return a dummy dataset
        let mut records = Vec::new();
        
        // Create some dummy records
        for i in 0..10 {
            let mut record = Record::new(format!("record_{}", i));
            record.set_field("name", Value::String(format!("Item {}", i)))
                  .set_field("value", Value::Number(i as f64 * 10.5))
                  .set_field("active", Value::Boolean(i % 2 == 0));
            records.push(record);
        }
        
        Ok(Dataset {
            name: Path::new(source).file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            records,
            schema: None,
        })
    }
}

/// JSON dataset loader
pub struct JsonLoader;

impl DatasetLoader for JsonLoader {
    fn load(&self, source: &str) -> Result<Dataset> {
        // In a real implementation, this would read from a JSON file
        // For this example, we'll return a dummy dataset
        let mut records = Vec::new();
        
        // Create some dummy records
        for i in 0..5 {
            let mut record = Record::new(format!("json_{}", i));
            record.set_field("id", Value::Number(i as f64))
                  .set_field("name", Value::String(format!("JSON Item {}", i)))
                  .set_field("tags", Value::Array(vec![
                      Value::String("tag1".to_string()),
                      Value::String("tag2".to_string()),
                  ]));
            records.push(record);
        }
        
        Ok(Dataset {
            name: Path::new(source).file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            records,
            schema: None,
        })
    }
}

/// Dataset writer interface
pub trait DatasetWriter: Send + Sync {
    /// Write a dataset to a destination
    fn write(&self, dataset: &Dataset, destination: &str) -> Result<()>;
}

/// CSV dataset writer
pub struct CsvWriter {
    /// Delimiter to use
    delimiter: char,
    /// Whether to write a header row
    write_header: bool,
}

impl CsvWriter {
    /// Create a new CSV writer
    pub fn new(delimiter: char, write_header: bool) -> Self {
        Self {
            delimiter,
            write_header,
        }
    }
}

impl DatasetWriter for CsvWriter {
    fn write(&self, dataset: &Dataset, destination: &str) -> Result<()> {
        // In a real implementation, this would write to a CSV file
        // For this example, we'll just log the operation
        println!("Writing dataset '{}' with {} records to CSV file '{}'", 
            dataset.name, dataset.records.len(), destination);
        Ok(())
    }
}

/// Data pipeline service that manages pipelines and executes them
pub struct DataPipelineService {
    /// Available pipelines
    pipelines: RwLock<HashMap<String, Arc<Pipeline>>>,
    /// Available dataset loaders
    loaders: RwLock<HashMap<String, Arc<dyn DatasetLoader>>>,
    /// Available dataset writers
    writers: RwLock<HashMap<String, Arc<dyn DatasetWriter>>>,
}

impl DataPipelineService {
    /// Create a new data pipeline service
    pub fn new() -> Self {
        let mut loaders = HashMap::new();
        loaders.insert("csv".to_string(), Arc::new(CsvLoader::new(',', true)) as Arc<dyn DatasetLoader>);
        loaders.insert("json".to_string(), Arc::new(JsonLoader) as Arc<dyn DatasetLoader>);
        
        let mut writers = HashMap::new();
        writers.insert("csv".to_string(), Arc::new(CsvWriter::new(',', true)) as Arc<dyn DatasetWriter>);
        
        Self {
            pipelines: RwLock::new(HashMap::new()),
            loaders: RwLock::new(loaders),
            writers: RwLock::new(writers),
        }
    }
    
    /// Register a pipeline
    pub async fn register_pipeline(&self, pipeline: Pipeline) -> Result<()> {
        let mut pipelines = self.pipelines.write().await;
        pipelines.insert(pipeline.name.clone(), Arc::new(pipeline));
        Ok(())
    }
    
    /// Register a dataset loader
    pub async fn register_loader(&self, name: impl Into<String>, loader: impl DatasetLoader + 'static) -> Result<()> {
        let mut loaders = self.loaders.write().await;
        loaders.insert(name.into(), Arc::new(loader) as Arc<dyn DatasetLoader>);
        Ok(())
    }
    
    /// Register a dataset writer
    pub async fn register_writer(&self, name: impl Into<String>, writer: impl DatasetWriter + 'static) -> Result<()> {
        let mut writers = self.writers.write().await;
        writers.insert(name.into(), Arc::new(writer) as Arc<dyn DatasetWriter>);
        Ok(())
    }
    
    /// Execute a pipeline on a dataset
    pub async fn execute_pipeline(
        &self, 
        pipeline_name: &str, 
        source: &str, 
        source_type: &str,
        destination: Option<(&str, &str)>,
    ) -> Result<Dataset> {
        // Get the pipeline
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines.get(pipeline_name)
            .ok_or_else(|| anyhow!("Pipeline '{}' not found", pipeline_name))?
            .clone();
            
        // Get the loader
        let loaders = self.loaders.read().await;
        let loader = loaders.get(source_type)
            .ok_or_else(|| anyhow!("Loader '{}' not found", source_type))?;
            
        // Load the dataset
        let dataset = loader.load(source)?;
        
        // Execute the pipeline
        let result = pipeline.execute(dataset).await?;
        
        // Write the result if requested
        if let Some((dest, dest_type)) = destination {
            let writers = self.writers.read().await;
            let writer = writers.get(dest_type)
                .ok_or_else(|| anyhow!("Writer '{}' not found", dest_type))?;
                
            writer.write(&result, dest)?;
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_operations() {
        let mut record = Record::new("test_record");
        record.set_field("name", Value::String("Test Item".to_string()))
              .set_field("value", Value::Number(42.5))
              .set_field("active", Value::Boolean(true));
              
        assert_eq!(record.id, "test_record");
        assert!(record.has_field("name"));
        assert!(record.has_field("value"));
        assert!(record.has_field("active"));
        assert!(!record.has_field("nonexistent"));
        
        match record.get_field("name") {
            Some(Value::String(s)) => assert_eq!(s, "Test Item"),
            _ => panic!("Expected String value for 'name'"),
        }
        
        match record.get_field("value") {
            Some(Value::Number(n)) => assert_eq!(*n, 42.5),
            _ => panic!("Expected Number value for 'value'"),
        }
        
        match record.get_field("active") {
            Some(Value::Boolean(b)) => assert!(*b),
            _ => panic!("Expected Boolean value for 'active'"),
        }
    }

    #[test]
    fn test_filter_transform() {
        // Create a dataset with test records
        let mut records = Vec::new();
        for i in 0..5 {
            let mut record = Record::new(format!("record_{}", i));
            record.set_field("value", Value::Number(i as f64));
            records.push(record);
        }
        
        let dataset = Dataset {
            name: "test_dataset".to_string(),
            records,
            schema: None,
        };
        
        // Create a filter transform to keep only records with value >= 2
        let filter = FilterTransform::new(
            "value_filter",
            "value",
            FilterOperator::GtEq,
            Value::Number(2.0),
        );
        
        // Apply the transform
        let result = filter.transform(&dataset).unwrap();
        
        // Check that only records with value >= 2 remain
        assert_eq!(result.records.len(), 3);
        for record in &result.records {
            match record.get_field("value") {
                Some(Value::Number(n)) => assert!(*n >= 2.0),
                _ => panic!("Expected Number value for 'value'"),
            }
        }
    }

    #[test]
    fn test_map_transform() {
        // Create a dataset with test records
        let mut records = Vec::new();
        for i in 0..3 {
            let mut record = Record::new(format!("record_{}", i));
            record.set_field("value", Value::Number(i as f64));
            records.push(record);
        }
        
        let dataset = Dataset {
            name: "test_dataset".to_string(),
            records,
            schema: None,
        };
        
        // Create a map transform to double the value
        let map = MapTransform::new(
            "double_value",
            "doubled",
            vec!["value".to_string()],
            |values| {
                if let Some(Value::Number(n)) = values.get(0) {
                    Ok(Value::Number(n * 2.0))
                } else {
                    Err(anyhow!("Expected Number value"))
                }
            },
        );
        
        // Apply the transform
        let result = map.transform(&dataset).unwrap();
        
        // Check that doubled values were added
        assert_eq!(result.records.len(), 3);
        for (i, record) in result.records.iter().enumerate() {
            match record.get_field("doubled") {
                Some(Value::Number(n)) => assert_eq!(*n, (i as f64) * 2.0),
                _ => panic!("Expected Number value for 'doubled'"),
            }
        }
    }
}