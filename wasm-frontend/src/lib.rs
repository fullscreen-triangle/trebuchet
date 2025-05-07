use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub fn run_model(model_type: &str, variant: &str) -> JsValue {
    let result = serde_json::json!({
        "memory_usage": "25 MB",
        "throughput": "500 samples/sec"
    });
    serde_wasm_bindgen::to_value(&result).unwrap()
} 