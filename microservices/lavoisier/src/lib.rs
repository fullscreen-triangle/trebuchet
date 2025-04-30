use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents a chemical element in the periodic table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// Atomic symbol (e.g., "H", "He", "Li")
    pub symbol: String,
    /// Element name (e.g., "Hydrogen", "Helium", "Lithium")
    pub name: String,
    /// Atomic number
    pub atomic_number: u32,
    /// Atomic weight (in atomic mass units)
    pub atomic_weight: f64,
    /// Element category (e.g., "Nonmetal", "Noble gas", "Alkali metal")
    pub category: String,
    /// Electron configuration
    pub electron_configuration: String,
    /// Properties of the element
    pub properties: HashMap<String, Value>,
}

/// Represents different types of values for element properties
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
}

/// Represents a chemical compound
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compound {
    /// Unique identifier for the compound
    pub id: String,
    /// Name of the compound
    pub name: String,
    /// Chemical formula
    pub formula: String,
    /// Molecular weight
    pub molecular_weight: f64,
    /// Element composition (element symbol to count)
    pub composition: HashMap<String, u32>,
    /// Properties of the compound
    pub properties: HashMap<String, Value>,
}

impl Element {
    /// Create a new element
    pub fn new(
        symbol: impl Into<String>,
        name: impl Into<String>,
        atomic_number: u32,
        atomic_weight: f64,
        category: impl Into<String>,
        electron_configuration: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            name: name.into(),
            atomic_number,
            atomic_weight,
            category: category.into(),
            electron_configuration: electron_configuration.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property to the element
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

impl Compound {
    /// Create a new compound
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        formula: impl Into<String>,
        molecular_weight: f64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            formula: formula.into(),
            molecular_weight,
            composition: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    /// Add an element to the compound composition
    pub fn with_element(mut self, symbol: impl Into<String>, count: u32) -> Self {
        self.composition.insert(symbol.into(), count);
        self
    }

    /// Add a property to the compound
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Get the count of a specific element in the compound
    pub fn get_element_count(&self, symbol: &str) -> u32 {
        *self.composition.get(symbol).unwrap_or(&0)
    }
}

/// Service for managing elements and compounds
pub struct ChemistryService {
    /// Element database
    elements: RwLock<HashMap<String, Element>>,
    /// Compound database
    compounds: RwLock<HashMap<String, Compound>>,
}

impl ChemistryService {
    /// Create a new chemistry service
    pub fn new() -> Self {
        Self {
            elements: RwLock::new(HashMap::new()),
            compounds: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize with periodic table elements
    pub async fn initialize_periodic_table(&self) -> Result<()> {
        let mut elements = self.elements.write().await;

        // Add a few common elements
        let hydrogen = Element::new(
            "H",
            "Hydrogen",
            1,
            1.008,
            "Nonmetal",
            "1s1",
        ).with_property("electronegativity", Value::Number(2.2))
         .with_property("melting_point", Value::Number(-259.16))
         .with_property("boiling_point", Value::Number(-252.87));
        
        let helium = Element::new(
            "He",
            "Helium",
            2,
            4.0026,
            "Noble gas",
            "1s2",
        ).with_property("electronegativity", Value::Number(0.0))
         .with_property("melting_point", Value::Number(-272.2))
         .with_property("boiling_point", Value::Number(-268.93));
        
        let carbon = Element::new(
            "C",
            "Carbon",
            6,
            12.011,
            "Nonmetal",
            "1s2 2s2 2p2",
        ).with_property("electronegativity", Value::Number(2.55))
         .with_property("melting_point", Value::Number(3550.0))
         .with_property("boiling_point", Value::Number(4027.0));
        
        let oxygen = Element::new(
            "O",
            "Oxygen",
            8,
            15.999,
            "Nonmetal",
            "1s2 2s2 2p4",
        ).with_property("electronegativity", Value::Number(3.44))
         .with_property("melting_point", Value::Number(-218.79))
         .with_property("boiling_point", Value::Number(-182.96));

        elements.insert(hydrogen.symbol.clone(), hydrogen);
        elements.insert(helium.symbol.clone(), helium);
        elements.insert(carbon.symbol.clone(), carbon);
        elements.insert(oxygen.symbol.clone(), oxygen);

        Ok(())
    }

    /// Add a new element
    pub async fn add_element(&self, element: Element) -> Result<()> {
        let mut elements = self.elements.write().await;
        elements.insert(element.symbol.clone(), element);
        Ok(())
    }

    /// Get an element by symbol
    pub async fn get_element(&self, symbol: &str) -> Result<Element> {
        let elements = self.elements.read().await;
        elements.get(symbol)
            .cloned()
            .ok_or_else(|| anyhow!("Element '{}' not found", symbol))
    }

    /// Add a new compound
    pub async fn add_compound(&self, compound: Compound) -> Result<()> {
        // Validate elements in composition
        let elements = self.elements.read().await;
        for symbol in compound.composition.keys() {
            if !elements.contains_key(symbol) {
                return Err(anyhow!("Element '{}' not found in periodic table", symbol));
            }
        }

        let mut compounds = self.compounds.write().await;
        compounds.insert(compound.id.clone(), compound);
        Ok(())
    }

    /// Get a compound by ID
    pub async fn get_compound(&self, id: &str) -> Result<Compound> {
        let compounds = self.compounds.read().await;
        compounds.get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Compound '{}' not found", id))
    }

    /// Calculate molecular weight of a formula
    pub async fn calculate_molecular_weight(&self, formula: &str) -> Result<f64> {
        let elements = self.elements.read().await;
        let parsed = self.parse_formula(formula)?;
        
        let mut weight = 0.0;
        for (symbol, count) in parsed {
            let element = elements.get(&symbol)
                .ok_or_else(|| anyhow!("Element '{}' not found in periodic table", symbol))?;
                
            weight += element.atomic_weight * count as f64;
        }
        
        Ok(weight)
    }
    
    /// Parse a chemical formula into element counts
    fn parse_formula(&self, formula: &str) -> Result<HashMap<String, u32>> {
        // This is a simplified parser that works for basic formulas like H2O, CO2, etc.
        // A real implementation would handle more complex formulas with brackets, hydrates, etc.
        let mut result = HashMap::new();
        let mut i = 0;
        let chars: Vec<char> = formula.chars().collect();
        
        while i < chars.len() {
            if !chars[i].is_ascii_uppercase() {
                return Err(anyhow!("Formula must start with uppercase letter"));
            }
            
            // Extract the element symbol (1 or 2 characters)
            let mut symbol = chars[i].to_string();
            i += 1;
            
            if i < chars.len() && chars[i].is_ascii_lowercase() {
                symbol.push(chars[i]);
                i += 1;
            }
            
            // Extract the count (optional)
            let mut count_str = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                count_str.push(chars[i]);
                i += 1;
            }
            
            let count = if count_str.is_empty() { 1 } else { count_str.parse::<u32>()? };
            
            // Add to the result
            *result.entry(symbol).or_insert(0) += count;
        }
        
        Ok(result)
    }
    
    /// Balance a chemical equation
    pub async fn balance_equation(&self, equation: &str) -> Result<String> {
        // In a real implementation, this would implement a matrix-based algorithm
        // For this example, we'll handle a few common cases
        
        match equation {
            "H2 + O2 -> H2O" => Ok("2 H2 + O2 -> 2 H2O".to_string()),
            "C + O2 -> CO2" => Ok("C + O2 -> CO2".to_string()),
            "CH4 + O2 -> CO2 + H2O" => Ok("CH4 + 2 O2 -> CO2 + 2 H2O".to_string()),
            _ => Err(anyhow!("Cannot balance equation: {}", equation)),
        }
    }
}

/// Chemical reaction simulator
pub struct ReactionSimulator {
    /// Reference to the chemistry service
    chemistry: Arc<ChemistryService>,
}

/// Reaction result with products and energy change
#[derive(Debug, Serialize, Deserialize)]
pub struct ReactionResult {
    /// Input reactants
    pub reactants: Vec<(String, f64)>, // (compound_id, moles)
    /// Output products
    pub products: Vec<(String, f64)>, // (compound_id, moles)
    /// Energy change in kJ/mol
    pub energy_change: f64,
    /// Whether the reaction is endothermic
    pub is_endothermic: bool,
}

impl ReactionSimulator {
    /// Create a new reaction simulator
    pub fn new(chemistry: Arc<ChemistryService>) -> Self {
        Self { chemistry }
    }
    
    /// Simulate a reaction between compounds
    pub async fn simulate_reaction(
        &self,
        reactant_ids: &[&str],
        reactant_moles: &[f64],
    ) -> Result<ReactionResult> {
        if reactant_ids.len() != reactant_moles.len() {
            return Err(anyhow!("Mismatch between reactants and moles"));
        }
        
        // For this example, we'll simulate some simple reactions
        // In a real implementation, this would use a reaction database or predict reactions
        
        // Water formation: 2 H2 + O2 -> 2 H2O
        if reactant_ids.contains(&"H2") && reactant_ids.contains(&"O2") {
            // Find indexes
            let h2_idx = reactant_ids.iter().position(|&id| id == "H2").unwrap();
            let o2_idx = reactant_ids.iter().position(|&id| id == "O2").unwrap();
            
            let h2_moles = reactant_moles[h2_idx];
            let o2_moles = reactant_moles[o2_idx];
            
            // Determine limiting reactant
            let h2_adjusted = h2_moles / 2.0; // 2 moles of H2 needed
            let limiting = h2_adjusted.min(o2_moles);
            
            // Calculate products
            let h2o_moles = limiting * 2.0; // 2 moles of H2O produced
            
            // Calculate remaining reactants
            let h2_remaining = h2_moles - (limiting * 2.0);
            let o2_remaining = o2_moles - limiting;
            
            // Create reaction result
            let mut reactants = Vec::new();
            let mut products = Vec::new();
            
            // Only add non-zero amounts
            if h2_remaining > 0.0 {
                reactants.push(("H2".to_string(), h2_remaining));
            }
            
            if o2_remaining > 0.0 {
                reactants.push(("O2".to_string(), o2_remaining));
            }
            
            products.push(("H2O".to_string(), h2o_moles));
            
            return Ok(ReactionResult {
                reactants,
                products,
                energy_change: -285.8 * h2o_moles, // kJ/mol for water formation
                is_endothermic: false,
            });
        }
        
        // For other combinations, return an error
        Err(anyhow!("No known reaction for the given reactants"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_element_operations() {
        let service = ChemistryService::new();
        
        // Add an element
        let element = Element::new(
            "N",
            "Nitrogen",
            7,
            14.007,
            "Nonmetal",
            "1s2 2s2 2p3",
        );
        
        service.add_element(element).await.unwrap();
        
        // Retrieve the element
        let retrieved = service.get_element("N").await.unwrap();
        assert_eq!(retrieved.symbol, "N");
        assert_eq!(retrieved.name, "Nitrogen");
        assert_eq!(retrieved.atomic_number, 7);
    }

    #[tokio::test]
    async fn test_compound_operations() {
        let service = ChemistryService::new();
        
        // Initialize periodic table to have required elements
        service.initialize_periodic_table().await.unwrap();
        
        // Create a compound
        let compound = Compound::new(
            "water",
            "Water",
            "H2O",
            18.015,
        )
        .with_element("H", 2)
        .with_element("O", 1)
        .with_property("melting_point", Value::Number(0.0))
        .with_property("boiling_point", Value::Number(100.0));
        
        service.add_compound(compound).await.unwrap();
        
        // Retrieve the compound
        let retrieved = service.get_compound("water").await.unwrap();
        assert_eq!(retrieved.formula, "H2O");
        assert_eq!(retrieved.get_element_count("H"), 2);
        assert_eq!(retrieved.get_element_count("O"), 1);
    }

    #[tokio::test]
    async fn test_molecular_weight_calculation() {
        let service = ChemistryService::new();
        service.initialize_periodic_table().await.unwrap();
        
        let weight = service.calculate_molecular_weight("H2O").await.unwrap();
        // Should be close to 18.015
        assert!((weight - 18.015).abs() < 0.01);
    }
}