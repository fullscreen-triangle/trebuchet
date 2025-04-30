// Import the WASM module once it's built
import init, { run_model } from '../pkg';

// Initialize the WASM module
async function initWasm() {
  try {
    await init();
    console.log('WASM module initialized successfully');
    setupUI();
  } catch (error) {
    console.error('Failed to initialize WASM module:', error);
    document.getElementById('model-selector').innerHTML = `
      <div class="alert alert-danger">
        Failed to initialize WASM module: ${error.message}
      </div>
    `;
  }
}

// Set up the UI components
function setupUI() {
  setupModelSelector();
  setupPerformanceMetrics();
}

// Set up the model selector component
function setupModelSelector() {
  const modelSelector = document.getElementById('model-selector');
  
  modelSelector.innerHTML = `
    <div class="mb-3">
      <label for="model-type" class="form-label">Model Type</label>
      <select class="form-select" id="model-type">
        <option value="audio">Audio Processing</option>
        <option value="genomics">Genomics Analysis</option>
        <option value="nlp">Natural Language Processing</option>
      </select>
    </div>
    
    <div class="mb-3">
      <label for="model-variant" class="form-label">Model Variant</label>
      <select class="form-select" id="model-variant">
        <option value="small">Small (Fast)</option>
        <option value="medium" selected>Medium (Balanced)</option>
        <option value="large">Large (Accurate)</option>
      </select>
    </div>
    
    <button id="run-model" class="btn btn-primary">Run Model</button>
  `;
  
  document.getElementById('run-model').addEventListener('click', runSelectedModel);
}

// Set up the performance metrics component
function setupPerformanceMetrics() {
  const metricsContainer = document.getElementById('performance-metrics');
  
  metricsContainer.innerHTML = `
    <div class="row">
      <div class="col-md-4">
        <div class="card">
          <div class="card-body text-center">
            <h5 class="card-title">Latency</h5>
            <p class="card-text" id="latency-metric">-</p>
          </div>
        </div>
      </div>
      
      <div class="col-md-4">
        <div class="card">
          <div class="card-body text-center">
            <h5 class="card-title">Memory Usage</h5>
            <p class="card-text" id="memory-metric">-</p>
          </div>
        </div>
      </div>
      
      <div class="col-md-4">
        <div class="card">
          <div class="card-body text-center">
            <h5 class="card-title">Throughput</h5>
            <p class="card-text" id="throughput-metric">-</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Run the selected model
async function runSelectedModel() {
  const modelType = document.getElementById('model-type').value;
  const modelVariant = document.getElementById('model-variant').value;
  
  document.getElementById('run-model').disabled = true;
  document.getElementById('run-model').textContent = 'Running...';
  
  try {
    const startTime = performance.now();
    
    // Call the WASM function
    const result = await run_model(modelType, modelVariant);
    
    const endTime = performance.now();
    const latency = endTime - startTime;
    
    // Update metrics
    document.getElementById('latency-metric').textContent = `${latency.toFixed(2)} ms`;
    document.getElementById('memory-metric').textContent = result.memory_usage;
    document.getElementById('throughput-metric').textContent = result.throughput;
    
  } catch (error) {
    console.error('Error running model:', error);
    alert(`Error running model: ${error.message}`);
  } finally {
    document.getElementById('run-model').disabled = false;
    document.getElementById('run-model').textContent = 'Run Model';
  }
}

// Initialize the application
initWasm(); 